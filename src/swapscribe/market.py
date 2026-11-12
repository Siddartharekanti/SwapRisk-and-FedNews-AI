"""Market data and a deliberately simplified yield-curve proxy.

US Treasury par yields are NOT swap/OIS discount curves. We interpolate them as
zero rates only to demonstrate sensitivity and hedge calculations. This is not
market-consistent derivatives pricing or executable trading advice.
"""
from __future__ import annotations
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

TENORS = {"3 Mo": 0.25, "6 Mo": 0.5, "1 Yr": 1., "2 Yr": 2.,
          "3 Yr": 3., "5 Yr": 5., "7 Yr": 7., "10 Yr": 10.,
          "20 Yr": 20., "30 Yr": 30.}
TENOR_VALUES = np.array(list(TENORS.values()), dtype=float)
TREASURY_URL = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
    "daily-treasury-rates.csv/{year}/all?type=daily_treasury_yield_curve"
    "&field_tdr_date_value={year}&page&_format=csv"
)


def clean_treasury_csv(raw: str) -> pd.DataFrame:
    """Normalize official Treasury CSV; keep USD yield percentages, not decimals."""
    df = pd.read_csv(StringIO(raw), dtype=str)
    df.columns = df.columns.str.strip()
    date_col = next((x for x in ["Date", "DATE", "date"] if x in df), None)
    if date_col is None:
        raise ValueError("Treasury CSV has no Date column")
    out = pd.DataFrame({"date": pd.to_datetime(df[date_col], errors="coerce")})
    for col in TENORS:
        out[col] = pd.to_numeric(df[col].str.replace(",", "", regex=False), errors="coerce") if col in df else np.nan
    out = out.dropna(subset=["date", "2 Yr", "10 Yr"]).sort_values("date")
    out = out.drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    if out.empty:
        raise ValueError("No usable Treasury yield observations")
    for col in TENORS:
        if col in out and (out[col].dropna().abs() > 40).any():
            raise ValueError(f"Implausible yield in {col}: check units and source")
    return out


def fetch_treasury(year: int, dest: Path, timeout: int = 30) -> Path:
    """Download one year directly from U.S. Treasury; fail clearly on rate limits."""
    if year < 1990 or year > pd.Timestamp.utcnow().year:
        raise ValueError("Year outside supported Treasury range")
    url = TREASURY_URL.format(year=year)
    response = requests.get(url, timeout=timeout, headers={"User-Agent": "SwapRisk-and-FedNews-AI/1.0"})
    response.raise_for_status()
    data = clean_treasury_csv(response.text)
    dest.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(dest, index=False)
    return dest


def load_rates(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    if df.empty or not {"date", "2 Yr", "10 Yr"}.issubset(df.columns):
        raise ValueError("Missing or empty market-rate data")
    df = df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    if df[["2 Yr", "10 Yr"]].isna().any().any():
        raise ValueError("Rate data must have complete 2Y and 10Y tenors")
    return df


def bootstrap_zero_rates(tenors: np.ndarray, par_rates: np.ndarray) -> np.ndarray:
    """Standard par-to-zero bootstrapping under semiannual coupon convention."""
    zero_rates = np.zeros_like(par_rates, dtype=float)
    for i, (T, y) in enumerate(zip(tenors, par_rates)):
        if T <= 1.0:
            zero_rates[i] = y
        else:
            t_coupons = np.arange(0.5, T, 0.5)
            z_coupons = np.interp(t_coupons, tenors[:i], zero_rates[:i])
            df_coupons = np.exp(-z_coupons * t_coupons)
            pv_coupons = 0.5 * y * float(np.sum(df_coupons))
            df_T = (1.0 - pv_coupons) / (1.0 + 0.5 * y)
            zero_rates[i] = -np.log(max(df_T, 1e-6)) / T
    return zero_rates


@dataclass(frozen=True)
class YieldCurve:
    """Exact bootstrapped zero-rate curve from par Treasury yields."""
    tenors: np.ndarray
    rates: np.ndarray  # bootstrapped zero rates (decimal per annum)
    par_rates: np.ndarray = None  # input par yields (decimal per annum)

    def __post_init__(self):
        t = np.asarray(self.tenors, dtype=float)
        p = np.asarray(self.par_rates if self.par_rates is not None else self.rates, dtype=float)
        if len(t) < 2 or len(t) != len(p) or np.any(np.diff(t) <= 0):
            raise ValueError("Curve tenors must be ascending and same length as rates")
        if np.any(~np.isfinite(p)) or np.any(~np.isfinite(t)) or np.any(t <= 0):
            raise ValueError("Nonfinite rate or invalid tenor")
        z = bootstrap_zero_rates(t, p)
        object.__setattr__(self, "tenors", t.copy())
        object.__setattr__(self, "par_rates", p.copy())
        object.__setattr__(self, "rates", z.copy())

    def rate(self, times):
        return np.interp(times, self.tenors, self.rates, left=self.rates[0], right=self.rates[-1])

    def par_rate(self, times):
        return np.interp(times, self.tenors, self.par_rates, left=self.par_rates[0], right=self.par_rates[-1])

    def discount(self, times):
        t = np.asarray(times, dtype=float)
        return np.exp(-self.rate(t) * t)

    def bump_parallel(self, basis_points: float) -> "YieldCurve":
        new_par = self.par_rates + basis_points / 10000
        return YieldCurve(self.tenors, new_par, par_rates=new_par)

    def bump_key(self, tenor: float, basis_points: float) -> "YieldCurve":
        idx = np.flatnonzero(np.isclose(self.tenors, tenor))
        if len(idx) != 1:
            raise ValueError(f"Tenor {tenor} not on curve nodes")
        new_par = self.par_rates.copy()
        new_par[idx[0]] += basis_points / 10000
        return YieldCurve(self.tenors, new_par, par_rates=new_par)


def curve_from_row(row: pd.Series) -> YieldCurve:
    tenors, rates = [], []
    for label, tenor in TENORS.items():
        value = pd.to_numeric(row.get(label, np.nan), errors="coerce")
        if pd.notna(value):
            tenors.append(tenor)
            rates.append(float(value) / 100)
    if len(tenors) < 3:
        raise ValueError("At least three available yield tenors required")
    return YieldCurve(np.array(tenors), np.array(rates))


def history_key_rate_changes(df: pd.DataFrame, key_tenors=(1., 2., 3., 5., 7., 10., 20.), max_rows=260) -> np.ndarray:
    """Business-day tenor yield differences, measured in basis points."""
    columns = [next((k for k, v in TENORS.items() if v == t), None) for t in key_tenors]
    if any(c is None or c not in df for c in columns):
        raise ValueError("Missing chosen key-rate series")
    series = df[columns].astype(float).dropna().tail(max_rows + 1)
    changes = series.diff().dropna().to_numpy() * 100  # percent -> bp
    if len(changes) < 5:
        raise ValueError("Need >=6 dated observations for scenario covariance")
    return changes
