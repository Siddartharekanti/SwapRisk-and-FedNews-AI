"""Illustrative fixed-for-floating vanilla USD swaps and hedge optimization.

Assumptions: valuation at reset, no accrued cashflows, no convexity/credit,
no multi-curve SOFR forward calibration, semiannual equal-year fractions, and
Treasury par yields approximated as continuous discount zero rates. Educational.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

from .market import YieldCurve, curve_from_row, history_key_rate_changes


@dataclass(frozen=True)
class Swap:
    swap_id: str
    notional_usd: float
    maturity_years: int
    fixed_rate_pct: float
    direction: str = "pay_fixed"

    def __post_init__(self):
        if self.notional_usd <= 0 or not isinstance(self.maturity_years, int) or self.maturity_years <= 0:
            raise ValueError("Swap notional and integer maturity must be positive")
        if self.direction not in ("pay_fixed", "receive_fixed"):
            raise ValueError("direction must be pay_fixed or receive_fixed")
        if abs(self.fixed_rate_pct) > 50:
            raise ValueError("Fixed rate is a percentage, e.g. 4.25")


def par_rate(curve: YieldCurve, maturity_years: int) -> float:
    """Par fixed rate (decimal) in illustrative single-curve model."""
    ts = np.arange(0.5, float(maturity_years) + 0.001, 0.5)
    annuity = 0.5 * float(np.sum(curve.discount(ts)))
    return (1. - float(curve.discount(float(maturity_years)))) / annuity


def price_swap(swap: Swap, curve: YieldCurve) -> float:
    ts = np.arange(0.5, float(swap.maturity_years) + 0.001, 0.5)
    annuity = 0.5 * float(np.sum(curve.discount(ts)))
    floating_leg = swap.notional_usd * (1. - float(curve.discount(float(swap.maturity_years))))
    fixed_leg = swap.notional_usd * (swap.fixed_rate_pct / 100.) * annuity
    sign = 1. if swap.direction == "pay_fixed" else -1.
    return sign * (floating_leg - fixed_leg)


def load_portfolio(path: Path) -> list[Swap]:
    df = pd.read_csv(path)
    fields = {"swap_id", "notional_usd", "maturity_years", "fixed_rate_pct", "direction"}
    if not fields.issubset(df.columns) or df.empty:
        raise ValueError(f"Portfolio requires {sorted(fields)} columns")
    if df["swap_id"].duplicated().any():
        raise ValueError("Duplicate swap IDs")
    return [Swap(str(row.swap_id), float(row.notional_usd), int(row.maturity_years),
                 float(row.fixed_rate_pct), str(row.direction)) for row in df.itertuples(index=False)]


def portfolio_pv(portfolio: list[Swap], curve: YieldCurve) -> float:
    return float(sum(price_swap(s, curve) for s in portfolio))


def parallel_dv01(portfolio: list[Swap], curve: YieldCurve) -> float:
    """Signed portfolio PV change for a +1bp parallel curve move, USD/bp."""
    return (portfolio_pv(portfolio, curve.bump_parallel(0.5)) -
            portfolio_pv(portfolio, curve.bump_parallel(-0.5)))


def key_rate_pv_changes(portfolio: list[Swap], curve: YieldCurve, tenors=(1., 2., 3., 5., 7., 10., 20.)) -> np.ndarray:
    """Signed USD PV change per +1bp shock at each curve node."""
    return np.asarray([(portfolio_pv(portfolio, curve.bump_key(t, 0.5)) -
                        portfolio_pv(portfolio, curve.bump_key(t, -0.5))) for t in tenors], dtype=float)


def hedge_instruments(curve: YieldCurve, maturities=(2, 5, 10), unit_usd=1_000_000.) -> list[Swap]:
    """Unit-notional at-par pay-fixed reference hedges; sign of solution may reverse."""
    return [Swap(f"HEDGE_{m}Y", unit_usd, int(m), 100 * par_rate(curve, int(m)), "pay_fixed")
            for m in maturities]


def optimize_hedge(portfolio: list[Swap], curve: YieldCurve, historical_changes_bp: np.ndarray,
                   tenors=(1., 2., 3., 5., 7., 10., 20.), hedge_maturities=(2, 5, 10), max_abs_millions=30., ridge=200.) -> dict:
    """Bounded weighted least-squares hedging on historical + prescribed stresses.

    x is signed notional in millions of USD (negative = receive fixed).
    Minimizes scenario P/L RMS and an L2 proxy for position/concentration costs.
    Ridge is in USD-per-million units after scenario normalization.
    """
    historical_changes_bp = np.asarray(historical_changes_bp, dtype=float)
    if historical_changes_bp.ndim != 2 or historical_changes_bp.shape[1] != len(tenors):
        raise ValueError("Historical shocks shape must be (observations, key tenors)")
    if len(historical_changes_bp) < 5 or np.any(~np.isfinite(historical_changes_bp)):
        raise ValueError("Invalid shock history")
    if max_abs_millions <= 0 or ridge < 0:
        raise ValueError("Invalid optimizer controls")

    r = key_rate_pv_changes(portfolio, curve, tenors)
    hedges = hedge_instruments(curve, hedge_maturities)
    h = np.column_stack([key_rate_pv_changes([inst], curve, tenors) for inst in hedges])
    # Historical one-day yield shocks + intentionally distinct stress scenarios.
    stress = np.concatenate([np.eye(len(tenors)) * 100, -np.eye(len(tenors)) * 100], axis=0)
    scenarios = np.vstack([historical_changes_bp, stress])
    a = scenarios @ r
    b = scenarios @ h
    scale = float(np.sqrt(len(scenarios)))
    design = np.vstack([b / scale, np.sqrt(ridge) * np.eye(len(hedges))])
    target = np.r_[-a / scale, np.zeros(len(hedges))]
    solution = lsq_linear(design, target, bounds=(-max_abs_millions, max_abs_millions), tol=1e-11, max_iter=250)
    if not solution.success:
        raise RuntimeError("Hedge optimizer failed: " + solution.message)
    x = solution.x
    after_sensitivity = r + h @ x
    before = scenarios @ r
    after = scenarios @ after_sensitivity
    return {
        "tenors_years": list(map(float, tenors)),
        "unhedged_key_rate_usd_per_bp": r.tolist(),
        "hedged_key_rate_usd_per_bp": after_sensitivity.tolist(),
        "hedge_unit_names": [s.swap_id for s in hedges],
        "hedge_notional_usd_millions": x.tolist(),
        "hedge_sign_convention": "+ pay fixed; - receive fixed",
        "rms_scenario_pnl_usd_before": float(np.sqrt(np.mean(before ** 2))),
        "rms_scenario_pnl_usd_after": float(np.sqrt(np.mean(after ** 2))),
        "worst_scenario_pnl_usd_before": float(np.min(before)),
        "worst_scenario_pnl_usd_after": float(np.min(after)),
        "historical_scenarios": int(len(historical_changes_bp)),
        "stress_scenarios": int(len(stress)),
        "hedge_maturities_years": list(map(int, hedge_maturities)),
        "optimizer_cost_proxy": "ridge L2, not actual market bid-ask quotes",
        "curve_model": "Treasury par yield interpolated as zero rate (educational proxy; NOT SOFR/OIS)",
    }


def risk_report(rates: pd.DataFrame, portfolio: list[Swap], as_of=None, max_abs_millions=30.) -> dict:
    rates = rates.copy().sort_values("date")
    if as_of is not None:
        rates = rates[rates.date <= pd.Timestamp(as_of)]
    if len(rates) < 7:
        raise ValueError("Need >=7 rate observations on or before valuation date")
    curve = curve_from_row(rates.iloc[-1])
    shock_history = history_key_rate_changes(rates, key_tenors=(1., 2., 3., 5., 7., 10., 20.))
    result = optimize_hedge(portfolio, curve, shock_history, max_abs_millions=max_abs_millions)
    result.update({
        "as_of": pd.Timestamp(rates.iloc[-1].date).strftime("%Y-%m-%d"),
        "portfolio_pv_usd": portfolio_pv(portfolio, curve),
        "parallel_dv01_usd_per_bp": parallel_dv01(portfolio, curve),
        "portfolio_swap_count": len(portfolio),
        "parallel_up_100bp_pnl_usd": portfolio_pv(portfolio, curve.bump_parallel(100.)) - portfolio_pv(portfolio, curve),
        "parallel_down_100bp_pnl_usd": portfolio_pv(portfolio, curve.bump_parallel(-100.)) - portfolio_pv(portfolio, curve),
    })
    return result


def save_report(report: dict, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return out
