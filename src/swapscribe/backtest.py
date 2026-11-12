"""Leakage-controlled historical risk hedging backtest with second-order Greeks & transaction costs.

The optimizer sees past rate changes only; the test interval begins after
rebalancing. Evaluates out-of-sample risk with key-rate DV01, convexity (Gamma),
and execution turnover friction.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .market import TENORS, curve_from_row, history_key_rate_changes
from .risk import Swap, portfolio_pv, hedge_instruments, key_rate_pv_changes, optimize_hedge

KEY_TENORS = (1., 2., 3., 5., 7., 10., 20.)


def _series_columns(tenors):
    return [next(name for name, maturity in TENORS.items() if maturity == t) for t in tenors]


def key_rate_convexity(portfolio: list[Swap], curve, tenors=KEY_TENORS, bump_bp=5.0) -> np.ndarray:
    """Second-order curvature (Gamma) in USD per bp^2 across key-rate nodes."""
    pv0 = portfolio_pv(portfolio, curve)
    conv = []
    for t in tenors:
        pv_up = portfolio_pv(portfolio, curve.bump_key(t, bump_bp))
        pv_dn = portfolio_pv(portfolio, curve.bump_key(t, -bump_bp))
        conv.append((pv_up + pv_dn - 2.0 * pv0) / (bump_bp ** 2))
    return np.asarray(conv, dtype=float)


def walk_forward_backtest(rates: pd.DataFrame, portfolio: list[Swap],
                          lookback: int = 70, holding_days: int = 5,
                          max_abs_millions: float = 30.,
                          transaction_cost_bp: float = 0.5) -> dict:
    """Expanding-step rebalance with rolling lookback, strictly future tests, second-order Greeks & costs.

    E.g. 70 past daily changes for hedge weights; apply on next 5 daily changes.
    Scoring incorporates first-order DV01 + second-order key-rate convexity (Gamma)
    and turnover friction (default 0.5 bp bid-ask on hedge rebalancing).
    """
    rates = rates.copy().sort_values("date").reset_index(drop=True)
    if len(rates) < lookback + holding_days + 2:
        raise ValueError("Not enough dated history for walk-forward backtest")
    columns = _series_columns(KEY_TENORS)
    if rates[columns].isna().any().any():
        raise ValueError("Walk-forward requires complete node yields; select recent Treasury years")
    outcome = {"unhedged": [], "hedged": [], "hedged_net_costs": [], "parallel_only": []}
    rebalances = []
    prev_weights = None
    total_turnover_mm = 0.0
    total_transaction_costs = 0.0

    for end in range(lookback + 1, len(rates) - holding_days, holding_days):
        past = rates.iloc[end-lookback-1:end+1]
        shocks = history_key_rate_changes(past, key_tenors=KEY_TENORS, max_rows=lookback)
        current_curve = curve_from_row(rates.iloc[end])
        result = optimize_hedge(portfolio, current_curve, shocks, tenors=KEY_TENORS,
                                max_abs_millions=max_abs_millions)
        r = key_rate_pv_changes(portfolio, current_curve, KEY_TENORS)
        r_conv = key_rate_convexity(portfolio, current_curve, KEY_TENORS)

        # Reference hedges
        hedges = hedge_instruments(current_curve, (2, 5, 10))
        h_matrix = np.column_stack([key_rate_pv_changes([inst], current_curve, KEY_TENORS) for inst in hedges])
        h_conv_matrix = np.column_stack([key_rate_convexity([inst], current_curve, KEY_TENORS) for inst in hedges])
        x = np.asarray(result["hedge_notional_usd_millions"])
        hedged_dv01 = r + h_matrix @ x
        hedged_conv = r_conv + h_conv_matrix @ x

        # Parallel baseline (single 10Y hedge)
        ref10 = hedge_instruments(current_curve, (10,))[0]
        h10 = key_rate_pv_changes([ref10], current_curve, KEY_TENORS)
        h10_conv = key_rate_convexity([ref10], current_curve, KEY_TENORS)
        dv01_hedge_mm = float(np.clip(-np.sum(r) / np.sum(h10), -max_abs_millions, max_abs_millions))
        baseline_dv01 = r + h10 * dv01_hedge_mm
        baseline_conv = r_conv + h10_conv * dv01_hedge_mm

        # Turnover and transaction costs for this rebalance window
        turnover_rebal = float(np.sum(np.abs(x - prev_weights))) if prev_weights is not None else float(np.sum(np.abs(x)))
        rebal_cost_usd = turnover_rebal * 1_000_000 * (transaction_cost_bp / 10_000)
        total_turnover_mm += turnover_rebal
        total_transaction_costs += rebal_cost_usd
        prev_weights = x.copy()

        for step_idx, j in enumerate(range(end + 1, end + 1 + holding_days)):
            # Rates on day j were never visible to optimizer at day end.
            change_bp = (rates.iloc[j][columns].to_numpy(dtype=float) -
                         rates.iloc[j-1][columns].to_numpy(dtype=float)) * 100
            # Second-order P&L: DV01 * dy + 0.5 * Gamma * dy^2
            pnl_unhedged = float(change_bp @ r + 0.5 * np.sum(r_conv * (change_bp ** 2)))
            pnl_hedged = float(change_bp @ hedged_dv01 + 0.5 * np.sum(hedged_conv * (change_bp ** 2)))
            pnl_baseline = float(change_bp @ baseline_dv01 + 0.5 * np.sum(baseline_conv * (change_bp ** 2)))

            # Charge transaction friction on the rebalance execution day
            day_cost = rebal_cost_usd if step_idx == 0 else 0.0
            pnl_hedged_net = pnl_hedged - day_cost

            outcome["unhedged"].append(pnl_unhedged)
            outcome["hedged"].append(pnl_hedged)
            outcome["hedged_net_costs"].append(pnl_hedged_net)
            outcome["parallel_only"].append(pnl_baseline)

        rebalances.append(pd.Timestamp(rates.iloc[end].date).date().isoformat())

    if not rebalances:
        raise ValueError("No walk-forward windows formed")

    return {
        "method": "70-day trailing rebalance; forward out-of-sample holding test with 2nd-order Gamma & transaction costs",
        "lookback_daily_changes": lookback,
        "holding_days": holding_days,
        "first_rebalance": rebalances[0], "last_rebalance": rebalances[-1],
        "rebalance_count": len(rebalances), "oos_test_days": len(outcome["unhedged"]),
        "oos_rms_pnl_usd_unhedged": float(np.sqrt(np.mean(np.square(outcome["unhedged"])))),
        "oos_rms_pnl_usd_parallel_only": float(np.sqrt(np.mean(np.square(outcome["parallel_only"])))),
        "oos_rms_pnl_usd_optimized": float(np.sqrt(np.mean(np.square(outcome["hedged"])))),
        "oos_rms_pnl_usd_optimized_net_of_costs": float(np.sqrt(np.mean(np.square(outcome["hedged_net_costs"])))),
        "total_turnover_usd_millions": round(total_turnover_mm, 2),
        "total_transaction_costs_usd": round(total_transaction_costs, 2),
        "transaction_cost_basis_points": transaction_cost_bp,
        "oos_pnl_series": outcome,
        "note": "Evaluates full 1st-order DV01 + 2nd-order key-rate convexity (Gamma) and bid-ask turnover friction across nonoverlapping holding intervals",
    }
