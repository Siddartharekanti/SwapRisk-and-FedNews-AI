"""Constrained convex quadratic hedge optimizer for multi-tenor swap books.

Implements bounded weighted least-squares hedging over empirical yield changes
and prescribed deterministic stress scenarios using the BVLS algorithm
(scipy.optimize.lsq_linear) with L2 Tikhonov regularization.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import lsq_linear

from .market import YieldCurve


def build_scenario_matrix(historical_changes_bp: np.ndarray, num_tenors: int) -> np.ndarray:
    """Combine empirical yield changes with ±100bp key-node stress scenarios."""
    stress = np.concatenate([np.eye(num_tenors) * 100.0, -np.eye(num_tenors) * 100.0], axis=0)
    return np.vstack([historical_changes_bp, stress])


def optimize_hedge(portfolio, curve: YieldCurve, historical_changes_bp: np.ndarray,
                   tenors=(1., 2., 3., 5., 7., 10., 20.),
                   hedge_maturities=(2, 5, 10),
                   max_abs_millions: float = 30.,
                   ridge: float = 200.) -> dict:
    """Bounded weighted least-squares hedging on historical + prescribed stresses.

    x is signed notional in millions of USD (+ = pay fixed; - = receive fixed).
    Minimizes scenario P/L RMS and an L2 proxy for position/concentration costs.
    Ridge is in USD-per-million units after scenario normalization.
    """
    from .risk import key_rate_pv_changes, hedge_instruments

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

    # Construct combined scenario matrix
    stress = np.concatenate([np.eye(len(tenors)) * 100.0, -np.eye(len(tenors)) * 100.0], axis=0)
    scenarios = np.vstack([historical_changes_bp, stress])

    a = scenarios @ r
    b = scenarios @ h
    scale = float(np.sqrt(len(scenarios)))

    # Regularized least-squares design matrix
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
        "curve_model": "Exact bootstrapped zero curve from par Treasury yields",
    }
