import numpy as np
import pandas as pd
import pytest

from swapscribe.market import curve_from_row, history_key_rate_changes
from swapscribe.risk import Swap, par_rate, price_swap, portfolio_pv, key_rate_pv_changes, optimize_hedge, risk_report
from swapscribe.backtest import walk_forward_backtest


def test_par_swap_zero_value(rates):
    curve = curve_from_row(rates.iloc[-1])
    rate = par_rate(curve, 5) * 100
    assert abs(price_swap(Swap("PAR", 1_000_000, 5, rate), curve)) < 1e-7


def test_direction_symmetry(rates):
    curve = curve_from_row(rates.iloc[-1])
    pay = Swap("P", 1_000_000, 5, 3.0, "pay_fixed")
    receive = Swap("R", 1_000_000, 5, 3.0, "receive_fixed")
    assert price_swap(pay, curve) == -price_swap(receive, curve)


def test_positive_rate_bump_increases_pay_fixed_pv(rates):
    curve = curve_from_row(rates.iloc[-1])
    swap = [Swap("P", 1_000_000, 4, 4.0, "pay_fixed")]
    assert portfolio_pv(swap, curve.bump_parallel(1)) > portfolio_pv(swap, curve)


def test_risk_and_bounded_hedge(rates, portfolio):
    curve = curve_from_row(rates.iloc[-1])
    hist = history_key_rate_changes(rates)
    report = optimize_hedge(portfolio, curve, hist, max_abs_millions=5)
    assert all(abs(x) <= 5.000001 for x in report["hedge_notional_usd_millions"])
    assert report["rms_scenario_pnl_usd_after"] < report["rms_scenario_pnl_usd_before"]
    assert len(report["tenors_years"]) == 7


def test_asof_blocks_future_valuation(rates, portfolio):
    cutoff = pd.Timestamp(rates.iloc[-40].date)
    expected = risk_report(rates[rates.date <= cutoff], portfolio)
    actual = risk_report(rates, portfolio, cutoff)
    assert expected["as_of"] == actual["as_of"]
    assert np.isclose(expected["portfolio_pv_usd"], actual["portfolio_pv_usd"])


def test_walkforward_oos(rates, portfolio):
    report = walk_forward_backtest(rates, portfolio, lookback=40, holding_days=7)
    assert report["oos_test_days"] > 50
    assert report["oos_rms_pnl_usd_unhedged"] > 0
    assert report["oos_rms_pnl_usd_optimized"] >= 0
    assert report["first_rebalance"] < report["last_rebalance"]


def test_invalid_portfolio():
    with pytest.raises(ValueError):
        Swap("bad", 0, 4, 4.0)
    with pytest.raises(ValueError):
        Swap("bad", 1e6, 4, 4.0, "something")
