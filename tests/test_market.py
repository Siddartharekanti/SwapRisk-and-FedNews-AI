import numpy as np
import pytest

from swapscribe.market import clean_treasury_csv, YieldCurve, curve_from_row, history_key_rate_changes


def test_treasury_parser_normalizes():
    raw = "Date,1 Yr,2 Yr,5 Yr,10 Yr\n01/03/2025,3.9,4.1,4.2,4.5\n01/02/2025,3.8,4.0,4.1,4.4\n"
    df = clean_treasury_csv(raw)
    assert df.iloc[0]["2 Yr"] == 4.0
    assert str(df.iloc[-1].date.date()) == "2025-01-03"
    assert np.isnan(df.iloc[0]["30 Yr"])


def test_parser_rejects_invalid():
    with pytest.raises(ValueError):
        clean_treasury_csv("Date,2 Yr,10 Yr\n2025-01-01,999,1.0\n")


def test_discount_curve_and_bump(rates):
    curve = curve_from_row(rates.iloc[-1])
    assert 0 < curve.discount(10) < 1
    assert curve.bump_parallel(100).discount(10) < curve.discount(10)
    assert curve.bump_key(10, 10).rate(10) > curve.rate(10)
    with pytest.raises(ValueError):
        curve.bump_key(99, 10)


def test_shocks_shape(rates):
    shocks = history_key_rate_changes(rates)
    assert shocks.shape == (219, 7)
    assert np.isfinite(shocks).all()


def test_curve_validation():
    with pytest.raises(ValueError):
        YieldCurve(np.array([2., 1.]), np.array([.04, .05]))
