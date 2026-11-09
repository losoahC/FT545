from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lib.option_pricing import gbsm, american, american_greeks

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@pytest.mark.parametrize("case,function", [(1, gbsm), (2, american_greeks)])
def test_workbook_options(case, function):
    options = pd.read_csv(DATA_DIR / "test12_1.csv").dropna(subset=["ID"])
    expected = pd.read_csv(DATA_DIR / f"testout12_{case}.csv")
    for (_, o), (_, e) in zip(options.iterrows(), expected.iterrows()):
        assert o.ID == e.ID
        actual = function(o["Option Type"] == "Call", o.Underlying, o.Strike,
                          o.DaysToMaturity/o.DayPerYear, o.RiskFreeRate,
                          o.RiskFreeRate-o.DividendRate, o.ImpliedVol)
        for key, value in actual.items():
            # American Greeks use central differences of a 500-step lattice.
            np.testing.assert_allclose(value, e[key], rtol=2e-5 if case == 2 else 1e-10,
                                       atol=1e-5 if case == 2 else 1e-12)


def test_workbook_cash_dividends():
    options = pd.read_csv(DATA_DIR / "test12_3.csv")
    expected = pd.read_csv(DATA_DIR / "testout12_3.csv")
    for (_, o), (_, e) in zip(options.iterrows(), expected.iterrows()):
        actual = american(o["Option Type"] == "Call", o.Underlying, o.Strike,
                          o.DaysToMaturity/o.DayPerYear, o.RiskFreeRate, o.RiskFreeRate,
                          o.ImpliedVol, steps=int(2*o.DaysToMaturity),
                          dividend_times=[2*int(d) for d in o.DividendDates.split(",")],
                          dividend_amounts=[float(d) for d in o.DividendAmts.split(",")])
        assert actual == pytest.approx(e.Value, rel=1e-10, abs=1e-12)


def test_european_parity_and_greeks():
    s, k, t, r, b, v = 100., 105., .7, .04, .02, .3
    call, put = [gbsm(kind, s, k, t, r, b, v) for kind in (True, False)]
    assert call["Value"]-put["Value"] == pytest.approx(s*np.exp((b-r)*t)-k*np.exp(-r*t))
    assert call["Delta"]-put["Delta"] == pytest.approx(np.exp((b-r)*t))
    h = 1e-3
    above, below = [gbsm(True, s+x, k, t, r, b, v)["Value"] for x in (h, -h)]
    assert call["Delta"] == pytest.approx((above-below)/(2*h), rel=1e-7)
    assert call["Gamma"] == pytest.approx((above+below-2*call["Value"])/h**2, rel=1e-5)
    for index, name, sign in [(2, "Theta", -1), (5, "Vega", 1)]:
        params = np.array([s, k, t, r, b, v])
        params[index] += h
        upper = gbsm(True, *params)["Value"]
        params[index] -= 2*h
        lower = gbsm(True, *params)["Value"]
        assert call[name] == pytest.approx(sign*(upper-lower)/(2*h), rel=2e-5)


def test_american_exercise_and_no_dividend_call():
    european = gbsm(True, 100, 100, 1, .05, .05, .2)["Value"]
    assert american(True, 100, 100, 1, .05, .05, .2) == pytest.approx(european, abs=.005)
    assert american(False, 60, 100, 1, .05, .05, .2) >= 40
    baseline = american(True, 100, 100, 1, .05, .05, .2, steps=30)
    for dates, amounts in [([], []), ([10, 20], [0., 0.]), ([40], [1.])]:
        assert american(True, 100, 100, 1, .05, .05, .2, steps=30,
                        dividend_times=dates, dividend_amounts=amounts) == pytest.approx(baseline)
    paying = american(True, 100, 100, 1, .05, .05, .2, steps=30,
                      dividend_times=[10], dividend_amounts=[1.])
    assert paying < baseline


@pytest.mark.parametrize("position,value", [(0, 0), (1, -1), (2, 0), (5, 0), (3, np.nan)])
def test_invalid_pricing_inputs(position, value):
    p = [100., 100., 1., .05, .05, .2]
    p[position] = value
    for function in (gbsm, american):
        with pytest.raises(ValueError):
            function(True, *p)


@pytest.mark.parametrize("dates,amounts", [([0], [1]), ([2, 1], [1, 1]), ([1], [-1]), ([1], []), ([1.5], [1])])
def test_invalid_dividend_schedule(dates, amounts):
    with pytest.raises(ValueError):
        american(True, 100, 100, 1, .05, .05, .2, dividend_times=dates, dividend_amounts=amounts)
