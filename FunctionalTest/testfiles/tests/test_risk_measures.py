from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import integrate, stats

from lib.distribution_fitting import fit_normal, fit_general_t
from lib.risk_measures import normal_risk, t_risk, sample_risk, simulate_t_risk

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@pytest.mark.parametrize("case, measure, kind", [
    (1, "VaR", "normal"), (2, "VaR", "t"), (3, "VaR", "simulation"),
    (4, "ES", "normal"), (5, "ES", "t"), (6, "ES", "simulation"),
])
def test_workbook_section_8(case, measure, kind):
    source = "test7_1.csv" if kind == "normal" else "test7_2.csv"
    data = pd.read_csv(DATA_DIR / source).iloc[:, 0]
    if kind == "normal":
        actual = normal_risk(**fit_normal(data))
    elif kind == "t":
        actual = t_risk(**fit_general_t(data))
    else:
        fitted = fit_general_t(data)
        actual = simulate_t_risk(**fitted, seed=8 if case == 3 else 86)
        theoretical = t_risk(**fitted)
        # 10,000 draws: allow sampling differences in the tail estimates.
        for key in [measure + " Absolute", measure + " Diff from Mean"]:
            assert abs(actual[key] - theoretical[key]) < (0.006 if measure == "VaR" else 0.012)
    expected = pd.read_csv(DATA_DIR / f"testout8_{case}.csv").iloc[0]
    for key in expected.index:
        if kind == "simulation":
            assert abs(actual[key] - expected[key]) < (0.008 if measure == "VaR" else 0.02)
        else:
            assert actual[key] == pytest.approx(expected[key], rel=1e-4, abs=1e-10)


def test_expected_shortfall_against_integration():
    for distribution, result in [
        (stats.norm(loc=0.02, scale=0.1), normal_risk(0.02, 0.1)),
        (stats.t(5, loc=0.02, scale=0.1), t_risk(0.02, 0.1, 5)),
    ]:
        quantile = distribution.ppf(0.05)
        tail, _ = integrate.quad(lambda x: x * distribution.pdf(x), -np.inf, quantile)
        assert result["VaR Absolute"] == pytest.approx(-quantile)
        assert result["ES Absolute"] == pytest.approx(-tail / 0.05)


def test_sample_tail_and_translation():
    data = np.array([-4., -2., 1., 3.])
    original = data.copy()
    actual = sample_risk(data, alpha=0.375)
    assert actual["ES Absolute"] == pytest.approx(10 / 3)
    shifted = sample_risk(data + 10, alpha=0.375)
    for measure in ["VaR", "ES"]:
        assert shifted[measure + " Absolute"] == pytest.approx(actual[measure + " Absolute"] - 10)
        assert shifted[measure + " Diff from Mean"] == pytest.approx(actual[measure + " Diff from Mean"])
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize("alpha", [0, 1, np.nan])
def test_invalid_probability(alpha):
    with pytest.raises(ValueError):
        normal_risk(0, 1, alpha)
    with pytest.raises(ValueError):
        sample_risk([1, 2], alpha)


def test_infinite_t_shortfall_rejected():
    with pytest.raises(ValueError):
        t_risk(0, 1, 1)


def test_simulation_is_reproducible():
    assert simulate_t_risk(0, 1, 5, seed=8) == simulate_t_risk(0, 1, 5, seed=8)
