from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from lib.distribution_fitting import (
    fit_normal, fit_general_t, fit_regression_t, aicc, fit_nig_moments, fit_nig_mle,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def read_sample(filename):
    return np.genfromtxt(DATA_DIR / filename, delimiter=",", skip_header=1)


def compare_parameters(actual, filename, rtol):
    expected = pd.read_csv(DATA_DIR / filename).iloc[0].to_dict()
    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        np.testing.assert_allclose(actual[name], value, rtol=rtol, atol=1e-10,
                                   err_msg=name)


def test_workbook_7_1():
    actual = fit_normal(read_sample("test7_1.csv"))
    compare_parameters(actual, "testout7_1.csv", rtol=1e-10)


def test_workbook_7_2():
    actual = fit_general_t(read_sample("test7_2.csv"))
    compare_parameters(actual, "testout7_2.csv", rtol=1e-4)


def test_workbook_7_3():
    data = read_sample("test7_3.csv")
    actual = fit_regression_t(data[:, :-1], data[:, -1])
    compare_parameters(actual, "testout7_3.csv", rtol=1e-5)
    # Joint likelihood fitting should improve on the OLS starting coefficients.
    design = np.column_stack([np.ones(len(data)), data[:, :-1]])
    ols = np.linalg.lstsq(design, data[:, -1], rcond=None)[0]
    coefficients = [actual["Alpha"], actual["B1"], actual["B2"], actual["B3"]]
    fitted_ll = stats.t.logpdf(data[:, -1] - design @ coefficients,
                              actual["nu"], scale=actual["sigma"]).sum()
    ols_ll = stats.t.logpdf(data[:, -1] - design @ ols,
                           actual["nu"], scale=actual["sigma"]).sum()
    assert fitted_ll >= ols_ll


def test_workbook_7_4():
    data = read_sample("test7_2.csv")
    fitted = fit_general_t(data)
    log_likelihood = stats.t.logpdf(
        data, fitted["nu"], loc=fitted["mu"], scale=fitted["sigma"],
    ).sum()
    actual = aicc(log_likelihood, len(data), k=3)
    compare_parameters({"AICC": actual}, "testout7_4.csv", rtol=1e-8)


def nig_distribution(parameters):
    return stats.norminvgauss(
        parameters["alpha"] * parameters["delta"],
        parameters["beta"] * parameters["delta"],
        loc=parameters["mu"], scale=parameters["delta"],
    )


def test_workbook_7_5():
    data = read_sample("test7_5.csv")
    actual = fit_nig_moments(data)
    compare_parameters(actual, "testout7_5.csv", rtol=1e-10)
    moments = nig_distribution(actual).stats(moments="mvsk")
    expected = [data.mean(), data.var(ddof=1), stats.skew(data), stats.kurtosis(data)]
    np.testing.assert_allclose(moments, expected, rtol=1e-10)


def test_workbook_7_6():
    data = read_sample("test7_5.csv")
    actual = fit_nig_mle(data)
    # Numerical optimizer results may differ slightly across SciPy versions.
    compare_parameters(actual, "testout7_6.csv", rtol=1e-4)
    mle_ll = nig_distribution(actual).logpdf(data).sum()
    moments_ll = nig_distribution(fit_nig_moments(data)).logpdf(data).sum()
    assert mle_ll >= moments_ll


def test_normal_uses_sample_standard_deviation():
    assert fit_normal([1, 2, 3]) == {"mu": 2, "sigma": 1}


def test_aicc_parameter_count():
    assert aicc(-10, n=20, k=3) == pytest.approx(27.5)


@pytest.mark.parametrize("function", [fit_normal, fit_general_t, fit_nig_moments, fit_nig_mle])
@pytest.mark.parametrize("data", [[], [1], [1, 1, 1, 1], [1, 2, np.nan, 4], [[1, 2]]])
def test_invalid_sample(function, data):
    with pytest.raises(ValueError):
        function(data)


def test_nig_infeasible_moments():
    with pytest.raises(ValueError, match="moments"):
        fit_nig_moments([-2, -1, 0, 1, 2])


def test_regression_rejects_dependent_predictors():
    x = np.column_stack([np.arange(10), np.arange(10)])
    with pytest.raises(ValueError, match="dependent"):
        fit_regression_t(x, np.arange(10) ** 2)


@pytest.mark.parametrize("n, k", [(4, 3), (3, 3), (20, 0)])
def test_aicc_insufficient_observations(n, k):
    with pytest.raises(ValueError):
        aicc(-10, n, k)


def test_t_fit_location_and_scale():
    data = read_sample("test7_2.csv")
    original = data.copy()
    fitted = fit_general_t(data)
    transformed = fit_general_t(3 * data + 2)
    assert transformed["mu"] == pytest.approx(3 * fitted["mu"] + 2, rel=1e-4)
    assert transformed["sigma"] == pytest.approx(3 * fitted["sigma"], rel=1e-4)
    assert transformed["nu"] == pytest.approx(fitted["nu"], rel=1e-4)
    np.testing.assert_array_equal(data, original)


def test_regression_response_shift_and_no_mutation():
    data = read_sample("test7_3.csv")
    original = data.copy()
    fitted = fit_regression_t(data[:, :-1], data[:, -1])
    shifted = fit_regression_t(data[:, :-1], data[:, -1] + 2)
    assert shifted["Alpha"] == pytest.approx(fitted["Alpha"] + 2, rel=1e-5)
    for name in ["sigma", "nu", "B1", "B2", "B3"]:
        assert shifted[name] == pytest.approx(fitted[name], rel=1e-5)
    assert shifted["mu"] == 0
    np.testing.assert_array_equal(data, original)


def test_nig_moments_location_and_scale():
    data = read_sample("test7_5.csv")
    original = data.copy()
    fitted = fit_nig_moments(data)
    transformed = fit_nig_moments(3 * data + 2)
    assert transformed["mu"] == pytest.approx(3 * fitted["mu"] + 2)
    assert transformed["delta"] == pytest.approx(3 * fitted["delta"])
    assert transformed["alpha"] == pytest.approx(fitted["alpha"] / 3)
    assert transformed["beta"] == pytest.approx(fitted["beta"] / 3)
    np.testing.assert_array_equal(data, original)
