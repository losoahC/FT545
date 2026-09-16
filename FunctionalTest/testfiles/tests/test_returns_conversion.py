from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lib.returns_conversion import return_calculate

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@pytest.mark.parametrize("method, expected_file", [
    pytest.param("arithmetic", "testout6_1.csv", id="6.1"),
    pytest.param("log", "testout6_2.csv", id="6.2"),
])
def test_workbook_returns(method, expected_file):
    prices = pd.read_csv(DATA_DIR / "test6.csv")
    original = prices.copy()
    expected = pd.read_csv(DATA_DIR / expected_file)
    actual = return_calculate(prices, method=method)
    assert actual.shape == expected.shape
    assert actual.columns.tolist() == expected.columns.tolist()
    pd.testing.assert_series_equal(actual["Date"], expected["Date"])
    np.testing.assert_allclose(
        actual.drop(columns="Date"), expected.drop(columns="Date"),
        rtol=1e-10, atol=1e-12,
    )
    pd.testing.assert_frame_equal(prices, original)


def test_simple_returns_and_custom_date_column():
    prices = pd.DataFrame({"Stock": [100, 110, 99], "Day": ["a", "b", "c"]})
    arithmetic = return_calculate(prices, date_column="Day")
    logarithmic = return_calculate(prices, method="LOG", date_column="Day")
    assert arithmetic.columns.tolist() == ["Stock", "Day"]
    assert arithmetic["Day"].tolist() == ["b", "c"]
    np.testing.assert_allclose(arithmetic["Stock"], [0.1, -0.1])
    np.testing.assert_allclose(logarithmic["Stock"], np.log([1.1, 0.9]))


@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf])
def test_invalid_prices(value):
    prices = pd.DataFrame({"Date": ["a", "b"], "Stock": [100, value]})
    with pytest.raises(ValueError):
        return_calculate(prices)


def test_invalid_method():
    prices = pd.DataFrame({"Date": ["a", "b"], "Stock": [100, 110]})
    with pytest.raises(ValueError):
        return_calculate(prices, method="unknown")


@pytest.mark.parametrize("prices", [
    pd.DataFrame({"Stock": [100, 110]}),
    pd.DataFrame({"Date": ["a"], "Stock": [100]}),
    pd.DataFrame({"Date": ["a", "b"]}),
])
def test_invalid_table(prices):
    with pytest.raises(ValueError):
        return_calculate(prices)
