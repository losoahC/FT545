from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from export_outputs import export_outputs

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
FILENAMES = (
    [f"testout_{section}.{i}.csv" for section, count in [(1, 4), (2, 3), (3, 4), (4, 1), (5, 5)]
     for i in range(1, count + 1)]
    + [f"testout{section}_{i}.csv" for section, count in [(6, 2), (7, 6), (8, 6), (9, 1)]
       for i in range(1, count + 1)]
)


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    folder = tmp_path_factory.mktemp("outputs")
    calculated = export_outputs(folder)
    assert set(calculated) == set(FILENAMES)
    assert {p.name for p in folder.glob("*.csv")} == set(FILENAMES)
    return folder, calculated


@pytest.mark.parametrize("filename", FILENAMES)
def test_exported_csv_matches_calculation_and_reference(exported, filename):
    folder, calculated = exported
    actual = pd.read_csv(folder / filename)
    expected = pd.read_csv(DATA_DIR / filename)
    assert actual.shape == expected.shape
    assert actual.columns.tolist() == expected.columns.tolist()
    pd.testing.assert_frame_equal(actual, calculated[filename], check_exact=False, rtol=1e-12, atol=1e-12)
    numeric = expected.select_dtypes(include="number").columns
    for column in expected.columns.difference(numeric):
        pd.testing.assert_series_equal(actual[column], expected[column])
    rtol, atol = 1e-10, 1e-12
    if filename.startswith("testout_5."):
        # Normalize by standard deviations so near-zero covariances are meaningful.
        reference = expected.to_numpy()
        scale = np.sqrt(np.outer(np.diag(reference), np.diag(reference)))
        assert np.max(np.abs(actual.to_numpy() - reference) / scale) < 0.04
        return
    if filename == "testout_4.1.csv":
        atol = 1e-7
    elif filename.startswith("testout7_") or filename.startswith("testout8_"):
        rtol = 1e-4
    if filename == "testout8_3.csv":
        atol = 0.008
    elif filename == "testout8_6.csv":
        atol = 0.02
    elif filename == "testout9_1.csv":
        rtol = 0.04
    np.testing.assert_allclose(actual[numeric], expected[numeric], rtol=rtol, atol=atol)
