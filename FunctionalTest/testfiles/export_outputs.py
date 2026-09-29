"""Calculate all cases before 10.1 and export their actual results as CSV files."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from lib.corr_and_covar import (
    covariance_missing_data_skip_missing_rows, correlation_missing_data_skip_missing_rows,
    covariance_missing_data_pairwise, correlation_missing_data_pairwise,
)
from lib.expo_weighted_corr_and_covar import ew_covariance, ew_correlation, ew_covariance_combined
from lib.fix_nonpsd_corr_covar import near_psd, higham_nearest_psd
from lib.chol_psd import chol_psd
from lib.simulations import simulate_normal, simulate_pca
from lib.returns_conversion import return_calculate
from lib.distribution_fitting import (
    fit_normal, fit_general_t, fit_regression_t, aicc, fit_nig_moments, fit_nig_mle,
)
from lib.risk_measures import normal_risk, t_risk, simulate_t_risk
from lib.portfolio_risk import portfolio_risk

DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "output"


def calculate_outputs():
    outputs = {}

    def read(name):
        return pd.read_csv(DATA_DIR / name)

    def matrix(name, value):
        outputs[name] = pd.DataFrame(value, columns=["x" + str(i + 1) for i in range(value.shape[1])])

    def parameters(name, value):
        outputs[name] = pd.DataFrame([value])

    data = read("test1.csv").to_numpy()
    for i, function in enumerate([
        covariance_missing_data_skip_missing_rows, correlation_missing_data_skip_missing_rows,
        covariance_missing_data_pairwise, correlation_missing_data_pairwise,
    ], 1):
        matrix(f"testout_1.{i}.csv", function(data))
    data = read("test2.csv").to_numpy()
    for i, function in enumerate([ew_covariance, ew_correlation, ew_covariance_combined], 1):
        matrix(f"testout_2.{i}.csv", function(data))
    # Chain computed results, rather than loading reference outputs as inputs.
    for i, function, source in [
        (1, near_psd, "testout_1.3.csv"), (2, near_psd, "testout_1.4.csv"),
        (3, higham_nearest_psd, "testout_1.3.csv"), (4, higham_nearest_psd, "testout_1.4.csv"),
    ]:
        matrix(f"testout_3.{i}.csv", function(outputs[source].to_numpy()))
    matrix("testout_4.1.csv", chol_psd(outputs["testout_3.1.csv"].to_numpy()))
    for i, source, repair in [
        (1, "test5_1.csv", None), (2, "test5_2.csv", None),
        (3, "test5_3.csv", near_psd), (4, "test5_3.csv", higham_nearest_psd),
    ]:
        samples = simulate_normal(100000, read(source), fix_method=repair, seed=545)
        matrix(f"testout_5.{i}.csv", np.cov(samples, rowvar=False))
    samples = simulate_pca(read("test5_2.csv"), 100000, explained=0.99, seed=545)
    matrix("testout_5.5.csv", np.cov(samples, rowvar=False))
    for i, method in enumerate(["arithmetic", "log"], 1):
        outputs[f"testout6_{i}.csv"] = return_calculate(read("test6.csv"), method)
    normal = fit_normal(read("test7_1.csv").iloc[:, 0])
    data = read("test7_2.csv").iloc[:, 0].to_numpy()
    fitted_t = fit_general_t(data)
    parameters("testout7_1.csv", normal)
    parameters("testout7_2.csv", fitted_t)
    regression = read("test7_3.csv").to_numpy()
    parameters("testout7_3.csv", fit_regression_t(regression[:, :-1], regression[:, -1]))
    likelihood = stats.t.logpdf(data, fitted_t["nu"], loc=fitted_t["mu"], scale=fitted_t["sigma"]).sum()
    parameters("testout7_4.csv", {"AICC": aicc(likelihood, len(data), 3)})
    nig_data = read("test7_5.csv").iloc[:, 0]
    parameters("testout7_5.csv", fit_nig_moments(nig_data))
    parameters("testout7_6.csv", fit_nig_mle(nig_data))
    normal_result = normal_risk(**normal)
    t_result = t_risk(**fitted_t)
    for i, measure, result in [
        (1, "VaR", normal_result), (2, "VaR", t_result),
        (3, "VaR", simulate_t_risk(**fitted_t, seed=8)),
        (4, "ES", normal_result), (5, "ES", t_result),
        (6, "ES", simulate_t_risk(**fitted_t, seed=86)),
    ]:
        parameters(f"testout8_{i}.csv", {key: value for key, value in result.items() if key.startswith(measure)})
    outputs["testout9_1.csv"] = portfolio_risk(read("test9_1_returns.csv"), read("test9_1_portfolio.csv"), seed=9)
    return outputs


def export_outputs(output_dir=DEFAULT_OUTPUT):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = calculate_outputs()
    for filename, frame in outputs.items():
        frame.to_csv(output_dir / filename, index=False)
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    outputs = export_outputs(args.output_dir)
    print(f"Exported {len(outputs)} CSV files to {args.output_dir.resolve()}")
