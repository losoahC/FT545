"""Calculate all functional cases (sections 1–13) and export actual CSV results."""

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
from lib.portfolio_optimization import risk_parity, max_sharpe
from lib.return_attribution import expost_factor
from lib.option_pricing import gbsm, american, american_greeks
from lib.multivariate_t import (
    kendall_correlation, fit_multivariate_t, fit_t_margins, fit_gaussian_copula,
    fit_t_copula, copula_criteria, tail_dependence_t, t_copula_portfolio_risk,
)

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
    cov = read("test5_2.csv").to_numpy()
    for i, budget in [(1, None), (2, [1, 1, 1, 1, .5])]:
        outputs[f"testout10_{i}.csv"] = pd.DataFrame({"W": risk_parity(cov, budget)})
    cov = read("test5_3.csv").to_numpy()
    means = read("test10_3_means.csv").Mean.to_numpy()
    for i, bounds in [(3, None), (4, np.tile([.1, .5], (5, 1)))]:
        outputs[f"testout10_{i}.csv"] = pd.DataFrame({"W": max_sharpe(cov, means, .04, bounds)})
    returns = read("test11_1_returns.csv")
    outputs["testout11_1.csv"] = expost_factor(read("test11_1_weights.csv").W, returns,
                                             returns, np.eye(3), include_alpha=False)
    outputs["testout11_2.csv"] = expost_factor(
        read("test11_2_weights.csv").W, read("test11_2_stock_returns.csv"),
        read("test11_2_factor_returns.csv"), read("test11_2_beta.csv").iloc[:, 1:])
    for i, function in [(1, gbsm), (2, american_greeks)]:
        rows = []
        for _, o in read("test12_1.csv").dropna(subset=["ID"]).iterrows():
            result = function(o["Option Type"] == "Call", o.Underlying, o.Strike,
                              o.DaysToMaturity/o.DayPerYear, o.RiskFreeRate,
                              o.RiskFreeRate-o.DividendRate, o.ImpliedVol)
            rows.append({"ID": int(o.ID), **result})
        outputs[f"testout12_{i}.csv"] = pd.DataFrame(rows)
    rows = []
    for _, o in read("test12_3.csv").dropna(subset=["ID"]).iterrows():
        value = american(o["Option Type"] == "Call", o.Underlying, o.Strike,
                         o.DaysToMaturity/o.DayPerYear, o.RiskFreeRate, o.RiskFreeRate,
                         o.ImpliedVol, steps=int(2*o.DaysToMaturity),
                         dividend_times=[2*int(d) for d in o.DividendDates.split(",")],
                         dividend_amounts=[float(d) for d in o.DividendAmts.split(",")])
        rows.append({"ID": int(o.ID), "Value": value})
    outputs["testout12_3.csv"] = pd.DataFrame(rows)
    returns = read("test13_returns.csv")
    matrix("testout13_1.csv", kendall_correlation(returns))
    mu, scale, nu, ll = fit_multivariate_t(returns)
    outputs["testout13_2.csv"] = pd.DataFrame({"mu": mu})
    matrix("testout13_3.csv", scale)
    parameters("testout13_4.csv", {"nu": nu, "ll": ll})
    models, uniforms = fit_t_margins(returns)
    _, ll_g = fit_gaussian_copula(uniforms)
    correlation, nu, ll_t = fit_t_copula(uniforms)
    parameters("testout13_5.csv", {"ll": ll_g})
    parameters("testout13_6.csv", {"nu": nu, "ll": ll_t})
    outputs["testout13_7.csv"] = pd.DataFrame([
        {"Copula": name, "LL": ll, "K": k, **copula_criteria(ll, k, len(returns))}
        for name, ll, k in [("Gaussian", ll_g, 0), ("T", ll_t, 1)]])
    outputs["testout13_8.csv"] = pd.DataFrame([
        {"I": i+1, "J": j+1, "Rho": correlation[i, j],
         "Lambda": tail_dependence_t(correlation[i, j], nu)}
        for i in range(len(correlation)) for j in range(i+1, len(correlation))])
    outputs["testout13_9.csv"] = t_copula_portfolio_risk(
        returns, read("test13_portfolio.csv"), fitted=(models, correlation, nu), seed=13)
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
