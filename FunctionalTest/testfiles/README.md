# Functional tests: Sections 1–13

Activate your Python environment, then run these commands from the repository root:

```bash
python -m pip install -r FunctionalTest/testfiles/requirements-test.txt
python -m pytest -c FunctionalTest/testfiles/pytest.ini -v
```

Alternatively, run `python -m pytest -v` from this directory.

The four functions in `lib/corr_and_covar.py` accept two-dimensional numeric data:
rows are observations, columns are variables, and `np.nan` represents missing
values. Each function returns a NumPy matrix. The input is the complete `data`
matrix so that row deletion can check all variables for missing values. Tests
handle CSV loading.

| Tests.xlsx case | Calculation | Input | Expected output |
| --- | --- | --- | --- |
| 1.1 | Sample covariance with complete-case deletion | data/test1.csv | data/testout_1.1.csv |
| 1.2 | Pearson correlation with complete-case deletion | data/test1.csv | data/testout_1.2.csv |
| 1.3 | Sample covariance with pairwise deletion | data/test1.csv | data/testout_1.3.csv |
| 1.4 | Pearson correlation with pairwise deletion | data/test1.csv | data/testout_1.4.csv |
| 2.1 | EW covariance, lambda=0.97 | data/test2.csv | data/testout_2.1.csv |
| 2.2 | EW correlation, lambda=0.94 | data/test2.csv | data/testout_2.2.csv |
| 2.3 | EW variances with lambda=0.97 and EW correlation with lambda=0.94 | data/test2.csv | data/testout_2.3.csv |
| 3.1 | Covariance repair using near_psd | data/testout_1.3.csv | data/testout_3.1.csv |
| 3.2 | Correlation repair using near_psd | data/testout_1.4.csv | data/testout_3.2.csv |
| 3.3 | Covariance repair using Higham's algorithm | data/testout_1.3.csv | data/testout_3.3.csv |
| 3.4 | Correlation repair using Higham's algorithm | data/testout_1.4.csv | data/testout_3.4.csv |
| 4.1 | PSD Cholesky factorization | data/testout_3.1.csv | data/testout_4.1.csv |
| 5.1 | Normal simulation with positive definite covariance | data/test5_1.csv | data/testout_5.1.csv |
| 5.2 | Normal simulation with positive semidefinite covariance | data/test5_2.csv | data/testout_5.2.csv |
| 5.3 | Normal simulation after near_psd repair | data/test5_3.csv | data/testout_5.3.csv |
| 5.4 | Normal simulation after Higham repair | data/test5_3.csv | data/testout_5.4.csv |
| 5.5 | PCA simulation retaining 99% of variance | data/test5_2.csv | data/testout_5.5.csv |
| 6.1 | Arithmetic returns | data/test6.csv | data/testout6_1.csv |
| 6.2 | Log returns | data/test6.csv | data/testout6_2.csv |
| 7.1 | Normal distribution fitting | data/test7_1.csv | data/testout7_1.csv |
| 7.2 | Generalized Student t fitting by maximum likelihood | data/test7_2.csv | data/testout7_2.csv |
| 7.3 | Regression with Student t errors | data/test7_3.csv | data/testout7_3.csv |
| 7.4 | AICc for the fitted Student t distribution | data/test7_2.csv | data/testout7_4.csv |
| 7.5 | NIG fitting by the method of moments | data/test7_5.csv | data/testout7_5.csv |
| 7.6 | NIG fitting by maximum likelihood | data/test7_5.csv | data/testout7_6.csv |

Section 7 is implemented in `lib/distribution_fitting.py`, with tests in
`tests/test_distribution_fitting.py`, and requires SciPy. Fitting functions
return parameter dictionaries:

| Case | Function | Return values |
| --- | --- | --- |
| 7.1 | fit_normal(data) | mu, sigma (sample standard deviation) |
| 7.2 | fit_general_t(data) | mu, sigma (scale parameter), nu |
| 7.3 | fit_regression_t(x, y) | mu=0, sigma, nu, Alpha, B1, B2, ... |
| 7.4 | aicc(log_likelihood, n, k) | AICc scalar |
| 7.5 | fit_nig_moments(data) | mu, alpha, beta, delta |
| 7.6 | fit_nig_mle(data) | mu, alpha, beta, delta |

For 7.3, `x` is a two-dimensional predictor matrix without an intercept column,
and `y` is a one-dimensional response. The function jointly fits the intercept,
regression coefficients, and Student t error parameters. For 7.4, compute the
summed log likelihood using the parameters from 7.2, then call
`aicc(log_likelihood, len(data), 3)`, where 3 is the number of estimated parameters.
Case 7.5 uses sample variance and uncorrected skewness and excess kurtosis to
match the reference convention. It raises an exception when the sample moments
do not admit finite NIG parameters.

Case 7.6 converts SciPy's `(a, b, loc, scale)` to `(mu, alpha, beta, delta)`;
see the [SciPy NIG documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norminvgauss.html).
Cases 7.1 and 7.5 use `rtol=1e-10`. Numerical optimization can produce small
parameter differences: cases 7.2 and 7.6 use `rtol=1e-4`, case 7.3 uses
`rtol=1e-5`, and case 7.4 uses `rtol=1e-8`. Tests also check NIG moment recovery,
likelihood improvement, and invalid inputs.

Sections 4–6 have corresponding implementation and test files:

| Implementation | Main functions | Tests |
| --- | --- | --- |
| lib/chol_psd.py | chol_psd(matrix) | tests/test_chol_psd.py |
| lib/simulations.py | simulate_normal / simulate_pca | tests/test_simulations.py |
| lib/returns_conversion.py | return_calculate | tests/test_returns_conversion.py |

`chol_psd` returns a lower triangular matrix L such that `L @ L.T` reconstructs
the input, including singular positive semidefinite matrices. The final zero
pivot in case 4.1 can be affected by floating-point rounding, so the reference
comparison uses `atol=1e-8`, while reconstruction is checked with `atol=1e-12`.

`simulate_normal(100000, covariance, fix_method=None, seed=545)` returns normal
samples with a population mean of zero, with one simulation per row. Pass
`near_psd` or `higham_nearest_psd` for cases 5.3 and 5.4, respectively.
`simulate_pca(covariance, 100000, explained=0.99, seed=545)` retains the fewest
principal components needed to explain at least 99% of the variance. Python and
Julia generate different random draws, so section 5 does not require exact
elementwise equality. Tests use six standard errors of Gaussian sample
covariance to compare simulations and reference CSVs against theoretical values.
Comparisons between two simulated results account for sampling error in both.

`return_calculate(prices, method="arithmetic", date_column="Date")` accepts a
pandas DataFrame. Use `method="log"` for log returns. Prices must be positive and
finite. Calculations follow the input row order, and the output preserves column
order and the ending date of each return period. Section 6 filenames in Excel
differ slightly from the files on disk; tests use `testout6_1.csv` and
`testout6_2.csv`. Cases 7.1–7.4 have the same naming discrepancy; tests use
`testout7_1.csv` through `testout7_4.csv`. All reference data remains unchanged.

`lib/fix_nonpsd_corr_covar.py` implements `near_psd(matrix)` and
`higham_nearest_psd(matrix)`. Both accept covariance or correlation matrices.
Inputs must be finite, symmetric square matrices with positive diagonal entries.
Covariance matrices are converted to correlation scale before repair, then the
original variances are restored. Higham's convergence checks operate on the
correlation scale, with a default tolerance of `1e-9` and at most 1000 iterations.
Failure to converge raises an exception. Results may have tiny negative
eigenvalues within the specified tolerance.

`lib/expo_weighted_corr_and_covar.py` implements cases 2.1–2.3 through
`ew_covariance`, `ew_correlation`, and `ew_covariance_combined`. Observations must
be ordered from oldest to newest; the last row receives the largest weight.
Weights are normalized and used to compute weighted means, with no additional
`n - 1` correction. These functions require at least two rows of finite numeric
data and do not accept missing values. Case 2.3 follows Excel and the reference
outputs; its comment in `test_setup.jl` reverses the two decay factors.

Sections 1–3 and 6 compare reference outputs element by element using
`rtol=1e-10, atol=1e-12`. Tests also cover missing-data policies, insufficient
observations, constant columns, and pairs without shared observations. Sample
covariance uses `n - 1`; correlation uses the same valid observations for each
pair. Fewer than two valid observations produce NaN, and correlations involving
a constant variable also produce NaN.

Tests.xlsx and the expected-output CSVs are read-only references for the tests.


## Sections 8 and 9

`lib/risk_measures.py` implements Normal, Student t, and empirical VaR/ES.
`normal_risk(mu, sigma)` and `t_risk(mu, sigma, nu)` return four values:
`VaR Absolute`, `VaR Diff from Mean`, `ES Absolute`, and `ES Diff from Mean`.
Losses are reported with the opposite sign from returns; the default lower-tail
probability is 0.05. Student t ES requires nu > 1. Empirical VaR uses NumPy's
linear quantile; empirical ES averages exactly the worst alpha fraction of
observations, using a fractional boundary observation when necessary.

`simulate_t_risk` uses 10,000 draws by default. Cases 8.3 and 8.6 use seeds 8 and
86, respectively. Analytical results are compared with relative tolerance 1e-4
for optimizer differences. Simulated results use absolute tolerances of 0.008
for VaR and 0.02 for ES against the independent Julia simulations. Tests also
check analytical ES against numerical integration and simulation against theory.

`lib/portfolio_risk.py` implements case 9.1. `portfolio_risk(returns, portfolio)`
fits the specified Normal or T marginal for each stock, uses the direct
Spearman correlation convention in `test_setup.jl` for the Gaussian copula,
and simulates 100,000 scenarios. The portfolio input requires `Stock`, `Holding`,
`Starting Price`, and `Distribution` columns and positive current position
values. The report contains each stock and a Total row, with `VaR95`, `ES95`,
`VaR95_Pct`, and `ES95_Pct`. Total risk is calculated from scenario-level summed
P&L, not by adding individual risk measures. Tests allow 4% relative simulation
error and verify position scaling and percentage calculations.

| Case | Calculation | Input | Expected output |
| --- | --- | --- | --- |
| 8.1 | Normal VaR | data/test7_1.csv | data/testout8_1.csv |
| 8.2 | Student t VaR | data/test7_2.csv | data/testout8_2.csv |
| 8.3 | Simulated Student t VaR | data/test7_2.csv | data/testout8_3.csv |
| 8.4 | Normal ES | data/test7_1.csv | data/testout8_4.csv |
| 8.5 | Student t ES | data/test7_2.csv | data/testout8_5.csv |
| 8.6 | Simulated Student t ES | data/test7_2.csv | data/testout8_6.csv |
| 9.1 | Gaussian copula portfolio risk | data/test9_1_returns.csv and data/test9_1_portfolio.csv | data/testout9_1.csv |

The corresponding tests are `tests/test_risk_measures.py` and
`tests/test_portfolio_risk.py`.

## Export actual test results

Run from the repository root:

```bash
python FunctionalTest/testfiles/export_outputs.py
```

This writes 50 actual-result CSV files for cases 1.1 through 13.9 to the repository's
`output/` directory. Filenames and
columns match the reference files on disk. The script calculates results from
input data, chains computed matrices for sections 3 and 4, and never copies
expected-output files. Existing generated files with matching names are replaced;
reference files in `data/` are not modified.

To choose another destination:

```bash
python FunctionalTest/testfiles/export_outputs.py --output-dir /path/to/results
```

`tests/test_export_outputs.py` exports to a temporary directory and checks all
50 files for schema, serialization, and reference agreement. Random results
are reproducible in the same environment but need not be identical to Julia
results. The final Cholesky pivot in a chained calculation can differ by tiny
rounding errors; the export comparison allows an absolute tolerance of 1e-7.

## Sections 10–13

The remaining cases follow the same `lib/` implementation, `tests/` reference
comparison, and actual-result CSV export pattern as sections 1–9.

| Cases | Calculation | Implementation | Tests |
| --- | --- | --- | --- |
| 10.1–10.2 | Equal and custom risk-budget portfolios | lib/portfolio_optimization.py | tests/test_portfolio_optimization.py |
| 10.3–10.4 | Long-only maximum Sharpe portfolios with optional bounds | lib/portfolio_optimization.py | tests/test_portfolio_optimization.py |
| 11.1–11.2 | Buy-and-hold asset and factor attribution | lib/return_attribution.py | tests/test_return_attribution.py |
| 12.1 | European GBSM value and Greeks | lib/option_pricing.py | tests/test_option_pricing.py |
| 12.2 | American value and Greeks with continuous dividends | lib/option_pricing.py | tests/test_option_pricing.py |
| 12.3 | American value with discrete cash dividends | lib/option_pricing.py | tests/test_option_pricing.py |
| 13.1–13.4 | Kendall correlation and rank-based multivariate t fit | lib/multivariate_t.py | tests/test_multivariate_t.py |
| 13.5–13.8 | Gaussian/t copula likelihood, selection, tail dependence | lib/multivariate_t.py | tests/test_multivariate_t.py |
| 13.9 | t-copula portfolio VaR and ES | lib/multivariate_t.py | tests/test_multivariate_t.py |

`risk_parity(covariance, risk_budget=None)` returns fully invested positive
weights, with component variance contributions proportional to the budget.
`max_sharpe(covariance, means, risk_free_rate, bounds=None)` returns fully
invested long-only weights. Bounds are an asset-by-two array of lower and upper
limits, defaulting to `[0, 1]`. The reference for 10.3–10.4 uses the indefinite
matrix from 5.3 directly: its entries are nonnegative and diagonal positive,
so long-only portfolios still have positive variance. The implementation follows
that convention without repairing this matrix. Other indefinite matrices are
rejected. Weight comparisons use `rtol=1e-5, atol=5e-7` to accommodate optimizer
differences and Julia's tiny negative boundary weights. Tests also verify risk
budget shares, investment constraints, and feasible objective perturbations.

`expost_factor(weights, returns, factor_returns, betas, include_alpha=True)`
returns the three-row attribution report. Betas have one row per stock and one
column per factor. Weights drift with asset performance; Carino linking makes
return contributions sum to the compounded portfolio return. Volatility
contributions sum to sample portfolio volatility. Case 11.1 uses identity betas
and omits Alpha. Zero period and total returns use the continuous limit of
`log(1+r)/r`. Both reference reports use `rtol=1e-10, atol=1e-12`.

`gbsm(call, underlying, strike, ttm, rf, carry, volatility)` returns a dictionary
with Value, Delta, Gamma, Vega, Rho, and Theta. Maturity is in years; Vega and Rho
are per unit change, and Theta is per year. European Rho varies the rate and
carry together, keeping the dividend yield fixed. European Theta is `-dV/dT`.
`american(...)` uses a CRR tree, with 500 steps by default and exercise at every
node. `american_greeks(...)` matches the reference's central finite differences,
with a spot displacement of 1.5 for Gamma. For this American report, Rho holds
carry fixed and Theta is **+dV/dT**, as specified in `test_setup.jl`. These
conventions explain the different signs from the European report. American
Greek comparisons use `rtol=2e-5, atol=1e-5` for numerical differentiation;
European and cash-dividend values use `rtol=1e-10, atol=1e-12`.

For cash dividends, supply `dividend_times` as increasing integer tree indices
and `dividend_amounts` as cash amounts. Case 12.3 uses two steps per calendar day.
The tree compares exercise immediately before each dividend with continuation
after subtracting the cash amount; subsequent trees restart from the adjusted
spot. Tests also check put-call parity, analytical Greeks against finite
differences, early exercise, zero dividends, and invalid schedules.

Section 13 computes `rho = sin(pi*tau/2)` from Kendall's tau, repairs the
correlation if needed, and adds a small ridge for positive definiteness. The
multivariate t fit holds sample means and sample standard deviations fixed;
it is the reference's rank-based profile fit. Its scale is
`S = outer(sd, sd) * R * (nu-2)/nu`, so covariance is `nu/(nu-2)*S`.
Degrees of freedom use two 200-point searches over `theta=1/nu`, first over
`[0.01, 0.49]`, then between the coarse winner's neighbors. This reproduces the
reference grid exactly.

Both copulas use the same individually fitted generalized t margins and Kendall
correlation. The Gaussian has zero additional fitted parameters and the t copula
has one (nu), matching the reference AICc/BIC convention. Likelihood and selection
comparisons use `rtol=1e-6` for marginal optimizer differences. Other deterministic
section 13 cases use `rtol=1e-10, atol=1e-12`. Simulated t-copula scenarios share
one chi-square shock across assets in each row. Case 13.9 uses 100,000 scenarios,
seed 13, and a 4% reference tolerance for independent Julia/Python sampling.
Tests check uniform simulated margins, joint tails, scale/covariance conversion,
portfolio position scaling, and the sums of scenario-level P&L.
