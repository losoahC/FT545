# Functional tests: Sections 1–7

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
