# Assignment 2

This folder answers all five problems in `Assignment 2.pdf`, with predictions,
fits, and reconciliation for every numbered part. The report is in English to
match the assignment prompt. The original data and prompt are unchanged.

## Deliverables

- `Ans.pdf`: submission-ready written response.
- `Ans.md`: written response in Markdown.
- `Codes/predictions.md`: predictions recorded after descriptive inspection and
  before fitting or calculating risk measures.
- `Codes/run_assignment.py`: computations for all five problems.
- `Codes/copulas.py`: assignment-specific Gaussian/t copula likelihood and fitting.
- `Codes/validate.py`: numerical checks of formulas and mathematical properties.
- `results/`: calculated CSV tables, `summary.json`, and all 1,000 daily copula
  log-likelihood contributions.
- `PIC/`: the return, bond P&L, and rank-pair plots embedded in the PDF.

## Python environment

Run from the repository root, using Python 3.11 or later:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -r Assignments/Assignment2/Codes/requirements.txt
```

An existing repository `venv` already contains the computational dependencies on
this machine. `run_all.sh` uses it automatically if available, otherwise `python3`.
To choose an interpreter explicitly, set `PYTHON_BIN`.

## Reproduce every number and plot

From the repository root:

```bash
bash Assignments/Assignment2/Codes/run_all.sh
```

Or from this folder:

```bash
bash Codes/run_all.sh
```

The command runs all computations and validates key mathematical properties.
It regenerates the tables in `results/` and the figures in `PIC/`.
The code resolves data and library paths from its own location, so it can run
from any working directory. For an alternative environment:

```bash
PYTHON_BIN=/path/to/python bash Codes/run_all.sh
```

Individual steps:

```bash
python Codes/run_assignment.py
python Codes/validate.py
```

The final `Ans.pdf` and `Ans.md` are included as written responses. Report
generation scripts and rendering styles are not included in this assignment.

The supplied `assignment.qmd` is the instructor's prompt source, not the answer;
the Julia `Project.toml` is not needed for this Python implementation.

## Existing library functions reused

Code imports these sources directly from `../../FunctionalTest/testfiles/lib`;
do not download only the Assignment2 folder without that library.

| Module | Functions used | Purpose |
| --- | --- | --- |
| `corr_and_covar` | `correlation_missing_data_skip_missing_rows`, `correlation_missing_data_pairwise`, `covariance_missing_data_skip_missing_rows` | Complete/pairwise correlation and sample covariance |
| `fix_nonpsd_corr_covar` | `near_psd`, `higham_nearest_psd` | Rebonato-Jackel and Higham repair |
| `expo_weighted_corr_and_covar` | `ew_covariance` | EW volatility |
| `returns_conversion` | `return_calculate` | Arithmetic returns |
| `distribution_fitting` | `fit_normal`, `fit_general_t`, `aicc` | Distribution fits and model criteria |
| `risk_measures` | `normal_risk`, `t_risk`, `sample_risk` | Analytical and historical VaR/ES |
| `simulations` | `simulate_normal` | Normal/correlated latent simulation |

The simulator also uses the existing `chol_psd` internally. `portfolio_risk`
is not used because its Gaussian/Spearman convention differs from Problem 4's
Kendall-based Gaussian/t comparison. `fit_regression_t` is not used because
Problem 5 specifies normal errors.

## Conventions and verification

- Returns follow the original oldest-to-newest row order; `Day` is metadata.
- Mean, sample variance, empirical skewness, and excess kurtosis define moments.
- Normal margins in Problem 4 use MLE sigma, converted from the library's sample
  sigma before AICc. Other sample covariance estimates use `n-1`.
- EW covariance uses the library's weighted mean and normalized weights.
- Historical VaR is a linear empirical quantile; ES averages the exact worst
  alpha fraction. Negative bond VaR values denote gains, and are retained for
  the subadditivity calculation.
- Problem 4 uses `rank/(n+1)` for descriptive plots, selected parametric CDFs
  for copula likelihood, and `sin(pi*tau/2)` for R. Both copulas hold R fixed;
  criteria count R's three estimated entries plus one extra parameter for t.
- Simulations use 100,000 scenarios and seeds 545 (copula normal scores), 549
  (copula chi-square scales), and 550 (market/errors). The t copula uses one
  shared chi-square scale per scenario.
- Problem 5 omits fitted intercepts from simulations to enforce zero expected
  returns. Residual covariance uses `n-1` for the exact OLS identity; the report
  additionally gives regression residual standard errors with `n-2`.
- Results were verified with Python 3.12.4, NumPy 2.5.2, pandas 3.0.5,
  SciPy 1.18.1, and Matplotlib 3.11.1.
  Fixed seeds reproduce draws within an environment; optimizer/version changes
  can affect last digits and X1's effectively infinite fitted t degrees of freedom.

The new validation checks include an independent manual multivariate-t density,
the Gaussian limit, uniform simulated margins, Kendall dependence, PSD repairs,
ES subadditivity and positive scaling, daily likelihood aggregation, and the
OLS covariance identity.

The 165 relevant existing library tests can be rerun from the repository root:

```bash
python -m pytest -c FunctionalTest/testfiles/pytest.ini \
  FunctionalTest/testfiles/tests/test_corr_and_covar.py \
  FunctionalTest/testfiles/tests/test_expo_weighted_corr_and_covar.py \
  FunctionalTest/testfiles/tests/test_fix_nonpsd_corr_covar.py \
  FunctionalTest/testfiles/tests/test_chol_psd.py \
  FunctionalTest/testfiles/tests/test_returns_conversion.py \
  FunctionalTest/testfiles/tests/test_distribution_fitting.py \
  FunctionalTest/testfiles/tests/test_risk_measures.py \
  FunctionalTest/testfiles/tests/test_simulations.py -q
```

After submitting the repository, the assignment asks you to respond **"done"**
to the question in Canvas. That external submission has not been performed.
