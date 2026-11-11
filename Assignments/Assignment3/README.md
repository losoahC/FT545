# Assignment 3

Written responses to all five problems are in **Ans.pdf** and **Ans.md**. The
report includes every numbered Predict/Fit/Reconcile requirement, all quote IVs,
Greek and risk tables, plots, interpretation, and stated numerical limitations.
The original prompt and data are unchanged.

## Reproduce from the repository root

The existing repository virtual environment is detected automatically:

```bash
bash Assignments/Assignment3/Codes/run_all.sh
```

Or run `bash Codes/run_all.sh` from this folder. Paths work from any current
directory. To select another interpreter, use
`PYTHON_BIN=/path/to/python bash Assignments/Assignment3/Codes/run_all.sh`.

For a fresh environment (Python 3.11+):

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -r Assignments/Assignment3/Codes/requirements.txt
bash Assignments/Assignment3/Codes/run_all.sh
```

The command runs descriptive preparation, all calculations, independent numerical
checks, and PDF/Markdown generation. It recreates all report numbers and plots in
`results/` and `PIC/`, and the two written outputs. It preserves the original
`Codes/predictions.md`, which was written before the full fits. A run takes around
a minute on this machine; optimizer and library versions affect timing and final
digits. Do not download this folder alone without the reused FunctionalTest code.

## Code and files

- `Codes/run_assignment.py`: estimates, trees, simulation, optimization, regressions.
- `Codes/validate.py`: pricing, risk, portfolio and regression identities.
- `Codes/build_report.py`: reproducible written interpretation and report layout.
- `Codes/predictions.md`: predictions after allowed diagnostics, before full fits.
- `Codes/requirements.txt`: computational/report dependencies.
- `results/*.csv`: complete calculated tables with more digits than the report.
- `results/run_metadata.json`: simulation count, seeds and environment versions.
- `PIC/`: implied-volatility plot, four P&L curves, efficient frontier/CML.

Reused functions are imported directly from `../../FunctionalTest/testfiles/lib`:

| Source | Used for |
| --- | --- |
| `option_pricing.py` | GBSM values/Greeks and American-tree benchmarks |
| `returns_conversion.py` | Arithmetic stock returns |
| `distribution_fitting.py` | Normal/t fits and AICc |
| `risk_measures.py` | Analytical normal risk and fractional empirical ES |
| `portfolio_optimization.py` | Long-only maximum Sharpe with optional cap |

The assignment-specific non-recombining cash-dividend tree adds a European mode.
It is checked against the original American tree; it never merges nodes across
a dividend. The scenario pricer vectorizes the same student-written GBSM formula.
No third-party option pricing library or reference-output CSV is used.

## Important conventions

- Problem 1 uses the specified November 13 valuation date and 35/365 maturity;
  the report date is November 11. Vega is per unit sigma, cent sensitivities in
  percentage points. Quotes below no-arbitrage bounds have no IV (blank CSV field).
- Problem 2 reports N=200 prices, N=50 through 305 convergence including adjacent
  multiples of 5, and same-grid European/American early exercise premia. The
  finite no-dividend American tree is slightly below the exact European formula;
  this discrepancy is explicitly treated as discretization error.
- Problem 3 has 255 days/year, one million independent ten-day paths, seed 54503,
  zero daily expected returns and fixed IV. Daily Student t scale is not its SD.
  Fit AICc uses normal MLE sigma; delta normal uses sample sigma. The stock hedge
  in the supplied file is approximately, not exactly, delta neutral.
- Problem 4 uses monthly rf=.03/12, 12-times arithmetic annual means, sqrt(12)
  annual SD and Sharpe. Historical risk uses original monthly observations;
  at 1% ES includes 2.4 observations. Bootstrap: 500 IID month samples, seed 54504.
- Problem 5 divides all FF factor/RF percentages by 100, matches all 1,257 dates,
  uses daily excess-return regressions, classic OLS SE, and Gaussian AICc counting
  residual variance. Recent decade is 2014-10-01 through 2024-09-30; annualization
  uses 255. Momentum stress isolates the beta exposure and omits alpha/RF/errors.
- Sample variance/SD use n-1; skew and excess kurtosis use empirical moments.
  Historical quantiles are linear and ES uses exactly the specified tail mass.

## Verification

```bash
venv/bin/python Assignments/Assignment3/Codes/validate.py
venv/bin/python -m pytest -c FunctionalTest/testfiles/pytest.ini \
  FunctionalTest/testfiles/tests/test_option_pricing.py \
  FunctionalTest/testfiles/tests/test_portfolio_optimization.py \
  FunctionalTest/testfiles/tests/test_distribution_fitting.py \
  FunctionalTest/testfiles/tests/test_risk_measures.py \
  FunctionalTest/testfiles/tests/test_returns_conversion.py -q
```

The five existing test modules pass **91 tests**. New checks independently test
European/cash-dividend parity, numerical Greeks, agreement with the existing
American cash tree, fractional ES, tangency conditions, feasible random portfolio
comparisons, and exact omitted-factor alpha accounting. The PDF was rendered and
visually inspected. Reproducibility was checked with Python 3.12.4, NumPy 2.5.2,
pandas 3.0.5 and SciPy 1.18.1; see metadata for the actual computation environment.

The assignment asks for a final **"done"** response in Canvas after repository
submission. No external submission, push, or Canvas action has been performed.
