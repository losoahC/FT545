# Predictions recorded before model fitting

These predictions were made after inspecting overlap counts, moments, return
plots, P&L histograms, and rank scatter plots, before computing model fits or risk
measures. They are retained even if the fitted results disagree.

1. Pairwise deletion may be indefinite because different entries use different
   samples. Complete-case correlation is PSD. The index's near linear dependence
   makes the matrix close to singular and vulnerable to small estimation errors.
   D-related correlations, especially C-D (83 observations), are least reliable.
2. The final roughly 40 return days (461-500) look more volatile. Predicted VaR
   order: fitted t < historical < equal-weight normal < EW normal 0.97 < EW normal
   0.94. The first two are uncertain: t emphasizes the quieter bulk, whereas the
   normal's variance is increased by the new regime. Recent-weighted estimates
   should be highest. Excess kurtosis 2.13 is inconsistent with a typical sample
   from one constant-volatility normal regime; a variance mixture is plausible.
3. Defaults occur in 406 A scenarios, 393 B scenarios, and 782 union scenarios,
   with 17 simultaneous defaults. Each individual rate is below 5%, but the
   union rate exceeds 5%. Predict higher diversified 5% VaR but lower diversified
   ES: diversified default losses usually involve only half the position. Counts
   alone do not determine ES; the plotted default severity supports that guess.
4. Predict a normal margin for X1 and t margins for X2 and X3. X2's skewness and
   kurtosis depend heavily on row 952; removing it reduces excess kurtosis from
   7.08 to 1.71. A symmetric t is plausible but cannot model true asymmetry.
   Rank pairs have dense joint corners: 6-12 joint extreme days versus 0.625
   under independence. Predict the t copula wins; dependence in the middle and
   the tails should be distinguished by fitting rather than counts alone.
5. With positive residual covariance, ignoring it reduces long-long variance
   and increases long-short variance. Predict underestimated P1 VaR and
   overestimated P2 VaR. P2 should be more sensitive in relative terms because
   similar market betas cancel in a spread.
