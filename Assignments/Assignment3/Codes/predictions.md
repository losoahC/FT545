# Predictions recorded before running the full fits

The --prepare stage has run. Evidence available here is the supplied quotes,
put-call parity, current book Greeks, Problem 4 descriptive moments, and Problem 5
CAPM/residual diagnostics. No implied-volatility inversion, dividend tree, return
distribution fit, simulation, portfolio optimization, or multifactor fit has run.

## 1. Dirty option chain

The 175 strike has parity gap +$0.88818 against the specified $0.47 band. At least
one quote there is inconsistent with the others; parity alone cannot identify
which one. Equal parity-consistent European call/put prices imply equal IVs
because C(sigma)-P(sigma) is independent of sigma and both prices increase with
sigma. Far-OTM calls and puts have low vega, as do their deep-ITM counterparts;
tick rounding and spreads will cause unstable IVs. Calls at 215, 220, 225 share
exactly the same 0.05/0.10 quote despite increasing strikes. I expect an artificial
upturn in their inverted IVs. Apart from this floor and the parity error, I expect
downside skew (higher IV at lower strikes).

## 2. Cash dividends

At t=.1 the interest threshold is 50(1-exp(-.045*.4)) = about .892; .25 is smaller.
At t=.4 the threshold is about .2245; 1.50 is larger. Call exercise can therefore
be optimal just before the second dividend, but the first cannot justify it
under this condition. Dividends should lower both calls and raise both puts.
The call early exercise premium should rise from zero; I expect the put premium
to shrink, since the future drop makes waiting more attractive. The late large
dividend's effect on the American call should be attenuated by early exercise.
Gamma on a fixed tree may depend strongly on the bump.

## 3. Option books

Current spot is 99.12. From largest relative delta-normal failure to smallest:
ShortStraddle, HedgedShortCall, CoveredCall, RiskReversal. The first two have
nearly zero delta (-.889 and -.444), gamma -8.09 and -4.05 and positive 10-day
theta about $130/$73. Delta-normal will severely understate both. CoveredCall
has delta 69.9 and gamma -3.79; I expect downside tail risk to be understated,
though positive theta can offset curvature near the 5% boundary. RiskReversal's
delta 61.4 dominates its small positive gamma .536; I expect a closer fit,
potentially an overstatement of downside risk. Negative-gamma exposure grows
away from spot, so the ranking is a qualitative prediction rather than a theorem.
All four flat-spot P&Ls should be positive (RiskReversal barely so). The straddle
and hedged call should have especially large ES/VaR ratios because extreme moves
lose on convexity even though ordinary moves collect time decay.

## 4. Selling insurance

PUTW's annual Sharpe .783 exceeds EQ .380, BOND .213 and STK .325. Its skew
-6.53 and excess kurtosis 59.85 are conspicuous. One month, June 2008, loses
16.72% and accounts for 45.95% of its squared deviations. Excluding it raises
PUTW's mean from .6243% to .6969% monthly. I expect the long-only optimizer to
give PUTW a large weight, probably a majority. Its historical 1% ES should exceed
the matched normal ES because normal moments spread rare insurance losses too
evenly through the distribution. Mean standard errors are .318%, .087%, .602%,
and .107% per month, so optimized precise weights will be fragile.

## 5. Skill versus exposure

FUND1 initially looks more skilled: annual Sharpe .848, CAPM daily alpha .0441%
and t=3.20, versus FUND2 .750, .0245% and t=1.83. FUND1 CAPM residual correlations
with SMB/HML/Mom are -.061/.205/.801, making momentum the main omitted exposure;
I predict most of FUND1's alpha disappears with momentum. FUND2 correlations
.330/-.121/.053 suggest size/value exposure but less momentum. I expect its
alpha to move less and possibly become more precise after residual variance
falls. Correlation alone does not determine the direction of the intercept
change; sample factor premia must complete the explanation.
