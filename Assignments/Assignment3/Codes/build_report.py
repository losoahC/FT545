"""Build the submission PDF and Markdown from the reproduced result tables."""
from pathlib import Path
from html import escape
import sys
import numpy as np
import pandas as pd
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                               TableStyle, Image, PageBreak, KeepTogether)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER

ROOT=Path(__file__).resolve().parents[1]
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyCustom',fontName='Helvetica',fontSize=9.5,leading=12.5,spaceAfter=7))
styles.add(ParagraphStyle(name='CellCustom',fontName='Helvetica',fontSize=7.5,leading=9))
styles['Title'].fontSize=23
styles['Heading1'].fontSize=17
styles['Heading2'].fontSize=12
story=[];md=[]


def read(name):return pd.read_csv(ROOT/'results'/f'{name}.csv')
def p(text):
    story.append(Paragraph(escape(text),styles['BodyCustom']));md.append(text+'\n')
def h(text,level=2):
    story.append(Paragraph(escape(text),styles[f'Heading{level}']));md.append('#'*level+' '+text+'\n')
def page():story.append(PageBreak())
def table(frame,formats=None):
    frame=frame.copy();formats=formats or {}
    for col in frame:
        frame[col]=frame[col].map(lambda x: 'n/a' if pd.isna(x) else
                                 (formats[col](x) if col in formats else
                                  f'{x:.5f}' if isinstance(x,(float,np.floating)) else str(x)))
    vals=[list(frame.columns)]+frame.values.tolist()
    cells=[[Paragraph(escape(str(x)),styles['CellCustom']) for x in row] for row in vals]
    widths=[504/len(frame.columns)]*len(frame.columns)
    t=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dce7f1')),
                          ('VALIGN',(0,0),(-1,-1),'TOP'),('BOTTOMPADDING',(0,0),(-1,-1),2.5),
                          ('TOPPADDING',(0,0),(-1,-1),2.5),
                          ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f3f5f7')]),
                          ('LINEBELOW',(0,0),(-1,0),.5,colors.HexColor('#9bafbf'))]))
    story.extend([t,Spacer(1,10)])
    md.append('| '+' | '.join(frame.columns)+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'+
              '\n'.join('| '+' | '.join(map(str,row))+' |' for row in frame.values)+'\n')
def pic(name,height):
    path=ROOT/'PIC'/name
    story.extend([Image(str(path),width=504,height=height),Spacer(1,8)])
    md.append(f'![{name}](PIC/{name})\n')
def pct(x):return f'{100*x:.3f}%'
def num(x):return f'{x:,.2f}'


story.append(Paragraph('Assignment 3',styles['Title']))
story.append(Paragraph('Options, Option Risk, and Portfolio Construction',styles['Heading1']))
p('Haolun Wu | FinTech 545 | November 11, 2026')
h('Methods and conventions')
p('This report answers every Predict, Fit, and Reconcile part. Predictions were recorded in Codes/predictions.md after the allowed descriptive calculations and before the full fits. The calculations import my existing FunctionalTest library directly. I extend my cash-dividend CRR code to European exercise and use a vectorized version of my GBSM formula for simulation.')
p('Returns are arithmetic. Variance and standard deviation use n-1 except normal maximum-likelihood sigma for AICc. Skewness and excess kurtosis use empirical central moments; ordinary kurtosis equals the reported excess plus three. Risk is expressed as positive losses: VaR = -q_alpha(P&L) and ES = minus the average of the worst alpha fraction. Historical ES includes fractional boundary mass. Rates and returns in CSV output are decimals.')
p('Problem 1 uses the assignment valuation date November 13, 2026, even though this report is prepared November 11; maturity is 35/365 years. Problem 2 uses the supplied year fractions. Problem 3 uses 255 trading days/year, 30-day initial options and a 10-day horizon. For Problem 4 I use monthly rf = .03/12, with arithmetic annual means and square-root-of-12 volatility. Problem 5 uses 255 trading days/year, stated explicitly because no annualization count is specified there. Annual expected returns are arithmetic estimates, not guaranteed compounded growth.')
p('One million option-book scenarios use seed 54503, independent daily shocks, and common uniforms for comparing normal and Student t. Both simulated return locations are reset to zero while each fitted scale is retained. Implied volatilities remain fixed; this is a spot-risk exercise, not a volatility-shock model. The 500 month-bootstrap portfolios use seed 54504.')
h('Reproducibility')
p('From the repository root run: bash Assignments/Assignment3/Codes/run_all.sh. This recreates the result tables, plots, numerical validation, Ans.md, and Ans.pdf. Detailed commands, assumptions, library dependencies, and verification are in Assignment3/README.md. The CSVs preserve more precision than the tables printed here.')
p('All original input files remain unchanged. Canvas submission and the final "done" response are separate from this local deliverable.')

page();h('1. Implied volatility from a dirty chain',1)
h('Predict: (a) parity at every strike')
p('For European options, C - P = S exp(-qT) - K exp(-rT). Define gap = C_mid - P_mid - S exp(-qT) + K exp(-rT). I use the prompt\'s flag band, the sum of the full call and put bid-ask spreads. This is deliberately more conservative than the executable half-sum test around mids. The table is in dollars.')
table(read('p1_parity').rename(columns={'Gap':'Parity gap','Band':'Spread band'}),{'Strike':lambda x:f'{x:.0f}','Parity gap':lambda x:f'{x:+.5f}','Spread band':lambda x:f'{x:.2f}'})
p('Only K=175 is flagged: +$0.88818 versus $0.47. At least one side there is stale or erroneous; parity cannot identify the bad quote by itself. This is an internal pricing inconsistency, not evidence of two different economically valid volatilities for the same strike and maturity. Fees, execution, and synchronization still matter for an actual trade.')
h('Predict: (b)-(c) common IV and reliability')
p('At a common sigma, C(sigma)-P(sigma) is exactly the parity forward value. Thus if observed prices satisfy parity and one price has an IV, the other price is reproduced at that same IV. Strictly positive vega makes this solution unique. Small tick-rounding gaps can still lead to noticeably different inverted IVs when vega is tiny.')
p('Since d sigma/d price = 1/vega, I expect the far wings to be least reliable. Deep-ITM calls have very small time value relative to their spreads; far-OTM calls above 210 all show the same $0.05/$0.10 floor despite increasing strikes. Their rising mid-IV can therefore be mechanical. Deep-ITM puts are also unreliable: their price is mostly intrinsic/forward value, leaving little time value to invert.')

page();h('1. Fit: (d)-(e) every quote and vega',1)
p('I invert my GBSM price with Brent root finding. The lower bound is max(S exp(-qT)-K exp(-rT),0) for a call and its sign-reversed counterpart for a put. Quotes below this bound have no nonnegative IV. Upper bounds are S exp(-qT) and K exp(-rT). All mids and asks are valid here. The 140, 145, 150, 155 call bids and 205 through 225 put bids are below their lower bounds and have no IV; n/a below is intentional.')
p('IV columns are percentages. Vega is dollars per 1.00 change in sigma for one option on one share, not dollars per volatility point. The last column is the local change in percentage points of IV for a one-cent price change: 100(0.01/vega). It is a derivative approximation; full bid/ask inversion captures nonlinearity.')
iv=read('p1_iv')
a=iv[['Type','Strike','IVBid','IVMid','IVAsk','Vega','CentVolPoints']]
a=a.rename(columns={'IVBid':'Bid IV %','IVMid':'Mid IV %','IVAsk':'Ask IV %','CentVolPoints':'1 cent: vol pts'})
table(a,{'Strike':lambda x:f'{x:.0f}',**{x:lambda v:f'{100*v:.3f}' for x in ['Bid IV %','Mid IV %','Ask IV %']},'Vega':lambda x:f'{x:.4f}','1 cent: vol pts':lambda x:f'{x:.4f}'})

page();h('1. Reconcile: (f)-(h)',1)
pic('p1_smile.png',252)
p('The shaded regions show bid/ask IV ranges where they exist; a missing bound means no IV at that quote. At K=175, call mid IV is 26.0457% and put mid IV 21.1620%, a 4.8837-point gap. Dividing the parity error by put vega gives 100(.88818/17.40083)=5.1042 vol points; call vega gives 4.7139 points. Using their average vega gives 4.9013 points, close to the actual gap. The difference is expected because vega changes across this finite volatility move. The isolated put dip is the visible parity anomaly; comparison with neighbouring strikes suggests the put quote is the likely bad side, though parity alone is symmetric.')
p('The call wing rises from 22.067% at 210 to 23.823%, 26.689%, and 29.447% at 215, 220, 225. The three latter mids are all $0.075. A one-cent move means .443, .482 and .519 IV points, and their bid/ask ranges are 22.562%-24.827%, 25.314%-27.781%, and 27.965%-30.623%. Those ranges are not an uncertainty estimate for a clean volatility surface: both bid and ask are pinned to the same quote floor, so even the whole interval is distorted. Deep-ITM put IV is more fragile still: at 225 a cent is 3.682 points.')
p('Parity does not catch this wing problem because its dollar gaps (.0355, .0602, .0698) are far smaller than its bands (1.05, 1.20, 1.36). A small OTM price error is a large fraction of time value but tiny beside the ITM put spread. Parity tests cross-contract dollar consistency; it does not certify useful time-value precision.')
p('Ignoring the 175 anomaly and the floored far call wing, the surface is a pronounced downside skew: IV is about 39% at 140, 30% at 160, 25% around 180, and 22% around 205-210. Relative to upside optionality, downside protection commands a premium. This is consistent with insurance demand and downside-tail risk rather than a symmetric lognormal market; the extreme left measurements themselves still have low vega and broad uncertainty.')

page();h('2. American options and discrete dividends',1)
h('Predict: (a)-(b) timing and direction')
table(read('p2_exercise').rename(columns={'Threshold':'K(1-exp(-r*remaining))','ExerciseSpot':'D2 exercise boundary'}),{'Threshold':num})
p('At t=.10, .25 < .89195, so the stated call-exercise condition fails. At t=.40, 1.50 > .22449, so exercise can be optimal sufficiently far in the money. The condition is necessary, not a claim that every call should be exercised. Before pricing, I predicted dividends lower both call values, raise both put values, increase the call exercise premium, and reduce the put premium by making waiting for the stock-price drop more valuable.')
h('Fit: (c) values with and without cash dividends')
table(read('p2_prices'),{k:lambda x:f'{x:.6f}' for k in ['NoDividends','NoDividendTree','Dividends','PVApprox']})
p('NoDividends uses GBSM for Europeans and the N=200 tree for Americans. NoDividendTree gives the common-grid European comparison too. Dividends uses the non-recombining N=200 cash tree in all four cases. The stock jumps down by each cash dividend; I restart each subtree at that ex-dividend spot and retain every resulting branch. American nodes compare exercise at the cum-dividend spot against discounted continuation; European nodes never exercise early.')
p('The PV approximation subtracts .25 exp(-.045*.1)+1.50 exp(-.045*.4) from spot and uses ordinary GBSM. It gives call 3.072293 and put 5.181974, both about .102881 below the cash-tree values. It fixes the forward level but treats the entire dividend-adjusted spot as a geometric Brownian motion from today. Actual cash jumps change the distribution and effective volatility of the residual equity; simply moving spot cannot exactly represent that process.')
p('The N=200 no-dividend American call is .004664 below the exact European formula due to finite-tree error. This is not a negative economic exercise premium. On the identical no-dividend tree the European and American calls coincide exactly; I use that common-grid comparison when reporting early exercise premia.')

page();h('2. Fit: (d)-(f) convergence and sensitivities',1)
table(read('p2_convergence'),{'Call':lambda x:f'{x:.6f}','Put':lambda x:f'{x:.6f}'})
p('Neighbouring N values oscillate rather than converge monotonically. For 100/105, the call differs by .00753 and the put by .01108; for 300/305, the differences shrink to .000255 and .001416. Values are settling near call 3.470 and put 5.371, so N=200 is adequate to a few mills, not to six economic decimals. Extra printed digits expose numerical behaviour rather than claim that precision.')
table(read('p2_greeks'),{'Bump':lambda x:f'{x:g}','Delta':lambda x:f'{x:.8f}','Gamma':lambda x:f'{x:.8f}'})
p('Central spot differences use delta=[V(S+h)-V(S-h)]/(2h) and gamma=[V(S+h)+V(S-h)-2V(S)]/h^2. Both use N=200 and retain the same dividend grid. Delta is broadly stable around .50 for the call and -.56 for the put, but the small-bump gamma is not a converged continuous-model Greek.')
table(read('p2_dividend_sensitivities'),{'Sensitivity':lambda x:f'{x:.6f}'})
p('Dividend sensitivities are dV/dD_i in dollars of option value per extra dollar of dividend. I use a central dividend bump of .001, holding dates and all other inputs fixed. They are derivatives of the stated finite tree; movement of its exercise boundary can make very small differences unstable.')

page();h('2. Reconcile: (g)-(i)',1)
prices=read('p2_prices').set_index('Option');prem=[]
for typ in ['Call','Put']:
    a=prices.loc['American'+typ];e=prices.loc['European'+typ]
    prem.append(dict(Option=typ,NoDividendPremium=a.NoDividendTree-e.NoDividendTree,DividendPremium=a.Dividends-e.Dividends))
table(pd.DataFrame(prem))
p('The call premium rises from zero to .292370; the put premium falls from .125450 on a common grid to .086280. All four value-direction predictions hold. The call loses .450060 with dividends, whereas the European call loses .747094. The put rises in value, but waiting for a known down-jump reduces the incremental benefit of early exercise.')
p('Immediately before the second dividend, the exercise boundary solves S-50 = C_A(S-1.50, K=50, remaining time=.10). Using the remaining 40 steps of the N=200 grid gives S*=52.724852. Above this boundary, exercising captures the cum-dividend stock value; continuing forfeits the dividend. There is no comparable first-dividend call-exercise region under the stated condition. The boundary is a numerical tree estimate and shifts slightly with the grid.')
p('Gamma is the unsettled Greek, most clearly for the call: .04591, .07862, .00458, then approximately zero as h shrinks. At fixed N the backward-induction price is assembled from maxima and weighted sums of piecewise-linear functions of initial spot. A small bump may stay within one linear segment, producing zero gamma, while a larger bump crosses payoff or exercise kinks. A still smaller bump can amplify floating-point cancellation. The put\'s finite differences also move; its nonzero tiny-bump result does not establish convergence. Refine the tree and test spot-bump scales together rather than interpreting h approaching zero at fixed N as the continuous-model gamma.')
p('Per dividend dollar, the American call sensitivity is about -.484 to D1 but only -.0842 to D2. Exercising before D2 protects enough of its in-the-money value from the large late dividend. The small first dividend does not justify exercise, so its effect is transmitted much more directly. European calls lack this escape: their sensitivities -.426 and -.372 are much closer. Thus the muted late-dividend American-call response is economically tied to exercise, not merely discounting.')

page();h('3. VaR of an option book',1)
h('Predict: (a)-(b) book exposures and flat-spot behaviour')
g=read('p3_greeks');table(g,{'Value':num,'Delta':lambda x:f'{x:.4f}','Gamma':lambda x:f'{x:.4f}','Theta':num,'Theta10':num})
p('Spot is $99.12. Greeks are position totals, including the stated share holdings. Theta is the conventional calendar-passage derivative -dV/dT, in dollars per year; Theta10 scales it by 10/255. A position is one option on one share: there is no additional 100-share contract multiplier.')
p('Before simulation I ranked relative delta-normal failure: ShortStraddle, HedgedShortCall, CoveredCall, then RiskReversal. The first two are nearly delta neutral but strongly short gamma; I expected severe understatement. CoveredCall is short gamma with material delta, so downside losses should exceed a linear estimate. RiskReversal has small positive initial gamma and large delta, suggesting a closer fit and possible normal downside overstatement. The first two are essentially tied in relative severity because one is close to half the other\'s gamma exposure.')
p('I predicted positive flat-spot P&L for all four from positive theta, only slightly positive for RiskReversal. I expected ES/VaR notably above the zero-mean normal 5% ratio 1.254 for the short straddle and hedged short call: ordinary quiet moves collect theta, while larger moves pay out on short convexity. The normal ratio is confidence-dependent (about 1.146 at 1%), so 1.25 is a useful 5% reference, not a universal normal constant.')
h('Fit: (c) returns and distribution selection')
table(read('p3_moments'))
table(read('p3_fits'))
p('There are 500 daily returns. The mean is .03214% and standard deviation 1.66743%; skew .019 is nearly zero but excess kurtosis 2.427 signals heavier tails. Normal MLE scale is .01665760; the descriptive/delta-normal sample scale is .01667428. Student t has location .000001182, scale .01350619 and nu=5.78918; its SD is .01669432, since scale is not SD. With k=2 and 3 respectively, AICc is -2671.926 versus -2700.349. Student t wins by 28.423 AICc units; the selected model is t, although both are retained for the requested comparison.')
p('I use AICc=-2 log L+2k+2k(k+1)/(n-k-1), with each model maximized on the same returns. Simulated daily mean is forced to zero by removing fitted location, not by subtracting each scenario\'s realized mean. Ten independent arithmetic shocks are compounded to S_10=S_0 product(1+r_t); options are revalued at 20/255 years with their original IVs. No simulated daily return in this run is below -100%; that boundary is an inherent limitation of unbounded arithmetic-shock models.')

page();h('3. Fit: (d)-(e) delta normal and full revaluation',1)
p('All values below are dollars for the full supplied books. Delta normal sets mean P&L to zero, omits gamma and theta, and uses sigma_P=abs(delta) S_0 sigma_daily sqrt(10). Normal full revaluation uses normal MLE scale; the tiny sample/MLE scale difference is disclosed above and does not explain the large risk gaps. ES averages scenario-level losses in the tail, not stand-alone option risks.')
risk=read('p3_risk')
for alpha in [.05,.01]:
    h(f'{100*alpha:g}% lower tail')
    a=risk[risk.Alpha==alpha][['Book','Model','MeanPnL','VaR','ES']]
    table(a,{'MeanPnL':num,'VaR':num,'ES':num})
p('Monte Carlo standard errors of the mean P&L are .375/.375 for CoveredCall (normal/t), .161/.169 for ShortStraddle, .317/.320 for RiskReversal, and .081/.084 for HedgedShortCall. Tail estimates also have sampling error and are not accurate to the last displayed cent. One million scenarios gives 50,000 observations in the 5% tail and 10,000 in the 1% tail; model and Greek approximation errors are much larger than minor run-to-run sampling differences.')

page();h('3. Fit: (f) nonlinear payoff geometry',1)
pic('p3_pnl.png',353)
p('The solid curves revalue the remaining 20-day European positions, so these are horizon P&L curves rather than expiry payoff diagrams. The dashed lines retain only initial delta. ShortStraddle and HedgedShortCall look flat to the dashed-line model but clearly lose away from spot. CoveredCall loses almost like stock in a fall but gives up upside. RiskReversal changes curvature across its two strikes: a small initial gamma does not describe its far-tail geometry.')
h('Reconcile: (g) what delta misses')
p('The two small delta-normal VaRs are $7.64 for ShortStraddle and $3.82 for HedgedShortCall at 5%; normal full-revaluation VaRs are $316.33 and $149.86. Delta normal sees the slope at one point. It misses the negative term .5 Gamma (Delta S)^2 and the positive Theta Delta t; their competition produces a concentrated gain region and substantial losses for large moves. The supplied hedged call holds 50 shares, so it is only approximately delta neutral: exact initial neutrality would require about 50.4444 shares.')
p('I also evaluate the full second-order expansion delta Delta S + .5 Gamma (Delta S)^2 + Theta(10/255) on the same terminal spots. This improves enormously on delta-only risk but is a local Taylor approximation with frozen Greeks, not a replacement for full revaluation across a third of the remaining option life.')

page();h('3. Reconcile: (g)-(j) tails, carry and model risk',1)
q=read('p3_quadratic');a=q[(q.Model=='StudentT')&(q.Alpha==.01)][['Book','QuadraticVaR','QuadraticES']].merge(risk[(risk.Model=='StudentT')&(risk.Alpha==.01)][['Book','VaR','ES']],on='Book')
table(a, {k:num for k in ['QuadraticVaR','QuadraticES','VaR','ES']})
p('For example, the t straddle quadratic 1% VaR/ES are 669.77/960.36 versus full 638.04/841.44: gamma plus theta captures the mechanism, but overstates ES by about 14%. HedgedShortCall shows the same pattern, 326.59/471.89 versus 310.72/412.42. RiskReversal goes the other way: quadratic ES 799.87 is below full 931.71 because its initially small gamma changes materially as spot enters the put wing. Adding gamma alone without theta would also miss the large quiet-market gains.')
table(read('p3_flat_spot'),{'PnL':num})
p('All flat-spot signs agree with the prediction. The simulated means remain positive but are much lower than flat-spot gains for the short-gamma books: under t the straddle averages 19.93 rather than 143.06, and the hedged call 18.26 rather than 79.83. Short-gamma positive carry compensates for convex losses during large moves; it is not free income. Here realized fitted daily volatility annualizes to about 26.6%, below the .29 IV of the short ATM options. That volatility premium helps explain positive average P&L even after allowing for spot randomness. The positive RiskReversal mean despite small flat-spot carry also reflects nonlinear payoff shape.')
p('Relative to normal full revaluation, t raises 1% ES by about 6.5% for CoveredCall, 12.8% for ShortStraddle, 7.6% for RiskReversal, and 13.1% for HedgedShortCall. At 5%, VaR changes only -.04%, +2.4%, -.04%, and +2.5%, respectively. Similar fitted variances and ten-day aggregation leave the central part fairly alike, while t adds large shocks far into the tails. ES averages precisely those extreme losses and short-gamma books magnify the difference. CoveredCall and RiskReversal have approximately linear far-downside loss growth, so their relative sensitivity is smaller.')
p('The predicted ordering holds qualitatively: near-flat books fail most severely; CoveredCall delta-normal understates downside; RiskReversal is closer. My prediction of possible RiskReversal overstatement works at 5% VaR (527.75 versus normal 491.00), but fails at t 1% VaR and ES (761.78/931.71 versus 746.40/855.12). I missed that local positive gamma changes across a large move. At normal 5%, straddle and hedged-call ES/VaR are 1.533 and 1.563, both above 1.254; under t they are 1.606 and 1.639.')
p('Reply to the risk manager (two sentences): This book is approximately delta neutral at today\'s price, but it is short gamma and can lose heavily after a large move in either direction. Its positive theta pays for that convexity risk, and the fitted-t full revaluation gives 1% VaR about $311 and ES about $412 despite a delta-normal VaR near $5.')

page();h('4. A Sharpe ratio that sells insurance',1)
h('Predict: (a)-(b) moments and the exceptional month')
m=read('p4_moments')
table(m[['Asset','mean','variance','skew','excess_kurtosis','Sharpe','MeanSE']],{'mean':pct,'variance':lambda x:f'{x:.7f}','skew':lambda x:f'{x:.3f}','excess_kurtosis':lambda x:f'{x:.3f}','Sharpe':lambda x:f'{x:.3f}','MeanSE':pct})
p('PUTW looks unusually attractive on annualized Sharpe (.783), yet its skew is -6.525 and excess kurtosis 59.850. Its worst month is June 2008 (-16.7197%), the same month EQ loses 23.5098%. That one observation contributes 45.95% of PUTW\'s total squared deviations and 80.61% of its fourth central-moment sum. Without it, excess kurtosis is still 39.87 and Sharpe rises to 1.272, showing how one omitted crash can make insurance selling look even more attractive. Removing it changes monthly mean from .6243% to .6969%, reducing the mean cost of the observed crash by .07257 percentage points. A sample with only one such event can badly mismeasure its arrival frequency.')
p('Before optimizing, I predicted a large, probably majority PUTW allocation and a historical 1% ES far above a matched normal. A steady premium stream with rare large payouts makes an attractive mean/volatility ratio possible without making the downside approximately Gaussian.')
h('Fit: (c) tangency weights')
table(read('p4_weights'),{k:pct for k in ['EQ','BOND','STK','PUTW']})
p('Unconstrained tangency uses w proportional to inverse(covariance) times (mu-rf), normalized to sum to one. It permits borrowing through short asset weights; the realized solution shorts EQ by 4.39%. The long-only and capped versions maximize the same monthly excess-return/volatility objective under sum(w)=1 and their respective bounds. These constraints describe the investor\'s portfolio; PUTW\'s own cash collateral does not remove the investor\'s exposure to its put payout.')
p('The long-only solution gives PUTW 73.01%, BOND 26.07%, STK .92%, and essentially zero EQ, confirming the prediction. Capping PUTW at 20% shifts weight mainly into BOND (63.62%) and EQ (14.63%). Weights reflect both expected returns and correlations, so individual-asset Sharpe ranking alone would not recover them.')

page();h('4. Fit: (c)-(e) frontier, risk and CAPM',1)
pic('p4_frontier.png',302)
p('The curve minimizes variance at each long-only target mean. Its upper branch above the global minimum-variance point is the efficient frontier; the lower branch is shown to expose the full minimum-variance locus. The capital market line starts at annual rf=3% and is tangent at the long-only maximum-Sharpe portfolio. Annualized returns are 12 times monthly means, and volatility is sqrt(12) times monthly SD. The unconstrained point lies just outside the long-only boundary.')
pr=read('p4_risk')
for alpha in [.05,.01]:
    h(f'{alpha*100:g}% monthly risk: percentage of invested wealth')
    a=pr[pr.Alpha==alpha][['Portfolio','Sharpe','HistoricalVaR','HistoricalES','NormalVaR','NormalES']]
    table(a,{'Sharpe':lambda x:f'{x:.4f}',**{k:pct for k in ['HistoricalVaR','HistoricalES','NormalVaR','NormalES']}})
p('Normal risk uses each portfolio\'s full-sample mean and SD, rather than zero-mean loss. Historical risk is computed directly on the monthly portfolio returns. With n=240, historical 1% ES averages the worst two months and 40% of the third; such a thin tail is necessarily imprecise. A low historical 5% VaR can coexist with very large ES.')

page();h('4. Reconcile: (e)-(h)',1)
table(read('p4_capm'),{'Alpha':pct,'AlphaSE':pct,'AlphaT':lambda x:f'{x:.3f}','Beta':lambda x:f'{x:.4f}','BetaSE':lambda x:f'{x:.4f}','R2':lambda x:f'{x:.4f}'})
p('The all-month CAPM intercept is .27448% monthly, t=3.052, beta=.1848 and R2=.3026. In the 26 months with EQ excess below -5%, beta rises to .9428, with a separate fitted intercept and R2=.9521. The large intercept in this truncated subsample is not a tradable alpha; its role is to fit the stressed branch. I use ordinary OLS standard errors here; nonlinear and nonnormal errors mean the nominal t statistic alone is not a reliable skill test.')
p('The optimizer has bought a concentrated insurance-selling strategy: cash and small premiums in quiet months, substantial stock-like loss exposure after the put enters the money. Mean and variance use only two moments and cannot distinguish this shape from a benign distribution having those moments. Long-only historical 1% ES is 9.205%, versus normal 2.877%, a factor of 3.20; the 20% cap reduces historical 1% ES to 5.491%. Its 5% VaR is higher than the uncapped portfolio\'s because it relinquishes some quiet-month premium smoothing, so VaR alone could encourage the wrong ranking.')
p('Expected Shortfall is coherent: monotonicity recognizes larger scenario losses; positive homogeneity scales a concentrated position\'s loss; translation invariance accounts consistently for added cash; and subadditivity rewards actual diversification rather than allowing risk to be hidden by splitting positions. These properties support aggregation, while tail averaging is what exposes this payoff\'s rare payouts. Coherence does not make an estimate based on 2.4 tail observations accurate. Normal ES is also coherent within a correctly specified model; the observed discrepancy comes from the wrong loss distribution.')
p('PUTW\'s CAPM alpha is not evidence of manager skill. A single linear beta averages a small sensitivity when the put expires worthless with a much larger downside sensitivity when EQ falls. The two estimated betas show the misspecification directly. Compensation for nonlinear downside insurance risk can appear as a positive intercept in that regression.')
h('Mean precision and resampling')
table(m[['Asset','mean','MeanSE']],{'mean':pct,'MeanSE':pct})
table(read('p4_bootstrap'),{k:pct for k in ['P05','Median','P95']})
p('The mean standard errors assume independent months; roughly twice each SE gives a rough 95% band. The mean excess-return estimates in SE units are only about 1.70 for EQ, .95 for BOND, 1.45 for STK and 3.50 for PUTW. Correlations and tail-frequency estimation add further uncertainty, so these 240 months do not justify precise tangency weights.')
p('In 500 IID month resamples, the 5th-95th percentile PUTW weight is 24.38%-99.38%, BOND 0%-70.60%, EQ 0%-13.04%, and STK 0%-8.64%. I would trust the qualitative attraction to PUTW more than the precise 73.01% weight, and impose a concentration limit. The cap itself is a policy choice. IID bootstrap ignores serial dependence and cannot invent unobserved crises; this is a sensitivity check.')

page();h('5. Alpha or factor exposure?',1)
h('Predict: (a)-(b) CAPM and omitted exposures')
p('I merge funds and factors by actual dates, convert YYYYMMDD factor dates, strip the momentum header, and divide every factor and RF percentage by 100. All 1,257 fund dates match from October 1, 1996 through September 28, 2001. Regressions use fund excess returns against zero-investment factor returns. CAPM uses Mkt-RF rather than total market return.')
pre=read('p5_preliminary')
table(pre[['Fund','MeanExcess','Volatility','Sharpe','Alpha','AlphaT','Beta','R2']],{'MeanExcess':pct,'Volatility':pct,'Alpha':pct,'AlphaT':lambda x:f'{x:.3f}','Sharpe':lambda x:f'{x:.3f}','Beta':lambda x:f'{x:.4f}','R2':lambda x:f'{x:.4f}'})
p('Volatility in the table is daily total-return SD; the annualized Sharpe uses the SD of daily excess returns. They are nearly identical, but their definitions differ slightly because RF varies. FUND1 initially looks more skilled: daily alpha .04407%, t=3.204 and annualized Sharpe .848 versus FUND2 .02449%, t=1.835 and Sharpe .750.')
table(read('p5_residual_correlations'),{k:lambda x:f'{x:.4f}' for k in ['SMB','HML','Mom']})
p('FUND1\'s CAPM residual has .8015 correlation with momentum and .2052 with value. I predicted its apparent alpha would fall strongly when these exposures are recognized. FUND2 has size correlation .3295 but little momentum correlation .0530, so I predicted a smaller alpha change and potentially improved precision. Residual correlations diagnose missing variation; sample factor premia determine how much of the intercept changes.')
h('Fit: (c) regression method')
p('For each fund I fit excess return = alpha + beta\' factors + residual, with intercept and unrestricted betas. Coefficient standard errors are conventional homoskedastic OLS SE using RSS/(n-p); alpha t is coefficient/SE. AICc uses the Gaussian MLE residual variance RSS/n and counts p regression coefficients plus one variance parameter: k=3, 5, 6 for CAPM, FF3, Carhart4. All models use identical dates, so criteria are comparable. These SEs describe the stated model; serial correlation or heteroskedasticity would call for robust inference.')

page();h('5. Fit: (c) every coefficient and its SE',1)
coef=read('p5_coefficients')
for fund in ['FUND1','FUND2']:
    h(fund)
    table(coef[coef.Fund==fund][['Model','Term','Coefficient','SE','T']],{'Coefficient':lambda x:f'{x:.8f}','SE':lambda x:f'{x:.8f}','T':lambda x:f'{x:.4f}'})
p('Alpha coefficients are daily decimal returns; factor betas are unitless. For example FUND1 Carhart alpha .000002938 corresponds to .0002938% daily, not .2938%. Extra digits make the small intercept and its uncertainty visible.')

page();h('5. Fit: (c)-(e) selection, expectations and stress',1)
table(read('p5_models'),{'AlphaT':lambda x:f'{x:.4f}','R2':lambda x:f'{x:.5f}','AICc':lambda x:f'{x:.3f}'})
p('FUND1 is decisively better explained by Carhart4: R2 rises to .9563 and alpha t falls to .042. FUND2 gains mainly from size; adding momentum improves R2 only .000065 and worsens AICc by 1.806, so FF3 is narrowly preferred by this criterion. I still use the requested four-factor betas for both forward expectations and stress.')
h('(d) Expected annual return, recent factor premia')
fp=read('p5_factor_premia');table(fp,{'FundSample':pct,'Recent10Years':pct})
p('The recent window is October 1, 2014 through September 30, 2024 (2,516 matched factor days), the most recent calendar decade available in the supplied files. E[R] without alpha = 255(mean RF + beta\' mean factors); with alpha adds 255 alpha_hat. The old fund-sample premia are printed beside the recent ones to make the regime change explicit. The files do not extend to the 2026 assignment date, so I do not extrapolate or silently substitute newer data.')
exp=read('p5_expected')
table(exp[['Fund','ExpectedWithoutAlpha','ExpectedWithAlpha','RealizedArithmetic','RealizedGeometric']],{k:pct for k in ['ExpectedWithoutAlpha','ExpectedWithAlpha','RealizedArithmetic','RealizedGeometric']})
h('(e) Worst historical momentum day')
table(read('p5_momentum_stress'),{'MomentumShock':pct,'FundReturn':pct,'LossDollars':num,'LossSD':lambda x:f'{x:.3f}'})
p('The minimum momentum return is -18.44% on September 5, 1939. Holding the other factors flat, the exposure-only shock is beta_Mom times -18.44%: FUND1 loses $725,708.51 on $10 million, about 6.179 sample daily SDs; FUND2 loses $10,758.42, .150 SD. This is a deterministic factor stress, not a six-sigma normal-probability claim. I omit alpha, RF and residual shocks here to isolate momentum exposure; each is separate from the specified other-factors-flat scenario.')

page();h('5. Reconcile: (f)-(h)',1)
p('The alpha movements agree with the predictions. FUND1 daily alpha falls from .000440660 (CAPM) to .000296309 (FF3) to .000002938 (Carhart4), effectively zero. FUND2 changes modestly from .000244907 to .000277146 to .000272797, with t about 2.15 in Carhart4. A small residual correlation does not by itself guarantee the intercept rises: the negative size premium in this sample is the key sign information for FUND2.')
h('(f) Accounting for FUND1\'s apparent alpha')
account=read('p5_omitted_factor_accounting')
table(account[account.Fund=='FUND1'][['Factor','Beta','SampleMean','MarketAdjustedPremium','AlphaChangeContribution']],{'Beta':lambda x:f'{x:.6f}',**{k:lambda x:f'{x:.9f}' for k in ['SampleMean','MarketAdjustedPremium','AlphaChangeContribution']}})
p('The fund-sample daily factor means are -.01628% SMB, .03717% HML, and .07819% Mom. These premia, especially momentum, reward FUND1\'s .291 value and .394 momentum betas. Because factors correlate with the market, raw beta times mean alone is not an exact intercept decomposition. Regress each omitted factor F_j on market M; let b_jM be its slope. Then alpha_CAPM-alpha_4 = sum beta_j [mean(F_j)-b_jM mean(M)].')
p('For FUND1 the exact daily contributions are -.000000121 from SMB, +.000135851 from HML, and +.000301991 from momentum. They sum to .000437722, exactly the alpha decline .000440660-.000002938. Momentum accounts for about 69.0% and value 31.0% of the decline, with size negligible. The four-factor market beta also rises from .8469 under CAPM to .9492 once correlated exposures are separated. This is exposure being relabelled rather than profitable skill disappearing in real time.')
h('(g) APT interpretation and fee opinion')
p('In an APT-style model, expected excess return is explained by factor loadings times priced factor premia. Alpha is the intercept left over relative to that chosen factor span and those measurement conventions; it is not automatically an arbitrage or an invariant skill parameter. An omitted priced factor, a nonlinear exposure, sampling error, or trading costs can all produce apparent alpha.')
p('Opinion: I would not pay FUND1 a fee for CAPM alpha because almost all of it is explained by value and momentum in Carhart4. FUND2 is the more plausible candidate for an alpha fee: about 6.96% annualized four-factor alpha remains, with t about 2.15, but this five-year estimate is only modest evidence and would need persistence, implementability, fee information and broader risk checks before a commitment. I would assess both with Carhart4 to capture momentum crash exposure and consult FF3\'s slightly better AICc for FUND2; I would not choose a simpler model merely to manufacture a larger fee-worthy intercept.')
h('(h) Realized versus factor-based expected returns')
p('FUND1 realized arithmetic annual return is 20.90%, versus 13.71% without alpha under recent premia. Under the old sample premia its four-factor no-alpha return is 20.83%, almost its realized return. The lower recent momentum premium and value premium account for much of the forward reduction. FUND2 realized arithmetic return is 13.60%, versus 6.89% without alpha or 13.85% if the fitted alpha persists. Its old-sample factor-only return is 6.65%; the unexplained intercept, rather than a high momentum loading, bridges that gap.')
p('For an optimizer, a factor model separates exposures from the assumptions about future premia and can pool noisy mean estimates. It does not eliminate estimation risk: old betas may change, recent factor averages are uncertain, and assuming persistent alpha can dominate the expected-return input. Problem 4\'s concentration issue illustrates why I would test premium and alpha scenarios, shrink uncertain inputs, and retain risk/concentration constraints rather than feed one fitted return vector to the optimizer as truth.')

page();h('6. Code and verification',1)
p('The code is organized as run_assignment.py (all computations), validate.py (independent identities), build_report.py (this PDF and Markdown), predictions.md (the preserved pre-fit reasoning), requirements.txt and run_all.sh. Data is read from Assignment3 and reusable student functions from FunctionalTest/testfiles/lib. No expected-output file is used as an input to an assignment calculation.')
p('Existing functions reused: gbsm and american for European values/Greeks and American benchmarks; return_calculate for arithmetic stock returns; fit_normal, fit_general_t and aicc for model estimation/selection; normal_risk and sample_risk for tail calculations; max_sharpe for bounded portfolios. The cash-tree extension explicitly supports European exercise and is independently cross-checked against my original American cash-dividend implementation.')
p('Validation checks closed-form put-call parity, analytic delta/gamma against finite differences, vectorized GBSM against scalar GBSM, both cash-tree American values against the original library, dividend-adjusted European parity, equality of no-dividend European/American calls on the same grid, correct fractional historical ES, fully invested/long-only/cap constraints, the unrestricted tangency first-order condition, long-only objective against 20,000 feasible portfolios, and the exact omitted-factor alpha decomposition. Relevant existing FunctionalTest option, distribution, return, risk and optimization tests are also run.')
p('The important limitations are explicit throughout: cash-tree Greek kinks and grid error, finite historical tails, classic regression standard errors, constant IV and independent daily shock assumptions, and sensitivity of portfolio weights to estimated means. The numerical results answer the stated models and data; they do not turn these assumptions into certainty.')


def footer(canvas,doc):
    canvas.setStrokeColor(colors.HexColor('#c4cdd4'));canvas.line(54,39,558,39)
    canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#516473'))
    canvas.drawString(54,27,'Haolun Wu | FinTech 545 | Assignment 3')
    canvas.drawRightString(558,27,str(doc.page))


if __name__=='__main__':
    doc=SimpleDocTemplate(str(ROOT/'Ans.pdf'),pagesize=(612,792),rightMargin=54,leftMargin=54,
                          topMargin=48,bottomMargin=52,title='Assignment 3 - Haolun Wu',author='Haolun Wu')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    (ROOT/'Ans.md').write_text('\n'.join(md))
    print('Built Ans.pdf and Ans.md')
