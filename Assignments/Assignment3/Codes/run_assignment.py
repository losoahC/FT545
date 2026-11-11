"""Reproduce Assignment 3 using the student's FunctionalTest library.

Run --prepare first to save the permitted pre-fit diagnostics. Predictions are
written separately in predictions.md; the default run then estimates all models.
All paths are relative to this file, not to the caller's working directory.
"""
from pathlib import Path
import argparse
import json
import sys
import os

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.mpl-cache'))
sys.path.insert(0, str(ROOT.parents[1] / 'FunctionalTest/testfiles/lib'))
import numpy as np
import pandas as pd
from scipy import stats, optimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from option_pricing import gbsm, american
from distribution_fitting import fit_normal, fit_general_t, aicc
from returns_conversion import return_calculate
from risk_measures import normal_risk, sample_risk
from portfolio_optimization import max_sharpe

OUT = ROOT / 'results'
PIC = ROOT / 'PIC'
OUT.mkdir(exist_ok=True)
PIC.mkdir(exist_ok=True)
plt.rcParams.update({'figure.dpi': 150, 'axes.spines.top': False,
                     'axes.spines.right': False, 'font.size': 10})


def save(name, rows):
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(OUT / (name + '.csv'), index=False)
    return frame


def moments(x):
    return dict(n=len(x), mean=np.mean(x), variance=np.var(x, ddof=1),
                sd=np.std(x, ddof=1), skew=stats.skew(x),
                excess_kurtosis=stats.kurtosis(x))


def ols(y, x, names):
    """Gaussian OLS; classic homoskedastic SE, MLE likelihood for AICc.

    Count regression coefficients AND the estimated residual variance in AICc.
    Residual SE and coefficient SE use n-p, rather than the MLE n divisor.
    """
    design = np.column_stack([np.ones(len(y)), x])
    y = np.asarray(y)
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - design @ beta
    n, p = design.shape
    rss = residual @ residual
    se = np.sqrt(np.diag(np.linalg.inv(design.T @ design)) * rss / (n-p))
    ll = -n/2 * (np.log(2*np.pi*rss/n) + 1)
    return dict(coef=dict(zip(['Alpha']+names, beta)),
                se=dict(zip(['Alpha']+names, se)), alpha_t=beta[0]/se[0],
                r2=1-rss/np.sum((y-y.mean())**2), aicc=aicc(ll, n, p+1),
                residual=residual)


def factor_data():
    funds = pd.read_csv(ROOT/'problem5.csv', parse_dates=['Date'])
    ff = pd.read_csv(ROOT/'F-F_Research_Data_Factors_daily.CSV')
    mom = pd.read_csv(ROOT/'F-F_Momentum_Factor_daily.CSV')
    mom.columns = mom.columns.str.strip()
    factors = ff.merge(mom, on='Date', validate='one_to_one')
    factors['Date'] = pd.to_datetime(factors.Date.astype(str), format='%Y%m%d')
    factors.loc[:, factors.columns != 'Date'] /= 100
    joined = funds.merge(factors, on='Date', validate='one_to_one')
    assert len(joined) == len(funds), 'Do not silently drop fund dates'
    return joined, factors


def book_data():
    prices = pd.read_csv(ROOT/'problem3.csv')
    rets = return_calculate(prices, date_column='Date').ABC.to_numpy()
    spot = prices.ABC.iloc[-1]
    positions = pd.read_csv(ROOT/'problem3_portfolios.csv')
    rows = []
    for name, book in positions.groupby('Portfolio', sort=False):
        totals = dict(Value=0., Delta=0., Gamma=0., Theta=0.)
        for p in book.itertuples():
            g = (dict(Value=spot, Delta=1., Gamma=0., Theta=0.) if p.Type == 'Stock'
                 else gbsm(p.Type == 'Call', spot, p.Strike, p.DaysToExpiry/255,
                           .0425, .0425, p.ImpliedVol))
            for k in totals:
                totals[k] += p.Holding*g[k]
        rows.append(dict(Book=name, **totals, Theta10=totals['Theta']*10/255))
    return spot, rets, positions, pd.DataFrame(rows)


def prepare():
    q = pd.read_csv(ROOT/'problem1.csv')
    q['Mid'] = (q.Bid+q.Ask)/2
    q['Spread'] = q.Ask-q.Bid
    c, p = [q[q.Type == t].set_index('Strike') for t in ['Call', 'Put']]
    t = 35/365
    parity = pd.DataFrame({'Strike': c.index, 'Gap': (c.Mid-p.Mid-
                           (182.4*np.exp(-.006*t)-c.index*np.exp(-.0425*t))).values,
                           'Band': (c.Spread+p.Spread).values})
    parity['Flag'] = abs(parity.Gap) > parity.Band
    save('p1_parity', parity)
    _, _, _, greeks = book_data()
    save('p3_greeks', greeks)
    r = pd.read_csv(ROOT/'problem4.csv')
    rf = .03/12  # annual quoted rate converted arithmetically to monthly
    rows=[]
    for asset in r.columns[1:]:
        x=r[asset].to_numpy(); m=moments(x); worst=np.argmin(x)
        rows.append(dict(Asset=asset, **m, Sharpe=np.sqrt(12)*(m['mean']-rf)/m['sd'],
                         MeanSE=m['sd']/np.sqrt(len(x)), WorstMonth=r.Month.iloc[worst],
                         WorstReturn=x[worst], MeanWithoutWorst=np.delete(x,worst).mean(),
                         KurtosisWithoutWorst=stats.kurtosis(np.delete(x,worst)),
                         SharpeWithoutWorst=np.sqrt(12)*(np.delete(x,worst).mean()-rf)/np.std(np.delete(x,worst),ddof=1),
                         WorstShareFourthMoment=(x[worst]-x.mean())**4/np.sum((x-x.mean())**4),
                         WorstShareSquaredDeviations=(x[worst]-x.mean())**2/np.sum((x-x.mean())**2)))
    save('p4_moments', rows)
    data, _ = factor_data(); rows=[]; correlations=[]
    for fund in ['FUND1','FUND2']:
        excess=data[fund]-data.RF
        fit=ols(excess,data[['Mkt-RF']],['Mkt-RF'])
        rows.append(dict(Fund=fund,MeanExcess=excess.mean(),Volatility=data[fund].std(),
                         ExcessVolatility=excess.std(),Sharpe=np.sqrt(255)*excess.mean()/excess.std(),
                         Alpha=fit['coef']['Alpha'],AlphaT=fit['alpha_t'],
                         Beta=fit['coef']['Mkt-RF'],R2=fit['r2']))
        correlations.append(dict(Fund=fund,**{k:np.corrcoef(fit['residual'], data[k])[0,1]
                                               for k in ['SMB','HML','Mom']}))
    save('p5_preliminary',rows); save('p5_residual_correlations',correlations)
    print('Prepared parity, Greeks, moments, CAPM and residual correlations.')


def implied(call, strike, price):
    """Strict European no-arbitrage bounds; no invented IV for invalid quotes."""
    t=35/365; s=182.4; r=.0425; b=r-.006
    low=max(0., (s*np.exp(-.006*t)-strike*np.exp(-r*t))*(1 if call else -1))
    high=s*np.exp(-.006*t) if call else strike*np.exp(-r*t)
    if not low < price < high:
        return np.nan
    return optimize.brentq(lambda v:gbsm(call,s,strike,t,r,b,v)['Value']-price,1e-7,10)


def problem1():
    q=pd.read_csv(ROOT/'problem1.csv'); rows=[]
    for p in q.itertuples():
        mid=(p.Bid+p.Ask)/2; call=p.Type=='Call'
        vols=[implied(call,p.Strike,x) for x in [p.Bid,mid,p.Ask]]
        vega=gbsm(call,182.4,p.Strike,35/365,.0425,.0365,vols[1])['Vega'] if np.isfinite(vols[1]) else np.nan
        low=max(0.,(182.4*np.exp(-.006*35/365)-p.Strike*np.exp(-.0425*35/365))*(1 if call else -1))
        rows.append(dict(Type=p.Type,Strike=p.Strike,Bid=p.Bid,Mid=mid,Ask=p.Ask,
                         LowerBound=low,IVBid=vols[0],IVMid=vols[1],IVAsk=vols[2],
                         Vega=vega,CentVolPoints=100*.01/vega))
    d=save('p1_iv',rows)
    fig,ax=plt.subplots(figsize=(8,4))
    for typ,mark in [('Call','o'),('Put','s')]:
        a=d[d.Type==typ]; ax.plot(a.Strike,100*a.IVMid,marker=mark,label=typ)
        ax.fill_between(a.Strike,100*a.IVBid,100*a.IVAsk,alpha=.14)
    ax.set(xlabel='Strike ($)',ylabel='Implied volatility (%)',title='XYZ European IV: mids and bid/ask bands')
    ax.legend(); fig.tight_layout(); fig.savefig(PIC/'p1_smile.png'); plt.close(fig)


def cash_tree(call, spot=48.5, strike=50., t=.5, r=.045, vol=.30,
              steps=200, dividends=(.25,1.5), exercise=True):
    """Non-recombining CRR tree; branch again after each cash subtraction.

    Each interval recombines, but nodes are never merged across a dividend.
    European mode suppresses ALL early exercise. Dividend dates are exactly
    N/5 and 4N/5. Batch initial nodes to bound peak memory. At an American
    dividend node compare exercise on the cum-dividend spot with continuation
    on the ex-dividend spot. No PV-dividend approximation enters this tree.
    """
    assert steps%5==0
    dt=t/steps; u=np.exp(vol*np.sqrt(dt)); p=(np.exp(r*dt)-1/u)/(u-1/u)
    disc=np.exp(-r*dt); sign=1 if call else -1
    schedule=tuple((j,d) for j,d in zip([steps//5,4*steps//5],dividends) if d)
    def recurse(spots,n,divs):
        end=divs[0][0] if divs else n
        grid=spots[:,None]*u**(2*np.arange(end+1)-end)
        intrinsic=np.maximum(sign*(grid-strike),0)
        if divs:
            date,amount=divs[0]; flat=np.maximum(grid-amount,1e-12).ravel()
            values=np.empty_like(flat)
            for start in range(0,len(flat),256):
                values[start:start+256]=recurse(flat[start:start+256],n-date,
                                               tuple((j-date,d) for j,d in divs[1:]))
            values=values.reshape(grid.shape)
            if exercise: values=np.maximum(intrinsic,values)
        else: values=intrinsic
        for j in range(end-1,-1,-1):
            values=disc*((1-p)*values[:,:-1]+p*values[:,1:])
            if exercise:
                grid=spots[:,None]*u**(2*np.arange(j+1)-j)
                values=np.maximum(values,np.maximum(sign*(grid-strike),0))
        return values[:,0]
    return float(recurse(np.atleast_1d(spot),steps,schedule)[0])


def problem2():
    rows=[]
    for n in [50,55,100,105,150,155,200,205,300,305]:
        row={'Steps':n}
        for call in [True,False]:
            row['Call' if call else 'Put']=cash_tree(call,steps=n)
        rows.append(row)
    save('p2_convergence',rows)
    prices=[]; greeks=[]; sensitivities=[]
    for call in [True,False]:
        for ex in [False,True]:
            name=('American' if ex else 'European')+('Call' if call else 'Put')
            value=cash_tree(call,exercise=ex)
            no=(american(call,48.5,50,.5,.045,.045,.3,steps=200) if ex else gbsm(call,48.5,50,.5,.045,.045,.3)['Value'])
            pv=.25*np.exp(-.045*.1)+1.5*np.exp(-.045*.4)
            approx=gbsm(call,48.5-pv,50,.5,.045,.045,.3)['Value']
            no_tree=cash_tree(call,exercise=ex,dividends=(0,0))
            prices.append(dict(Option=name,NoDividends=no,NoDividendTree=no_tree,Dividends=value,PVApprox=approx if not ex else np.nan))
            for j in range(2):
                h=.001; up=[.25,1.5]; down=up.copy(); up[j]+=h; down[j]-=h
                derivative=(cash_tree(call,exercise=ex,dividends=up)-cash_tree(call,exercise=ex,dividends=down))/(2*h)
                sensitivities.append(dict(Option=name,Dividend=j+1,Sensitivity=derivative))
            if ex:
                for h in [1,.1,.01,.001]:
                    up=cash_tree(call,spot=48.5+h); down=cash_tree(call,spot=48.5-h)
                    greeks.append(dict(Option=name,Bump=h,Delta=(up-down)/(2*h),Gamma=(up+down-2*value)/h**2))
    save('p2_prices',prices);save('p2_greeks',greeks);save('p2_dividend_sensitivities',sensitivities)
    # At t=.4 only .1 years remain. Exercise is evaluated just BEFORE D2.
    boundary=optimize.brentq(lambda s:s-50-american(True,s-1.5,50,.1,.045,.045,.3,steps=40),50.001,100)
    save('p2_exercise', [dict(Dividend=1,Amount=.25,Threshold=50*(1-np.exp(-.045*.4))),
                         dict(Dividend=2,Amount=1.5,Threshold=50*(1-np.exp(-.045*.1)),ExerciseSpot=boundary)])


def option_value(call,s,k,t,v):
    """Vectorized form of our existing GBSM value for large scenario arrays."""
    d1=(np.log(s/k)+(.0425+.5*v*v)*t)/(v*np.sqrt(t)); d2=d1-v*np.sqrt(t)
    z=1 if call else -1
    return z*(s*stats.norm.cdf(z*d1)-k*np.exp(-.0425*t)*stats.norm.cdf(z*d2))


def revalue(book,spots):
    values=np.zeros_like(np.asarray(spots,dtype=float))
    for p in book.itertuples():
        values+=p.Holding*(spots if p.Type=='Stock' else option_value(p.Type=='Call',spots,p.Strike,(p.DaysToExpiry-10)/255,p.ImpliedVol))
    return values


def problem3(nsim):
    spot,rets,positions,g=book_data(); save('p3_moments',[moments(rets)])
    norm=fit_normal(rets); student=fit_general_t(rets)
    # Normal MLE sigma for a fair maximized-likelihood AICc comparison.
    mle=norm['sigma']*np.sqrt((len(rets)-1)/len(rets))
    fits=[dict(Model='Normal',Mu=norm['mu'],Scale=mle,SD=mle,Nu=np.nan,
               AICc=aicc(stats.norm.logpdf(rets,norm['mu'],mle).sum(),len(rets),2)),
          dict(Model='StudentT',Mu=student['mu'],Scale=student['sigma'],
               SD=student['sigma']*np.sqrt(student['nu']/(student['nu']-2)),Nu=student['nu'],
               AICc=aicc(stats.t.logpdf(rets,student['nu'],student['mu'],student['sigma']).sum(),len(rets),3))]
    save('p3_fits',fits); rows=[]; check=[]
    rng=np.random.default_rng(54503)
    # Common uniforms reduce noise when comparing the fitted distributions.
    uniforms=rng.uniform(1e-12,1-1e-12,size=(nsim,10))
    terminals={}
    for model in ['Normal','StudentT']:
        daily=(stats.norm.ppf(uniforms)*mle if model=='Normal' else stats.t.ppf(uniforms,student['nu'])*student['sigma'])
        # Both distributions have location zero; fitted mu is deliberately removed.
        assert np.all(daily>-1), 'Arithmetic return model generated a nonpositive price'
        terminals[model]=spot*np.prod(1+daily,axis=1)
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    for ax,greek in zip(axes.ravel(),g.itertuples()):
        book=positions[positions.Portfolio==greek.Book]
        scale=abs(greek.Delta)*spot*norm['sigma']*np.sqrt(10)
        for alpha in [.05,.01]:
            if scale>0: risk=normal_risk(0,scale,alpha)
            else: risk={'VaR Absolute':0.,'ES Absolute':0.}
            rows.append(dict(Book=greek.Book,Model='DeltaNormal',Alpha=alpha,MeanPnL=0.,
                             VaR=risk['VaR Absolute'],ES=risk['ES Absolute'],MeanSE=0.))
        for model,terminal in terminals.items():
            pnl=revalue(book,terminal)-greek.Value
            ds=terminal-spot
            quadratic=greek.Delta*ds+.5*greek.Gamma*ds**2+greek.Theta*10/255
            for alpha in [.05,.01]:
                risk=sample_risk(pnl,alpha); qr=sample_risk(quadratic,alpha)
                rows.append(dict(Book=greek.Book,Model=model,Alpha=alpha,MeanPnL=pnl.mean(),
                                 VaR=risk['VaR Absolute'],ES=risk['ES Absolute'],MeanSE=pnl.std(ddof=1)/np.sqrt(nsim)))
                check.append(dict(Book=greek.Book,Model=model,Alpha=alpha,
                                  QuadraticVaR=qr['VaR Absolute'],QuadraticES=qr['ES Absolute'],QuadraticMean=quadratic.mean()))
        grid=np.linspace(spot*.65,spot*1.4,501)
        ax.plot(grid,revalue(book,grid)-greek.Value,label='Full revaluation')
        ax.plot(grid,greek.Delta*(grid-spot),'--',label='Delta only')
        ax.axvline(spot,color='grey',lw=.6);ax.axhline(0,color='grey',lw=.6)
        ax.set(title=greek.Book,xlabel='ABC in 10 days ($)',ylabel='Book P&L ($)');ax.legend(fontsize=8)
    save('p3_risk',rows);save('p3_quadratic',check)
    save('p3_flat_spot',[dict(Book=x.Book,PnL=float(revalue(positions[positions.Portfolio==x.Book],np.array([spot]))[0]-x.Value)) for x in g.itertuples()])
    fig.tight_layout();fig.savefig(PIC/'p3_pnl.png');plt.close(fig)


def problem4():
    data=pd.read_csv(ROOT/'problem4.csv'); names=list(data.columns[1:]);r=data[names].to_numpy()
    mu=r.mean(axis=0);cov=np.cov(r,rowvar=False);rf=.03/12;ex=mu-rf
    uncon=np.linalg.solve(cov,ex);uncon/=uncon.sum()
    assert uncon@ex>0
    weights={'Unconstrained':uncon,'LongOnly':max_sharpe(cov,mu,rf),
             'Capped':max_sharpe(cov,mu,rf,bounds=np.array([[0,1],[0,1],[0,1],[0,.2]]))}
    save('p4_weights',[dict(Portfolio=k,**dict(zip(names,w))) for k,w in weights.items()])
    rows=[]
    for name,w in weights.items():
        pnl=r@w
        for alpha in [.05,.01]:
            hist=sample_risk(pnl,alpha);normal=normal_risk(pnl.mean(),pnl.std(ddof=1),alpha)
            rows.append(dict(Portfolio=name,Alpha=alpha,Mean=pnl.mean(),SD=pnl.std(ddof=1),
                             Sharpe=np.sqrt(12)*(pnl.mean()-rf)/pnl.std(ddof=1),
                             HistoricalVaR=hist['VaR Absolute'],HistoricalES=hist['ES Absolute'],
                             NormalVaR=normal['VaR Absolute'],NormalES=normal['ES Absolute']))
    save('p4_risk',rows)
    x=data.EQ-rf;y=data.PUTW-rf;mask=x<-.05;rows=[]
    for label,xx,yy in [('All',x,y),('EQExcessBelowMinus5',x[mask],y[mask])]:
        f=ols(yy,np.asarray(xx)[:,None],['EQ'])
        rows.append(dict(Sample=label,N=len(xx),Alpha=f['coef']['Alpha'],AlphaSE=f['se']['Alpha'],
                         AlphaT=f['alpha_t'],Beta=f['coef']['EQ'],BetaSE=f['se']['EQ'],R2=f['r2']))
    save('p4_capm',rows)
    # Solve minimum variance at target returns to trace the full long-only set.
    frontier=[]
    for target in np.linspace(mu.min(),mu.max(),120):
        sol=optimize.minimize(lambda w:w@cov@w,np.ones(4)/4,jac=lambda w:2*cov@w,
                             bounds=[(0,1)]*4,constraints=[{'type':'eq','fun':lambda w:w.sum()-1},
                             {'type':'eq','fun':lambda w:w@mu-target}],method='SLSQP',options={'ftol':1e-14,'maxiter':1000})
        if sol.success:frontier.append(dict(Mean=target,SD=np.sqrt(sol.x@cov@sol.x)))
    fr=save('p4_frontier',frontier);fig,ax=plt.subplots(figsize=(8,4.8))
    ax.plot(fr.SD*np.sqrt(12)*100,fr.Mean*12*100,label='Long-only frontier')
    sh=(weights['LongOnly']@ex)/np.sqrt(weights['LongOnly']@cov@weights['LongOnly'])
    sd=np.linspace(0,np.sqrt(np.diag(cov)).max()*np.sqrt(12)*100,100)
    ax.plot(sd,3+np.sqrt(12)*sh*sd,'--',label='Capital market line')
    for j,name in enumerate(names):
        ax.scatter(np.sqrt(cov[j,j]*12)*100,mu[j]*1200);ax.annotate(name,(np.sqrt(cov[j,j]*12)*100,mu[j]*1200),xytext=(5,5),textcoords='offset points')
    for name,w in weights.items():
        ax.scatter(np.sqrt(w@cov@w*12)*100,w@mu*1200,marker='x',label=name)
    ax.set(xlabel='Annualized volatility (%)',ylabel='Annualized arithmetic mean (%)',title='Monthly data: mean-variance opportunity set')
    ax.legend(fontsize=8);fig.tight_layout();fig.savefig(PIC/'p4_frontier.png');plt.close(fig)
    # IID bootstrap measures sensitivity of fitted long-only weights to months.
    rng=np.random.default_rng(54504);boots=[]
    for _ in range(500):
        sample=r[rng.integers(0,len(r),len(r))]
        boots.append(max_sharpe(np.cov(sample,rowvar=False),sample.mean(axis=0),rf))
    bs=np.asarray(boots)
    save('p4_bootstrap',[dict(Asset=name,P05=np.quantile(bs[:,j],.05),Median=np.median(bs[:,j]),P95=np.quantile(bs[:,j],.95)) for j,name in enumerate(names)])


def problem5():
    data,factors=factor_data(); models={'CAPM':['Mkt-RF'],'FF3':['Mkt-RF','SMB','HML'],'Carhart4':['Mkt-RF','SMB','HML','Mom']}
    coefficients=[];summaries=[];expected=[];stress=[];premia=[]
    # Calendar trailing decade, inclusive end date. Files end 2024-09-30.
    end=factors.Date.max();recent=factors[(factors.Date>end-pd.DateOffset(years=10))&(factors.Date<=end)]
    names=models['Carhart4'];recent_mean=recent[names+['RF']].mean();sample_mean=data[names+['RF']].mean()
    save('p5_factor_premia',[dict(Factor=k,FundSample=sample_mean[k],Recent10Years=recent_mean[k]) for k in names+['RF']])
    worst=factors.loc[factors.Mom.idxmin()]
    for fund in ['FUND1','FUND2']:
        fitted={}
        for model,cols in models.items():
            f=ols(data[fund]-data.RF,data[cols],cols);fitted[model]=f
            for k,v in f['coef'].items():
                coefficients.append(dict(Fund=fund,Model=model,Term=k,Coefficient=v,SE=f['se'][k],T=v/f['se'][k]))
            summaries.append(dict(Fund=fund,Model=model,N=len(data),AlphaT=f['alpha_t'],R2=f['r2'],AICc=f['aicc']))
        f=fitted['Carhart4'];beta=np.array([f['coef'][k] for k in names]);a=f['coef']['Alpha']
        daily=recent_mean.RF+beta@recent_mean[names].to_numpy()
        realized=data[fund].mean()*255
        expected.append(dict(Fund=fund,RecentStart=recent.Date.min().date(),RecentEnd=end.date(),RecentN=len(recent),
                             ExpectedWithoutAlpha=daily*255,ExpectedWithAlpha=(daily+a)*255,
                             RealizedArithmetic=realized,RealizedGeometric=np.expm1(np.log1p(data[fund]).mean()*255),
                             SampleModelWithoutAlpha=(sample_mean.RF+beta@sample_mean[names].to_numpy())*255))
        shock=f['coef']['Mom']*worst.Mom
        stress.append(dict(Fund=fund,Date=worst.Date.date(),MomentumShock=worst.Mom,FundReturn=shock,
                           LossDollars=-1e7*shock,LossSD=-shock/data[fund].std()))
        # Omitted-variable identity: residualize omitted factors on the market.
        # Alpha_CAPM - Alpha_4 = sum beta_j * (mean(F_j)-slope_j*mean(M)).
        for k in ['SMB','HML','Mom']:
            slope=np.cov(data[k],data['Mkt-RF'],ddof=1)[0,1]/data['Mkt-RF'].var()
            adj=data[k].mean()-slope*data['Mkt-RF'].mean()
            premia.append(dict(Fund=fund,Factor=k,Beta=f['coef'][k],SampleMean=data[k].mean(),
                               MarketAdjustedPremium=adj,AlphaChangeContribution=f['coef'][k]*adj))
    save('p5_coefficients',coefficients);save('p5_models',summaries);save('p5_expected',expected)
    save('p5_momentum_stress',stress);save('p5_omitted_factor_accounting',premia)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--scenarios',type=int,default=1000000)
    args=parser.parse_args()
    if args.prepare:prepare()
    else:
        for label,fn in [('1',problem1),('2',problem2),('3',lambda:problem3(args.scenarios)),('4',problem4),('5',problem5)]:
            print('Problem '+label,flush=True);fn()
        (OUT/'run_metadata.json').write_text(json.dumps(dict(scenarios=args.scenarios,seeds=[54503,54504],python=sys.version,numpy=np.__version__,pandas=pd.__version__),indent=2))
        print('All results and figures saved.')
