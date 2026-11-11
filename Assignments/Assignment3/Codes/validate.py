"""Independent identities and checks that matter for the assignment results."""
import numpy as np
import pandas as pd
from scipy import stats
from run_assignment import (ROOT, OUT, gbsm, american, cash_tree, option_value,
                            factor_data, ols, sample_risk)


def check():
    # Closed form parity, analytic Greeks, and the scenario-vectorized pricer.
    for s,k,t,v in [(99.12,100,30/255,.29),(182.4,175,35/365,.26)]:
        c=gbsm(True,s,k,t,.0425,.0425,v); p=gbsm(False,s,k,t,.0425,.0425,v)
        assert np.isclose(c['Value']-p['Value'],s-k*np.exp(-.0425*t))
        for call in [True,False]:
            g=gbsm(call,s,k,t,.0425,.0425,v);h=.001
            up=gbsm(call,s+h,k,t,.0425,.0425,v)['Value']
            dn=gbsm(call,s-h,k,t,.0425,.0425,v)['Value']
            assert np.isclose((up-dn)/(2*h),g['Delta'],atol=1e-8)
            assert np.isclose((up+dn-2*g['Value'])/h**2,g['Gamma'],atol=1e-7)
            assert np.isclose(option_value(call,np.array([s]),k,t,v)[0],g['Value'])
    # The new cash tree agrees with the original American implementation, and
    # European parity includes both cash dividends discounted to time zero.
    for call in [True,False]:
        own=cash_tree(call,steps=100)
        existing=american(call,48.5,50,.5,.045,.045,.30,steps=100,
                          dividend_times=[20,80],dividend_amounts=[.25,1.5])
        assert np.isclose(own,existing,atol=1e-12)
        european=cash_tree(call,steps=100,exercise=False)
        assert own>=european
    c=cash_tree(True,steps=100,exercise=False);p=cash_tree(False,steps=100,exercise=False)
    parity=48.5-.25*np.exp(-.045*.1)-1.5*np.exp(-.045*.4)-50*np.exp(-.045*.5)
    assert np.isclose(c-p,parity,atol=1e-9)
    # Non-dividend calls have zero early exercise premium on a common grid.
    assert np.isclose(cash_tree(True,steps=100,dividends=(0,0)),
                      cash_tree(True,steps=100,dividends=(0,0),exercise=False),atol=1e-12)
    # Fractional ES must use 2.4 observations for 240 months at 1%.
    values=np.arange(240.)-100
    assert np.isclose(sample_risk(values,.01)['ES Absolute'],(100+99+.4*98)/2.4)
    weights=pd.read_csv(OUT/'p4_weights.csv').set_index('Portfolio')
    assert np.allclose(weights.sum(axis=1),1)
    assert weights.loc['LongOnly'].min()>-1e-10
    assert weights.loc['Capped','PUTW']<=.200000001
    r=pd.read_csv(ROOT/'problem4.csv').iloc[:,1:].to_numpy()
    cov=np.cov(r,rowvar=False);ex=r.mean(axis=0)-.03/12
    w=weights.loc['Unconstrained'].to_numpy()
    # The unrestricted tangency direction solves Cw proportional to mu-rf.
    assert np.allclose((cov@w)/ex,np.repeat((cov@w/ex)[0],4))
    rng=np.random.default_rng(123);random_w=rng.dirichlet(np.ones(4),size=20000)
    ratios=random_w@ex/np.sqrt(np.einsum('ij,jk,ik->i',random_w,cov,random_w))
    lw=weights.loc['LongOnly'].to_numpy()
    assert (lw@ex)/np.sqrt(lw@cov@lw)>=ratios.max()-1e-9
    # Exact omitted-variable intercept identity on the same matched observations.
    data,_=factor_data();account=pd.read_csv(OUT/'p5_omitted_factor_accounting.csv')
    for fund in ['FUND1','FUND2']:
        y=data[fund]-data.RF
        cap=ols(y,data[['Mkt-RF']],['Mkt-RF'])
        four=ols(y,data[['Mkt-RF','SMB','HML','Mom']],['Mkt-RF','SMB','HML','Mom'])
        assert np.isclose(cap['coef']['Alpha']-four['coef']['Alpha'],
                          account[account.Fund==fund].AlphaChangeContribution.sum(),atol=1e-14)
        assert abs(four['residual'].mean())<1e-14
    risk=pd.read_csv(OUT/'p3_risk.csv')
    assert (risk.ES>=risk.VaR).all()
    assert (risk[risk.Model!='DeltaNormal'].MeanPnL>0).all()
    print('Passed pricing/Greeks, cash-tree parity and cross-checks, ES, tangency, constraints, and OLS identities.')


if __name__=='__main__':check()
