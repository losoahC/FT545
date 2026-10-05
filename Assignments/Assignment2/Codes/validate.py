"""Numerical checks for new copula code and the assignment's key identities.

These checks test mathematical properties and independent density formulas,
not snapshots of the reported numerical answers.
"""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
from scipy import special, stats
from copulas import gaussian_log_density, t_log_density, simulate_uniforms

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE.parents[1] / "FunctionalTest" / "testfiles"))
from lib.simulations import simulate_normal


def main():
    u = np.array([[.13,.42,.91],[.5,.5,.5],[.03,.08,.22]])
    np.testing.assert_allclose(gaussian_log_density(u,np.eye(3)),0,atol=1e-14)
    R = np.array([[1,.6,.4],[.6,1,.3],[.4,.3,1]])
    nu,d = 4.5,3
    z = stats.t.ppf(u,nu)
    q = np.einsum("ni,ij,nj->n",z,np.linalg.inv(R),z)
    # Independent manual multivariate-t formula verifies the density ratio.
    joint = (special.gammaln((nu+d)/2)-special.gammaln(nu/2)
             -d/2*np.log(nu*np.pi)-.5*np.linalg.slogdet(R)[1]
             -(nu+d)/2*np.log1p(q/nu))
    expected = joint-stats.t.logpdf(z,nu).sum(axis=1)
    np.testing.assert_allclose(t_log_density(u,R,nu),expected,atol=1e-12)
    np.testing.assert_allclose(t_log_density(u,R,10000),gaussian_log_density(u,R),atol=.002)
    normals = simulate_normal(100000,R,seed=91)
    for df in [None,nu]:
        sim = simulate_uniforms(normals,df,np.random.default_rng(92))
        # Simulated uniforms must have uniform margins, including for shared-
        # scale t draws. Threshold 0.008 exceeds typical 100k-draw KS noise.
        for j in range(d):
            assert stats.kstest(sim[:,j],"uniform").statistic < .008
        for i,j in [(0,1),(0,2),(1,2)]:
            expected_tau = 2/np.pi*np.arcsin(R[i,j])
            assert abs(stats.kendalltau(sim[:,i],sim[:,j]).statistic-expected_tau)<.015
    summary = json.loads((BASE/"results/summary.json").read_text())
    assert summary["problem5"]["decomposition_error"]<1e-12
    p1=pd.read_csv(BASE/"results/p1_summary.csv",index_col=0).set_index("Method")
    assert p1.loc["Pairwise","Min eigenvalue"]<0
    assert p1.loc["Pairwise","Tracking variance"]<0
    for name in ["Rebonato-Jackel","Higham"]:
        assert p1.loc[name,"Min eigenvalue"]>=-1e-11
        assert p1.loc[name,"Tracking variance"]>=0
    p3=pd.read_csv(BASE/"results/p3_risk.csv",index_col=0).set_index("Position")
    for level in ["5%","1%"]:
        assert p3.loc["$1m each",f"Historical ES {level}"]<=sum(p3.loc[["$1m A","$1m B"],f"Historical ES {level}"])
        for measure in ["VaR","ES"]:
            np.testing.assert_allclose(p3.loc["$2m A",f"Historical {measure} {level}"],2*p3.loc["$1m A",f"Historical {measure} {level}"])
    daily=pd.read_csv(BASE/"results/p4_daily_likelihood.csv",index_col=0)
    np.testing.assert_allclose(daily.Difference.sum(),summary["problem4"]["ll_difference"])
    assert daily["Central fitted uniforms"].sum()==summary["problem4"]["central_days"]
    print("Validation passed: density formulas, Gaussian limit, uniform margins,")
    print("Kendall dependence, PSD repairs, ES subadditivity, scaling, OLS identity,")
    print("and daily likelihood aggregation.")


if __name__ == "__main__":
    main()
