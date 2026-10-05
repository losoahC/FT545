"""Reproduce all five problems, using the repository's tested library.

Run from any working directory. Dollar positions are investment values, not
share counts. Predictions are frozen in predictions.md before model fitting.
"""
from pathlib import Path
import json
import os
import sys

BASE = Path(__file__).resolve().parents[1]
REPO = BASE.parents[1]
sys.path.insert(0, str(REPO / "FunctionalTest" / "testfiles"))
# Matplotlib must use a writable cache and a non-interactive backend.
os.environ.setdefault("MPLCONFIGDIR", str(BASE / ".mpl-cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from lib.corr_and_covar import (
    correlation_missing_data_pairwise, correlation_missing_data_skip_missing_rows,
    covariance_missing_data_skip_missing_rows,
)
from lib.fix_nonpsd_corr_covar import near_psd, higham_nearest_psd
from lib.expo_weighted_corr_and_covar import ew_covariance
from lib.returns_conversion import return_calculate
from lib.distribution_fitting import fit_normal, fit_general_t, aicc
from lib.risk_measures import normal_risk, t_risk, sample_risk
from lib.simulations import simulate_normal
from copulas import gaussian_log_density, t_log_density, fit_t_copula, simulate_uniforms

RESULTS = BASE / "results"
PICS = BASE / "PIC"
N_SIM = 100_000
SEED = 545
PAIRS = [(0, 1), (0, 2), (1, 2)]


def table(name, values, **kwargs):
    frame = values if isinstance(values, pd.DataFrame) else pd.DataFrame(values, **kwargs)
    frame.to_csv(RESULTS / (name + ".csv"), index=True)
    return frame


def moments(data):
    """Mean, sample variance, and standardized third/fourth central moments.

    Skewness and excess kurtosis use bias=True (empirical central moments),
    matching the repository's moment conventions. Raw kurtosis = excess + 3.
    """
    return pd.DataFrame({c: {"Mean": data[c].mean(), "Variance": data[c].var(ddof=1),
                            "Skewness": stats.skew(data[c], bias=True),
                            "Excess kurtosis": stats.kurtosis(data[c], fisher=True, bias=True)}
                         for c in data}).T


def risk(values, alpha=0.05):
    measured = sample_risk(values, alpha)
    return {"VaR": measured["VaR Absolute"], "ES": measured["ES Absolute"]}


def load(i):
    return pd.read_csv(BASE / f"problem{i}.csv")


def save_plot(fig, name):
    fig.tight_layout()
    fig.savefig(PICS / (name + ".png"), dpi=180)
    plt.close(fig)


def problem1():
    data = load(1).drop(columns="Day")
    columns = data.columns
    observed = data.notna().astype(int)
    table("p1_overlap", observed.T @ observed)
    complete = correlation_missing_data_skip_missing_rows(data)
    pairwise = correlation_missing_data_pairwise(data)
    std = data.std(ddof=1).to_numpy()  # Each series uses its entire observed history.
    weights = np.array([-0.4, -0.3, -0.2, -0.1, 1.0])
    repaired = {"Complete case": complete, "Pairwise": pairwise,
                "Rebonato-Jackel": near_psd(pairwise),
                "Higham": higham_nearest_psd(pairwise, tolerance=1e-12)}
    rows = []
    for name, corr in repaired.items():
        table("p1_corr_" + name.lower().replace(" ", "_"), corr, index=columns, columns=columns)
        eigen = np.linalg.eigvalsh(corr)
        covariance = corr * np.outer(std, std)
        # Ordinary Cholesky requires positive definiteness, not merely PSD.
        try:
            np.linalg.cholesky(corr)
            cholesky = "success"
        except np.linalg.LinAlgError:
            cholesky = "fails: matrix is not positive definite"
        rows.append({"Method": name, "Min eigenvalue": eigen[0],
                     "Distance from pairwise": np.linalg.norm(corr - pairwise, "fro"),
                     "Tracking variance": weights @ covariance @ weights,
                     "Cholesky": cholesky})
        table("p1_eigen_" + name.lower().replace(" ", "_"), {"Eigenvalue": eigen})
    table("p1_summary", rows)
    higham = repaired["Higham"]
    moves = [{"Pair": columns[i] + "-" + columns[j], "Overlap": int((observed.T @ observed).iloc[i,j]),
              "Original": pairwise[i,j], "Higham": higham[i,j], "Change": higham[i,j]-pairwise[i,j]}
             for i in range(5) for j in range(i+1, 5)]
    table("p1_higham_moves", pd.DataFrame(moves).sort_values("Change", key=abs, ascending=False))
    table("p1_full_history_std", {"Std": std}, index=columns)
    return {"complete_days": int(observed.all(axis=1).sum()),
            "estimator_gap": np.linalg.norm(complete-pairwise, "fro"),
            "repair_gap": np.linalg.norm(higham-repaired["Rebonato-Jackel"], "fro")}


def problem2():
    returns = return_calculate(load(2), date_column="Day")
    raw_mean = returns.Price.mean()
    x = returns.Price.to_numpy() - raw_mean
    table("p2_returns", {"Day": returns.Day, "Demeaned return": x})
    table("p2_moments", moments(pd.DataFrame({"Return": x})))
    # Boundary selected visually before fitting: last 40 of 500 returns.
    recent = 40
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(returns.Day, x, linewidth=0.8)
    ax.axvline(460.5, color="darkred", linestyle="--", label="Visual regime boundary")
    ax.set(xlabel="Return day", ylabel="Demeaned arithmetic return", title="Problem 2: volatility regime change")
    ax.legend(fontsize=8)
    save_plot(fig, "problem2_returns")
    normal = fit_normal(x)
    fitted_t = fit_general_t(x)
    rows = [{"Method": "Equal-weight normal", "Std or scale": normal["sigma"],
             "VaR": 1e6*normal_risk(0,normal["sigma"])["VaR Absolute"]}]
    ew_rows = []
    for decay in [0.97, 0.94]:
        weights = decay ** np.arange(len(x)-1, -1, -1)
        weights /= weights.sum()
        sigma = np.sqrt(ew_covariance(x[:,None], decay)[0,0])
        neff = 1 / np.sum(weights**2)
        half_life = np.log(0.5) / np.log(decay)
        se_sigma = sigma/np.sqrt(2*neff)
        rows.append({"Method": f"EW normal {decay}", "Std or scale": sigma,
                     "VaR": 1e6*normal_risk(0,sigma)["VaR Absolute"]})
        ew_rows.append({"Lambda": decay, "Effective sample": neff, "Half life": half_life,
                        "Recent 40 weight": weights[-recent:].sum(), "Sigma": sigma,
                        "Sigma SE": se_sigma, "VaR SE": -1e6*stats.norm.ppf(.05)*se_sigma})
    rows += [{"Method": "Fitted t", "Std or scale": fitted_t["sigma"],
              "VaR": 1e6*t_risk(fitted_t["mu"], fitted_t["sigma"], fitted_t["nu"])["VaR Absolute"]},
             {"Method": "Historical", "Std or scale": np.nan, "VaR": risk(1e6*x)["VaR"]}]
    table("p2_var", rows)
    table("p2_ew", ew_rows)
    regimes = pd.DataFrame([{"Regime": "Days 1-460", "Days": 460, "Mean": x[:-40].mean(), "Std": x[:-40].std(ddof=1)},
                            {"Regime": "Days 461-500", "Days": 40, "Mean": x[-40:].mean(), "Std": x[-40:].std(ddof=1)}])
    table("p2_regimes", regimes)
    # A zero-mean Gaussian variance mixture has raw kurtosis 3 E[v^2]/E[v]^2.
    variances = regimes.Std.to_numpy()**2
    fractions = regimes.Days.to_numpy()/len(x)
    mixture_excess = 3*np.sum(fractions*variances**2)/np.sum(fractions*variances)**2 - 3
    return {"raw_mean": raw_mean, "t": fitted_t, "recent_days": recent,
            "normal_mixture_excess": mixture_excess}


def problem3():
    data = load(3)[["A", "B"]]
    table("p3_moments", moments(data))
    defaults = data < -0.2
    pnl = {"$1m A": 1e6*data.A, "$1m B": 1e6*data.B,
           "$2m A": 2e6*data.A, "$1m each": 1e6*(data.A+data.B)}
    rows = []
    for label, values in pnl.items():
        fitted = fit_normal(values)
        rows.append({"Position": label, "Historical VaR 5%": risk(values)["VaR"],
                     "Historical ES 5%": risk(values)["ES"],
                     "Normal VaR 5%": normal_risk(fitted["mu"],fitted["sigma"])["VaR Absolute"],
                     "Historical VaR 1%": risk(values,.01)["VaR"],
                     "Historical ES 1%": risk(values,.01)["ES"]})
    table("p3_risk", rows)
    fig, axs = plt.subplots(1,2,figsize=(10,3.4))
    for ax, label in zip(axs,["$2m A", "$1m each"]):
        ax.hist(pnl[label]/1e6,bins=80)
        ax.set(xlabel="P&L ($ millions)",ylabel="Scenario count",title=label)
    save_plot(fig,"problem3_pnl")
    return {"A_defaults": int(defaults.A.sum()), "B_defaults": int(defaults.B.sum()),
            "union_defaults": int(defaults.any(axis=1).sum()), "both_defaults": int(defaults.all(axis=1).sum())}


def tail_counts(u, scale=1):
    return [{"Pair": f"X{i+1}-X{j+1}",
             "Lower": scale*np.sum((u[:,i]<.025)&(u[:,j]<.025)),
             "Upper": scale*np.sum((u[:,i]>.975)&(u[:,j]>.975))} for i,j in PAIRS]


def problem4():
    data = load(4)
    x = data.to_numpy()
    n, d = x.shape
    table("p4_moments", moments(data))
    sensitivity = []
    for column in data:
        values = data[column]
        index = int(np.argmax(abs(values-values.mean())))
        reduced = values.drop(index)
        sensitivity.append({"Series": column,"Removed row (1-based)":index+1,"Removed value":values.iloc[index],
                            "Skewness without":stats.skew(reduced),"Excess kurtosis without":stats.kurtosis(reduced)})
    table("p4_sensitivity", sensitivity)
    ranked = data.rank().to_numpy()/(n+1)
    table("p4_rank_tails",tail_counts(ranked))
    fig, axs = plt.subplots(1,3,figsize=(11,3.4))
    for ax,(i,j) in zip(axs,PAIRS):
        ax.scatter(ranked[:,i],ranked[:,j],s=5,alpha=.5)
        ax.set(xlabel=f"X{i+1} rank / (n+1)",ylabel=f"X{j+1} rank / (n+1)")
    save_plot(fig,"problem4_rank_pairs")
    fits, rows, uniforms = [], [], np.empty_like(x)
    for j,column in enumerate(data):
        normal = fit_normal(x[:,j])
        # The library returns the sample sigma. Convert it to the normal MLE
        # before evaluating likelihood or AICc (normal has two free parameters).
        normal["sigma"] *= np.sqrt((n-1)/n)
        fitted_t = fit_general_t(x[:,j])
        candidates = [("Normal",normal,stats.norm(normal["mu"],normal["sigma"]),2),
                      ("t",fitted_t,stats.t(fitted_t["nu"],fitted_t["mu"],fitted_t["sigma"]),3)]
        scored = []
        for name,parameters,distribution,k in candidates:
            ll = distribution.logpdf(x[:,j]).sum()
            ic = aicc(float(ll),n,k)
            scored.append((ic,name,parameters,distribution))
            rows.append({"Margin":column,"Model":name,"Mu":parameters["mu"],"Scale":parameters["sigma"],
                         "Nu":parameters.get("nu",np.nan),"Log likelihood":ll,"AICc":ic})
        _,name,parameters,distribution = min(scored,key=lambda row:row[0])
        fits.append({"name":name,"parameters":parameters,"distribution":distribution})
        uniforms[:,j] = distribution.cdf(x[:,j])
    table("p4_margin_fits",rows)
    # Avoid infinities in inverse CDFs. Clipping is a numerical safeguard only.
    uniforms = np.clip(uniforms,1e-12,1-1e-12)
    tau = np.eye(d)
    for i,j in PAIRS:tau[i,j]=tau[j,i]=stats.kendalltau(x[:,i],x[:,j]).statistic
    correlation = np.sin(np.pi*tau/2)
    if np.linalg.eigvalsh(correlation)[0]<=0:
        raise RuntimeError("Kendall-derived R is not positive definite")
    table("p4_kendall_tau",tau,index=data.columns,columns=data.columns)
    table("p4_R",correlation,index=data.columns,columns=data.columns)
    nu = fit_t_copula(uniforms,correlation)
    log_gauss = gaussian_log_density(uniforms,correlation)
    log_t = t_log_density(uniforms,correlation,nu)
    # Count R's three estimated off-diagonal parameters for BOTH models, plus
    # nu for t. Marginal parameters are common and excluded from copula scores.
    scores = []
    for name,logs,k in [("Gaussian",log_gauss,3),("t",log_t,4)]:
        ll = float(logs.sum())
        scores.append({"Copula":name,"Nu":np.inf if name=="Gaussian" else nu,
                       "Parameters":k,"Log likelihood":ll,"AICc":aicc(ll,n,k),"BIC":k*np.log(n)-2*ll})
    table("p4_copula_fits",scores)
    # Simulation uses the existing normal simulator to produce correlated
    # scores, then one shared random scale per t-copula scenario.
    normal_scores = simulate_normal(N_SIM,correlation,seed=SEED)
    rng = np.random.default_rng(SEED+4)
    risk_rows = []
    tails = []
    for name,df in [("Gaussian",None),("t",nu)]:
        simulated_u = simulate_uniforms(normal_scores,df,rng)
        simulated_u = np.clip(simulated_u,1e-12,1-1e-12)
        simulated = np.column_stack([fits[j]["distribution"].ppf(simulated_u[:,j]) for j in range(d)])
        pnl = 1e6*simulated.sum(axis=1)
        values = {"Model":name}
        for alpha in [.05,.01]:
            for key,value in risk(pnl,alpha).items():values[f"{key} {int(alpha*100)}%"] = value
        risk_rows.append(values)
        # Copula-uniform thresholds are each simulated margin's population
        # quantiles. Scale frequencies to 1000 days for comparison with ranks.
        for row in tail_counts(simulated_u,n/N_SIM):tails.append({"Model":name,**row})
    values = {"Model":"Historical"}
    for alpha in [.05,.01]:
        for key,value in risk(1e6*x.sum(axis=1),alpha).items():values[f"{key} {int(alpha*100)}%"] = value
    risk_rows.append(values)
    table("p4_risk",risk_rows)
    table("p4_simulated_tails",tails)
    difference = log_t-log_gauss
    # Primary definition follows fitted uniforms used for likelihood. Also
    # report rank-based classification because the prompt could mean empirical tails.
    central = ((uniforms>.05)&(uniforms<.95)).all(axis=1)
    rank_central = ((ranked>.05)&(ranked<.95)).all(axis=1)
    table("p4_daily_likelihood",{"Row":np.arange(1,n+1),"Gaussian log density":log_gauss,
                               "t log density":log_t,"Difference":difference,
                               "Central fitted uniforms":central,"Central empirical ranks":rank_central})
    i,j = max(PAIRS,key=lambda pair:correlation[pair])
    rho = correlation[i,j]
    tail_dependence = 2*stats.t.cdf(-np.sqrt((nu+1)*(1-rho)/(1+rho)),df=nu+1)
    return {"selected_margins":[f["name"] for f in fits],"nu":nu,"R_min_eigen":np.linalg.eigvalsh(correlation)[0],
            "ll_difference":difference.sum(),"central_days":int(central.sum()),
            "central_difference":difference[central].sum(),"rank_central_days":int(rank_central.sum()),
            "rank_central_difference":difference[rank_central].sum(),
            "tail_pair":f"X{i+1}-X{j+1}","tail_rho":rho,"tail_dependence":tail_dependence}


def problem5():
    returns = return_calculate(load(5),date_column="Day")
    x = returns.MKT.to_numpy()
    y = returns[["A","B"]].to_numpy()
    design = np.column_stack([np.ones(len(x)),x])
    coefficients = np.linalg.lstsq(design,y,rcond=None)[0]
    residuals = y-design@coefficients
    # All covariance estimates use n-1. Using n-2 for residual variances would
    # prevent the exact sample covariance decomposition checked below.
    residual_cov = covariance_missing_data_skip_missing_rows(residuals)
    stock_cov = covariance_missing_data_skip_missing_rows(y)
    beta = coefficients[1]
    market_var = x.var(ddof=1)
    model_cov = np.outer(beta,beta)*market_var+residual_cov
    residual_corr = residual_cov[0,1]/np.sqrt(np.prod(np.diag(residual_cov)))
    table("p5_regressions",{"Alpha":coefficients[0],"Beta":beta,
                           "Residual std (n-1)":np.sqrt(np.diag(residual_cov)),
                           "Residual std (n-2)":np.sqrt(np.sum(residuals**2,axis=0)/(len(x)-2))},index=["A","B"])
    table("p5_residual_cov",residual_cov,index=["A","B"],columns=["A","B"])
    table("p5_stock_cov",stock_cov,index=["A","B"],columns=["A","B"])
    rows = []
    for name,errors_cov in [("Full residual covariance",residual_cov),
                            ("Independent residuals",np.diag(np.diag(residual_cov)))]:
        covariance = np.zeros((3,3));covariance[0,0]=market_var;covariance[1:,1:]=errors_cov
        draws = simulate_normal(N_SIM,covariance,seed=SEED+5)
        # Set expectations to zero as instructed: report fitted intercepts but
        # do not add them to simulated returns. Market and errors are centered.
        simulated = draws[:,[0]]*beta+draws[:,1:]
        theoretical = np.outer(beta,beta)*market_var+errors_cov
        for portfolio,w in [("P1",np.array([1e6,1e6])),("P2",np.array([1e6,-1e6]))]:
            rows.append({"Assumption":name,"Portfolio":portfolio,"Simulated VaR":risk(simulated@w)["VaR"],
                         "Analytical model VaR":normal_risk(0,np.sqrt(w@theoretical@w))["VaR Absolute"],
                         "Delta normal VaR":normal_risk(0,np.sqrt(w@stock_cov@w))["VaR Absolute"]})
    table("p5_var",rows)
    return {"residual_correlation":residual_corr,"market_std":np.sqrt(market_var),
            "decomposition_error":abs(stock_cov-model_cov).max(),
            "market_residual_cross_cov":np.cov(np.column_stack([x,residuals]),rowvar=False)[0,1:].tolist()}


def main():
    RESULTS.mkdir(exist_ok=True);PICS.mkdir(exist_ok=True)
    summary = {"simulation_draws":N_SIM,"seed":SEED}
    for i,function in enumerate([problem1,problem2,problem3,problem4,problem5],1):
        summary[f"problem{i}"] = function()
        print(f"Problem {i} complete",flush=True)
    (RESULTS/"summary.json").write_text(json.dumps(summary,indent=2,default=lambda x:float(x))+"\n")
    print("All tables, figures and summary written to",BASE)


if __name__ == "__main__":
    main()
