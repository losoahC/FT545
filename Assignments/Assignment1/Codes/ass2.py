# Assignment 1 - Question 2

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.stats import norm, t
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
PIC_DIR = BASE_DIR / "PIC"
PIC_DIR.mkdir(exist_ok=True)

data = pd.read_csv(BASE_DIR / "problem2.csv")

# 1. Predict
# Plot y - x
import matplotlib.pyplot as plt
plt.scatter(data["x"], data["y"])
plt.xlabel("x")
plt.ylabel("y")
plt.title("Scatter plot of y vs x")
plt.savefig(PIC_DIR / 'ASS2-scatter_y_vs_x.png')
plt.close()

print(len(data))

# 2. Fit
# OLS
x = data["x"].to_numpy()
x = sm.add_constant(x)
y = data["y"].to_numpy()
model = sm.OLS(y, x).fit()

alpha_ols, beta_ols = model.params
alpha_se, beta_se = model.bse
print("OLS Estimates:")
print("Alpha:", alpha_ols, "Beta:", beta_ols)
print("Standard Errors:")
print("Alpha SE:", alpha_se, "Beta SE:", beta_se)
print("Residual standard error:", np.sqrt(model.scale))

# MLL
# Normal MLE
def negative_log_likelihood(params):
    alpha, beta, log_sigma = params
    sigma = np.exp(log_sigma)

    residual = y - (alpha + beta * x[:, 1])

    return -np.sum(
        norm.logpdf(residual, loc=0, scale=sigma)
    )

init_normal = [
    alpha_ols,
    beta_ols,
    np.log(np.std(model.resid, ddof=0))
]

normal_fit = minimize(
    negative_log_likelihood,
    init_normal,
    method="BFGS"
)

alpha_normal = normal_fit.x[0]
beta_normal = normal_fit.x[1]
sigma_normal = np.exp(normal_fit.x[2])

# MLE covariance of alpha and beta
XtX_inv = np.linalg.inv(x.T @ x)
cov_normal = sigma_normal**2 * XtX_inv

alpha_normal_se = np.sqrt(cov_normal[0, 0])
beta_normal_se = np.sqrt(cov_normal[1, 1])

print("Normal MLE Estimates:")
print(
    "Alpha:", alpha_normal,
    "Beta:", beta_normal,
    "Sigma:", sigma_normal
)

print("Normal MLE Standard Errors:")
print(
    "Alpha SE:", alpha_normal_se,
    "Beta SE:", beta_normal_se
)

# Student-t MLE 
def student_t_negative_log_likelihood(params):
    alpha, beta, log_scale, log_df_minus_2 = params

    scale = np.exp(log_scale)
    df = 2 + np.exp(log_df_minus_2)

    residual = y - (alpha + beta * x[:, 1])

    return -np.sum(
        t.logpdf(
            residual,
            df=df,
            loc=0,
            scale=scale
        )
    )


init_t = [
    alpha_ols,
    beta_ols,
    np.log(np.std(model.resid, ddof=0)),
    np.log(10 - 2)
]

t_fit = minimize(
    student_t_negative_log_likelihood,
    init_t,
    method="BFGS"
)

alpha_t = t_fit.x[0]
beta_t = t_fit.x[1]
scale_t = np.exp(t_fit.x[2])
df_t = 2 + np.exp(t_fit.x[3])

print("Student-t MLE Estimates:")
print(
    "Alpha:", alpha_t,
    "Beta:", beta_t,
    "Scale:", scale_t,
    "DF:", df_t
)

# Student-t MLE Standard Errors
# Using the inverse of the Hessian from the optimization result
cov_t = t_fit.hess_inv
alpha_t_se = np.sqrt(cov_t[0, 0])
beta_t_se = np.sqrt(cov_t[1, 1])
scale_t_se = np.sqrt(cov_t[2, 2])
df_t_se = np.sqrt(cov_t[3, 3])

print("Student-t MLE Standard Errors:")
print(
    "Alpha SE:", alpha_t_se,
    "Beta SE:", beta_t_se,
    "Scale SE:", scale_t_se,
    "DF SE:", df_t_se
)

def aicc(nll, k, n):
    aic = 2 * k + 2 * nll
    correction = 2 * k * (k + 1) / (n - k - 1)
    return aic + correction

# AICc
loglik_ols = model.llf

k_ols = 3
n = len(y)

aicc_ols = (
    -2 * loglik_ols
    + 2 * k_ols
    + 2 * k_ols * (k_ols + 1) / (n - k_ols - 1)
)

print("OLS AICc:", aicc_ols)

n = len(data)

normal_aicc = aicc(
    normal_fit.fun,
    k=3,
    n=n
)

student_t_aicc = aicc(
    t_fit.fun,
    k=4,
    n=n
)

print("Normal AICc:", normal_aicc)
print("Student-t AICc:", student_t_aicc)

print("Normal error 95% quantile:", norm.ppf(0.95, loc=0, scale=sigma_normal))
print("Normal error 99.5% quantile:", norm.ppf(0.995, loc=0, scale=sigma_normal))
print("Student-t error 95% quantile:", t.ppf(0.95, df=df_t, loc=0, scale=scale_t))
print("Student-t error 99.5% quantile:", t.ppf(0.995, df=df_t, loc=0, scale=scale_t))
