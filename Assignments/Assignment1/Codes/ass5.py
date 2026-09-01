# Assignment 1 - Question 5

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, pacf

BASE_DIR = Path(__file__).resolve().parents[1]
PIC_DIR = BASE_DIR / "PIC"
PIC_DIR.mkdir(exist_ok=True)

data = pd.read_csv(BASE_DIR / "problem5.csv")
x = data["x"].to_numpy()
n = len(x)

significance_band = 1.96 / np.sqrt(n)
print("Approximate ACF/PACF significance band:", significance_band)

acf_values = acf(x, nlags=10, fft=False)
pacf_values = pacf(x, nlags=10, method="ywm")

print("ACF lags 1-10:")
for lag, value in enumerate(acf_values[1:], start=1):
    print(lag, value)

print("PACF lags 1-10:")
for lag, value in enumerate(pacf_values[1:], start=1):
    print(lag, value)

fig, axes = plt.subplots(3, 1, figsize=(9, 9))
axes[0].plot(np.arange(n), x, linewidth=1)
axes[0].axhline(x.mean(), color="crimson", linewidth=1, alpha=0.8)
axes[0].set_title("problem5 x series")
axes[0].set_xlabel("Index")
axes[0].set_ylabel("x")
plot_acf(x, lags=30, alpha=0.05, ax=axes[1])
plot_pacf(x, lags=30, alpha=0.05, method="ywm", ax=axes[2])
plt.tight_layout()
plt.savefig(PIC_DIR / "ASS5-series_acf_pacf.png", dpi=300, bbox_inches="tight")
plt.close()


def aicc(result):
    """Small-sample AIC correction using the number of estimated parameters."""
    k = len(result.params)
    return result.aic + 2 * k * (k + 1) / (result.nobs - k - 1)


orders = {
    "AR(1)": (1, 0, 0),
    "AR(2)": (2, 0, 0),
    "AR(3)": (3, 0, 0),
    "MA(1)": (0, 0, 1),
    "MA(2)": (0, 0, 2),
    "MA(3)": (0, 0, 3),
}

print("Model AICc:")
for name, order in orders.items():
    result = ARIMA(x, order=order, trend="c").fit()
    print(name, aicc(result))
    print(dict(zip(result.param_names, result.params)))
