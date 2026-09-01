# Assignment 1 - Question 1

import pandas as pd
import numpy as np
from scipy.stats import norm
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
PIC_DIR = BASE_DIR / "PIC"
PIC_DIR.mkdir(exist_ok=True)

data = pd.read_csv(BASE_DIR / "problem1.csv")

# a. Predict
cm1 = data.mean().iloc[0]
cm2 = data.var().iloc[0]
cm3 = data.skew().iloc[0]
cm4 = data.kurt().iloc[0]

print("Mean:", cm1)
print("Variance:", cm2)
print("Skewness:", cm3)
print("Kurtosis:", cm4)

# Kurt is excessive for the normal distribution for the function kurt()

# b. Fit and reconcile
# Fitted normal
fitted_mean = cm1
fitted_std = np.sqrt(cm2)
model = norm(loc=fitted_mean, scale=fitted_std)

q01 = model.ppf(0.01)
actual = (data < q01).sum().iloc[0] 
expected = len(data) * 0.01
print("Quantile at 0.01 - Actual:", actual, "Expected:", expected)

# c. Plot the distribution and the fitted normal distribution
import matplotlib.pyplot as plt

plt.hist(data.iloc[:, 0], bins=50, density=True, alpha=0.5, label='Spotty Distribution')
x = np.linspace(data.iloc[:, 0].min(), data.iloc[:, 0].max(), 1000)
plt.plot(x, model.pdf(x), 'r-', label='Fitted Normal')
plt.legend()
plt.savefig(PIC_DIR / 'ASS1-spotty_vs_fitted_normal.png')
plt.close()
