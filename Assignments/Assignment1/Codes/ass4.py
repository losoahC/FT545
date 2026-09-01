# Assignment 1 - Question 4

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import norm

BASE_DIR = Path(__file__).resolve().parents[1]
PIC_DIR = BASE_DIR / "PIC"
PIC_DIR.mkdir(exist_ok=True)

data = pd.read_csv(BASE_DIR / "problem4.csv")

cov = data[["x1", "x2"]].cov()
print(cov)
mean = data[["x1", "x2"]].mean()

var_x1 = cov.loc["x1", "x1"]
var_x2 = cov.loc["x2", "x2"]
cov_x1_x2 = cov.loc["x1", "x2"]

conditional_var = (
    var_x2
    - cov_x1_x2**2 / var_x1
)

print("Conditional variance:", conditional_var)

factor = conditional_var / var_x2
print("Remaining variance factor:", factor)

# For a bivariate Normal, E[x2 | x1] is a straight line with this slope.
slope = cov_x1_x2 / var_x1
intercept = mean["x2"] - slope * mean["x1"]
conditional_sd = np.sqrt(conditional_var)
band_half_width = norm.ppf(0.975) * conditional_sd

print("Conditional mean intercept:", intercept)
print("Conditional mean slope:", slope)
print("Conditional standard deviation:", conditional_sd)
print("95% band half width:", band_half_width)

data["conditional_mean"] = intercept + slope * data["x1"]
data["lower"] = data["conditional_mean"] - band_half_width
data["upper"] = data["conditional_mean"] + band_half_width
data["inside_band"] = data["x2"].between(data["lower"], data["upper"])

print("Overall coverage:", data["inside_band"].mean())

z_abs = (data["x1"] - mean["x1"]).abs() / data["x1"].std(ddof=1)
buckets = {
    "inside 1 sd": z_abs <= 1,
    "between 1 and 2 sd": (z_abs > 1) & (z_abs <= 2),
    "beyond 2 sd": z_abs > 2,
}

for name, mask in buckets.items():
    print(name, "n:", mask.sum(), "coverage:", data.loc[mask, "inside_band"].mean())

x_grid = np.linspace(data["x1"].min(), data["x1"].max(), 300)
mean_grid = intercept + slope * x_grid

plt.figure(figsize=(8, 5))
plt.scatter(data["x1"], data["x2"], s=16, alpha=0.45, label="Data")
plt.plot(x_grid, mean_grid, color="crimson", linewidth=2, label="Conditional mean")
plt.fill_between(
    x_grid,
    mean_grid - band_half_width,
    mean_grid + band_half_width,
    color="crimson",
    alpha=0.16,
    label="95% conditional band",
)
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Conditional expectation of x2 given x1")
plt.legend()
plt.tight_layout()
plt.savefig(PIC_DIR / "ASS4-conditional_band.png", dpi=300, bbox_inches="tight")
plt.close()
