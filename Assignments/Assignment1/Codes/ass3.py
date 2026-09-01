# Assignment 1 - Question 3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
PIC_DIR = BASE_DIR / "PIC"
PIC_DIR.mkdir(exist_ok=True)

data = pd.read_csv(BASE_DIR / "problem3.csv")

# Predict
pairs = [
    ("x1", "x2"),
    ("x1", "x3"),
    ("x1", "x4"),
    ("x2", "x3"),
    ("x2", "x4"),
    ("x3", "x4")
]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

for ax, (x, y) in zip(axes.flatten(), pairs):
    ax.scatter(data[x], data[y])
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"{x} ~ {y}")

plt.tight_layout()

plt.savefig(
    PIC_DIR / "ASS3-scatter_plots.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("Pearson correlation matrix:")
print(data.corr(method="pearson"))
print("Spearman correlation matrix:")
print(data.corr(method="spearman"))

for x, y in pairs:
    pearson_corr = data[x].corr(data[y], method="pearson")
    spearman_corr = data[x].corr(data[y], method="spearman")

    print(
        f"{x} ~ {y}: "
        f"Pearson = {pearson_corr:.4f}, "
        f"Spearman = {spearman_corr:.4f}, "
        f"Gap = {abs(pearson_corr - spearman_corr):.4f}"
    )
