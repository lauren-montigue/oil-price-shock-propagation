# Oil Shock Transmission Visualization Scripts
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================
# LOAD RESULTS
# ============================================================

results = pd.read_csv(
    "outputs/transmission/arma_transmission_results.csv"
)

os.makedirs("outputs/plots", exist_ok=True)


# ============================================================
# 1. CONTEMPORANEOUS OIL SHOCK COEFFICIENTS
# ============================================================

plot_df = results[[
    "commodity",
    "oil_shock_lag_0_coef"
]].copy()

plot_df = plot_df.sort_values(
    "oil_shock_lag_0_coef",
    ascending=False
)

plt.figure(figsize=(12, 6))

plt.bar(
    plot_df["commodity"],
    plot_df["oil_shock_lag_0_coef"]
)

plt.xticks(rotation=90)
plt.ylabel("Coefficient")
plt.title("Contemporaneous Oil Shock Transmission")
plt.tight_layout()

plt.savefig(
    "outputs/plots/lag0_coefficients.png"
)

plt.show()


# ============================================================
# 2. SIGNIFICANT COMMODITIES ONLY
# p-value < 0.05
# ============================================================

sig_df = results[
    results["oil_shock_lag_0_pvalue"] < 0.05
][[
    "commodity",
    "oil_shock_lag_0_coef"
]]

sig_df = sig_df.sort_values(
    "oil_shock_lag_0_coef",
    ascending=False
)

plt.figure(figsize=(12, 6))

plt.bar(
    sig_df["commodity"],
    sig_df["oil_shock_lag_0_coef"]
)

plt.xticks(rotation=90)
plt.ylabel("Coefficient")
plt.title("Significant Oil Shock Transmission Effects")
plt.tight_layout()

plt.savefig(
    "outputs/plots/significant_lag0_coefficients.png"
)

plt.show()


# ============================================================
# 3. HEATMAP OF ALL OIL SHOCK LAGS
# ============================================================

heatmap_df = results[[
    "commodity",
    "oil_shock_lag_0_coef",
    "oil_shock_lag_1_coef",
    "oil_shock_lag_2_coef",
    "oil_shock_lag_3_coef"
]].copy()

heatmap_df = heatmap_df.set_index("commodity")

plt.figure(figsize=(10, 8))

plt.imshow(
    heatmap_df,
    aspect="auto"
)

plt.colorbar(label="Coefficient")

plt.xticks(
    range(len(heatmap_df.columns)),
    heatmap_df.columns,
    rotation=45
)

plt.yticks(
    range(len(heatmap_df.index)),
    heatmap_df.index
)

plt.title("Oil Shock Transmission Heatmap")
plt.tight_layout()

plt.savefig(
    "outputs/plots/oil_shock_heatmap.png"
)

plt.show()


# ============================================================
# 4. P-VALUE DISTRIBUTION
# ============================================================

pvals = results[
    "oil_shock_lag_0_pvalue"
].dropna()

plt.figure(figsize=(8, 5))

plt.hist(pvals, bins=20)

plt.axvline(
    0.05,
    linestyle="--"
)

plt.xlabel("p-value")
plt.ylabel("Count")
plt.title("Distribution of Oil Shock p-values")
plt.tight_layout()

plt.savefig(
    "outputs/plots/pvalue_distribution.png"
)

plt.show()


# ============================================================
# 5. TOP 10 MOST SENSITIVE COMMODITIES
# ============================================================

plot_df = results[[
    "commodity",
    "oil_shock_lag_0_coef"
]].copy()

plot_df["abs_coef"] = np.abs(
    plot_df["oil_shock_lag_0_coef"]
)

plot_df = plot_df.sort_values(
    "abs_coef",
    ascending=False
).head(10)

plt.figure(figsize=(10, 6))

plt.bar(
    plot_df["commodity"],
    plot_df["oil_shock_lag_0_coef"]
)

plt.xticks(rotation=45)
plt.ylabel("Coefficient")
plt.title("Top 10 Most Oil-Sensitive Commodities")
plt.tight_layout()

plt.savefig(
    "outputs/plots/top10_sensitive_commodities.png"
)

plt.show()


# ============================================================
# 6. NUMBER OF SIGNIFICANT COMMODITIES
# ============================================================

significant_count = (
    results["oil_shock_lag_0_pvalue"] < 0.05
).sum()

nonsignificant_count = (
    results["oil_shock_lag_0_pvalue"] >= 0.05
).sum()

plt.figure(figsize=(6, 6))

plt.pie(
    [significant_count, nonsignificant_count],
    labels=["Significant", "Not Significant"],
    autopct="%1.1f%%"
)

plt.title("Share of Commodities with Significant Oil Transmission")

plt.savefig(
    "outputs/plots/significance_share.png"
)

plt.show()


print("All plots saved to outputs/plots/")

# Recommended Plots for Your Report

## Most important

# 1. Contemporaneous Oil Shock Transmission

#    * Shows which commodities react most strongly.

# 2. Significant Transmission Effects

#    * Focuses only on statistically significant relationships.

# 3. Heatmap of Lagged Effects

#    * Shows whether transmission is immediate or delayed.

# ## Nice supporting visuals

# 4. P-value Distribution

#    * Demonstrates how widespread significance is.

# 5. Top 10 Most Sensitive Commodities

#    * Easy summary figure.

# 6. Significance Share Pie Chart

#    * High-level overview of findings.

# # Expected Interpretation

# You will likely observe:

# * Strong immediate transmission in energy-linked commodities.
# * Moderate transmission in metals.
# * Weaker transmission in agricultural/livestock commodities.
# * Most significant effects concentrated in lag 0.
# * Limited delayed propagation at
