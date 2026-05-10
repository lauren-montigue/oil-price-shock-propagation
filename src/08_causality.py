import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from statsmodels.tsa.stattools import grangercausalitytests

RETURNS_PATH = "data/commodity_returns_cleaned.csv"
REGIMES_PATH = "outputs/oil_shock_regimes.csv"

OUTPUT_DIR = "outputs/causality_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

selected_cols = [
    "WTI CRUDE",
    "NATURAL GAS",
    "GOLD",
    "CORN",
    "WHEAT",
    "SOYBEANS",
    "COPPER",
    "SILVER",
]

shock_var = "WTI CRUDE"

# Keep small because shock sample sizes are limited
max_lag = 1

significance_level = 0.05


# Load data
returns = pd.read_csv(RETURNS_PATH)
regimes = pd.read_csv(REGIMES_PATH)

returns.columns = returns.columns.str.strip()
regimes.columns = regimes.columns.str.strip()

returns["Date"] = pd.to_datetime(returns["Date"])
regimes["Date"] = pd.to_datetime(regimes["Date"])

returns = returns.sort_values("Date")
regimes = regimes.sort_values("Date")


merged_daily = returns.merge(
    regimes[["Date", "ShockRegime"]],
    on="Date",
    how="inner"
)

merged_daily[selected_cols] = merged_daily[selected_cols].apply(
    pd.to_numeric,
    errors="coerce"
)

merged_daily["ShockRegime"] = pd.to_numeric(
    merged_daily["ShockRegime"],
    errors="coerce"
)

merged_daily = merged_daily.dropna(
    subset=selected_cols + ["ShockRegime"]
)

merged_daily = merged_daily.set_index("Date").sort_index()


# Convert daily -> monthly
# Daily log returns aggregate additively:
# monthly log return = sum of daily log returns
#
# A month is classified as a shock month if ANY day
# in that month was labeled as a shock regime.

monthly_returns = (
    merged_daily[selected_cols]
    .resample("ME")
    .sum()
)

monthly_regime = (
    merged_daily["ShockRegime"]
    .resample("ME")
    .max()
)

monthly = monthly_returns.copy()

monthly["ShockRegime"] = monthly_regime

monthly = monthly.dropna()

monthly.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "monthly_returns_with_regimes.csv"
    )
)

# Split by regime
normal_months = monthly[
    monthly["ShockRegime"] == 0
]

shock_months = monthly[
    monthly["ShockRegime"] == 1
]

print("\nMonthly sample sizes:")
print(f"Normal months: {len(normal_months)}")
print(f"Shock months:  {len(shock_months)}")


# Granger helper
def run_pairwise_granger(
    data,
    target_var,
    variables,
    max_lag=1
):
    """
    Tests whether target_var Granger-causes each variable.

    In statsmodels:
        grangercausalitytests(data[[Y, X]])
    tests:
        X -> Y

    Therefore:
        data[[predictor, target_var]]
    tests:
        target_var -> predictor
    """

    results = []

    for predictor in variables:

        # Skip self-causality
        if predictor == target_var:
            continue

        try:

            # Test:
            # WTI CRUDE -> predictor
            test_result = grangercausalitytests(
                data[[predictor, target_var]],
                maxlag=max_lag,
                verbose=False
            )

            p_values = []

            for lag in range(1, max_lag + 1):

                p_val = (
                    test_result[lag][0]
                    ["ssr_chi2test"][1]
                )

                p_values.append(p_val)

            min_p = np.min(p_values)

            best_lag = np.argmin(p_values) + 1

            results.append({
                "Predictor": predictor,
                "MinPValue": min_p,
                "BestLag": best_lag,
                "Significant": (
                    min_p < significance_level
                )
            })

        except Exception as e:

            print(f"Failed for {predictor}: {e}")

    results_df = pd.DataFrame(results)

    return results_df.sort_values("MinPValue")


# Run Granger tests
normal_results = run_pairwise_granger(
    normal_months[selected_cols],
    target_var=shock_var,
    variables=selected_cols,
    max_lag=max_lag
)

shock_results = run_pairwise_granger(
    shock_months[selected_cols],
    target_var=shock_var,
    variables=selected_cols,
    max_lag=max_lag
)

# Save tables
normal_results.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "normal_months_granger.csv"
    ),
    index=False
)

shock_results.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "shock_months_granger.csv"
    ),
    index=False
)

print("\nNormal months results:")
print(normal_results)

print("\nShock months results:")
print(shock_results)


# Merge for comparison
comparison = pd.merge(
    normal_results,
    shock_results,
    on="Predictor",
    suffixes=("_Normal", "_Shock")
)


# PLOT 1:
# Stronger causality = taller bars
# Using -log10(p-value)

plot_df = comparison.copy()

# Transform p-values
plot_df["Strength_Normal"] = -np.log10(
    plot_df["MinPValue_Normal"]
)

plot_df["Strength_Shock"] = -np.log10(
    plot_df["MinPValue_Shock"]
)

x = np.arange(len(plot_df))

width = 0.35

plt.figure(figsize=(11, 5))

plt.bar(
    x - width/2,
    plot_df["Strength_Normal"],
    width,
    label="Normal Months"
)

plt.bar(
    x + width/2,
    plot_df["Strength_Shock"],
    width,
    label="Shock Months"
)


# Significance threshold
# p = 0.05

sig_threshold = -np.log10(significance_level)

plt.axhline(
    sig_threshold,
    linestyle="--",
    linewidth=1.5,
    color="red",
    label="p = 0.05"
)

plt.xticks(
    x,
    plot_df["Predictor"],
    rotation=45
)

plt.ylabel("-log10(p-value)")

plt.title(
    "WTI Crude Granger Causality Strength\n"
    "(Higher = Stronger Evidence)"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "granger_strength_comparison.png"
    ),
    dpi=300
)

plt.show()



# PLOT 2:
# Heatmap of causality strength

heatmap_df = plot_df[
    [
        "Predictor",
        "Strength_Normal",
        "Strength_Shock"
    ]
].copy()

heatmap_df = heatmap_df.set_index("Predictor")

plt.figure(figsize=(6, 4))

im = plt.imshow(
    heatmap_df,
    aspect="auto"
)

plt.colorbar(
    label="-log10(p-value)"
)

plt.xticks(
    [0, 1],
    ["Normal", "Shock"]
)

plt.yticks(
    range(len(heatmap_df.index)),
    heatmap_df.index
)

plt.title(
    "WTI Crude Causality Heatmap"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "granger_heatmap.png"
    ),
    dpi=300
)

plt.show()



# PLOT 3:
# Significant relationships only

plot_df["Significant_Normal"] = (
    plot_df["MinPValue_Normal"]
    < significance_level
).astype(int)

plot_df["Significant_Shock"] = (
    plot_df["MinPValue_Shock"]
    < significance_level
).astype(int)

plt.figure(figsize=(10, 4))

plt.plot(
    plot_df["Predictor"],
    plot_df["Significant_Normal"],
    marker="o",
    linewidth=2,
    label="Normal Months"
)

plt.plot(
    plot_df["Predictor"],
    plot_df["Significant_Shock"],
    marker="o",
    linewidth=2,
    label="Shock Months"
)

plt.yticks(
    [0, 1],
    ["No", "Yes"]
)

plt.ylabel("Significant?")

plt.title(
    "Significant WTI Granger Relationships"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "granger_significance.png"
    ),
    dpi=300
)

plt.show()