# src/06_regime_irfs.py

import os
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR


# -----------------------------
# Paths
# -----------------------------

RETURNS_PATH = "data/commodity_returns_cleaned.csv"
REGIMES_PATH = "outputs/oil_shock_regimes.csv"  # change if yours is elsewhere
OUTPUT_DIR = "outputs/regime_irfs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------
# Settings
# -----------------------------

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
horizon = 20
lags = 3


# -----------------------------
# Load and merge data
# -----------------------------

returns = pd.read_csv(RETURNS_PATH)
regimes = pd.read_csv(REGIMES_PATH)

returns.columns = returns.columns.str.strip()
regimes.columns = regimes.columns.str.strip()

returns["Date"] = pd.to_datetime(returns["Date"])
regimes["Date"] = pd.to_datetime(regimes["Date"])

merged = returns.merge(
    regimes[["Date", "ShockRegime"]],
    on="Date",
    how="inner"
)

merged = merged.sort_values("Date").set_index("Date")

merged[selected_cols] = merged[selected_cols].apply(pd.to_numeric, errors="coerce")
merged["ShockRegime"] = pd.to_numeric(merged["ShockRegime"], errors="coerce")

merged = merged[selected_cols + ["ShockRegime"]].dropna()

merged.to_csv(os.path.join(OUTPUT_DIR, "merged_returns_with_regimes.csv"))


# -----------------------------
# Helper function
# -----------------------------

def fit_var_and_save_irfs(data, label):
    print(f"\nRunning {label} VAR")
    print(f"Observations: {len(data)}")

    if len(data) < 100:
        print(f"WARNING: {label} sample is small. Results may be noisy.")

    model = VAR(data[selected_cols])
    results = model.fit(lags)

    print(results.summary())

    irf = results.irf(horizon)

    # Save individual WTI response plots
    for response_var in selected_cols:
        fig = irf.plot(
            impulse=shock_var,
            response=response_var,
            orth=True,
        )

        fig.set_size_inches(8, 5)
        plt.title(f"{label}: Response of {response_var} to WTI Shock")
        plt.tight_layout()

        filename = f"{label}_WTI_to_{response_var}.png"
        filename = filename.replace(" ", "_").replace("/", "_")

        plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
        plt.close()

    # Save IRF values
    irf_values = []
    shock_index = selected_cols.index(shock_var)

    for h in range(horizon + 1):
        for response_var in selected_cols:
            response_index = selected_cols.index(response_var)

            irf_values.append({
                "regime": label,
                "horizon": h,
                "impulse": shock_var,
                "response": response_var,
                "irf_value": irf.orth_irfs[h, response_index, shock_index],
            })

    irf_df = pd.DataFrame(irf_values)
    irf_df.to_csv(
        os.path.join(OUTPUT_DIR, f"{label}_wti_irf_values.csv"),
        index=False
    )

    return irf_df


# -----------------------------
# Split data by regime
# -----------------------------

normal_data = merged[merged["ShockRegime"] == 0]
shock_data = merged[merged["ShockRegime"] == 1]

print("\nSample sizes:")
print(f"Normal periods: {len(normal_data)}")
print(f"Shock periods:  {len(shock_data)}")


# -----------------------------
# Fit regime-specific VARs
# -----------------------------

normal_irfs = fit_var_and_save_irfs(normal_data, "normal")
shock_irfs = fit_var_and_save_irfs(shock_data, "shock")


# -----------------------------
# Save combined IRF values
# -----------------------------

combined_irfs = pd.concat([normal_irfs, shock_irfs], ignore_index=True)
combined_irfs.to_csv(
    os.path.join(OUTPUT_DIR, "normal_vs_shock_wti_irf_values.csv"),
    index=False
)


# -----------------------------
# Comparison plots: normal vs shock
# -----------------------------

for response_var in selected_cols:
    subset = combined_irfs[combined_irfs["response"] == response_var]

    plt.figure(figsize=(8, 5))

    for regime in ["normal", "shock"]:
        temp = subset[subset["regime"] == regime]
        plt.plot(temp["horizon"], temp["irf_value"], label=regime)

    plt.axhline(0, color="black", linewidth=1)
    plt.title(f"WTI Shock Response: {response_var}")
    plt.xlabel("Horizon")
    plt.ylabel("Orthogonalized IRF")
    plt.legend()
    plt.tight_layout()

    filename = f"comparison_WTI_to_{response_var}.png"
    filename = filename.replace(" ", "_").replace("/", "_")

    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    plt.close()


print("\nRegime IRF analysis complete.")
print(f"Outputs saved in: {OUTPUT_DIR}")