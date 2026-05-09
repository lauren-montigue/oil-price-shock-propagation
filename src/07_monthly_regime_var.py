# src/07_monthly_regime_var.py

import os
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR


# -----------------------------
# Paths
# -----------------------------

RETURNS_PATH = "data/commodity_returns_cleaned.csv"
REGIMES_PATH = "outputs/oil_shock_regimes.csv"  # change if needed
OUTPUT_DIR = "outputs/monthly_regime_var"

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
horizon = 12
lags = 1


# -----------------------------
# Load daily data
# -----------------------------

returns = pd.read_csv(RETURNS_PATH)
regimes = pd.read_csv(REGIMES_PATH)

returns.columns = returns.columns.str.strip()
regimes.columns = regimes.columns.str.strip()

returns["Date"] = pd.to_datetime(returns["Date"])
regimes["Date"] = pd.to_datetime(regimes["Date"])

returns = returns.sort_values("Date")
regimes = regimes.sort_values("Date")


# -----------------------------
# Merge daily returns with regimes
# -----------------------------

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

merged_daily = merged_daily.dropna(subset=selected_cols + ["ShockRegime"])
merged_daily = merged_daily.set_index("Date").sort_index()


# -----------------------------
# Convert daily data to monthly data
# -----------------------------
# Since the daily commodity variables are log returns,
# monthly log returns are the sum of daily log returns.
#
# For ShockRegime:
# A month is labeled as a shock month if any day in that month
# was classified as a shock regime.

monthly_returns = merged_daily[selected_cols].resample("ME").sum()

monthly_regime = merged_daily["ShockRegime"].resample("ME").max()

monthly = monthly_returns.copy()
monthly["ShockRegime"] = monthly_regime

monthly = monthly.dropna()

monthly.to_csv(os.path.join(OUTPUT_DIR, "monthly_returns_with_regimes.csv"))


# -----------------------------
# Split monthly data by regime
# -----------------------------

normal_months = monthly[monthly["ShockRegime"] == 0]
shock_months = monthly[monthly["ShockRegime"] == 1]

print("\nMonthly sample sizes:")
print(f"Normal months: {len(normal_months)}")
print(f"Shock months:  {len(shock_months)}")


# -----------------------------
# Helper function
# -----------------------------

def fit_monthly_var_and_irfs(data, label):
    print(f"\nRunning monthly {label} VAR")
    print(f"Observations: {len(data)}")

    if len(data) <= lags + 5:
        raise ValueError(
            f"Not enough observations to estimate {label} VAR. "
            f"Try reducing variables or changing regime definition."
        )

    model = VAR(data[selected_cols])
    results = model.fit(lags)

    print(results.summary())

    with open(os.path.join(OUTPUT_DIR, f"{label}_monthly_var_summary.txt"), "w") as f:
        f.write(str(results.summary()))

    irf = results.irf(horizon)

    irf_values = []
    shock_index = selected_cols.index(shock_var)

    for h in range(horizon + 1):
        for response_var in selected_cols:
            response_index = selected_cols.index(response_var)

            irf_values.append({
                "regime": label,
                "horizon_months": h,
                "impulse": shock_var,
                "response": response_var,
                "irf_value": irf.orth_irfs[h, response_index, shock_index],
            })

    irf_df = pd.DataFrame(irf_values)
    irf_df.to_csv(
        os.path.join(OUTPUT_DIR, f"{label}_monthly_wti_irf_values.csv"),
        index=False
    )

    return irf_df


# -----------------------------
# Fit monthly regime VARs
# -----------------------------

normal_irfs = fit_monthly_var_and_irfs(normal_months, "normal")
shock_irfs = fit_monthly_var_and_irfs(shock_months, "shock")

combined_irfs = pd.concat([normal_irfs, shock_irfs], ignore_index=True)
combined_irfs.to_csv(
    os.path.join(OUTPUT_DIR, "normal_vs_shock_monthly_wti_irfs.csv"),
    index=False
)


# -----------------------------
# Comparison plots
# -----------------------------

for response_var in selected_cols:
    subset = combined_irfs[combined_irfs["response"] == response_var]

    plt.figure(figsize=(8, 5))

    for regime in ["normal", "shock"]:
        temp = subset[subset["regime"] == regime]
        plt.plot(
            temp["horizon_months"],
            temp["irf_value"],
            marker="o",
            label=regime
        )

    plt.axhline(0, color="black", linewidth=1)
    plt.title(f"Monthly WTI Shock Response: {response_var}")
    plt.xlabel("Months after WTI shock")
    plt.ylabel("Orthogonalized IRF")
    plt.legend()
    plt.tight_layout()

    filename = f"monthly_comparison_WTI_to_{response_var}.png"
    filename = filename.replace(" ", "_").replace("/", "_")

    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    plt.close()


print("\nMonthly regime VAR analysis complete.")
print(f"Outputs saved in: {OUTPUT_DIR}")