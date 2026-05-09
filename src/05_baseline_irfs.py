# src/05_baseline_irfs.py

import os
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR


# -----------------------------
# Paths
# -----------------------------

INPUT_PATH = "data/commodity_returns_cleaned.csv"
OUTPUT_DIR = "outputs/irfs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------
# Load data
# -----------------------------

df = pd.read_csv(INPUT_PATH)
df.columns = df.columns.str.strip()

df["Date"] = pd.to_datetime(df["Date"])
df = df.set_index("Date").sort_index()


# -----------------------------
# Select variables
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

var_data = df[selected_cols].apply(pd.to_numeric, errors="coerce").dropna()


# -----------------------------
# Fit baseline VAR
# -----------------------------

model = VAR(var_data)

# Your earlier model selected 3 lags.
# Keeping it fixed makes results easier to explain.
results = model.fit(3)

print(results.summary())


# -----------------------------
# Generate IRFs
# -----------------------------

horizon = 20
irf = results.irf(horizon)

shock_var = "WTI CRUDE"


# -----------------------------
# Save full IRF plot
# -----------------------------

fig = irf.plot(orth=False)
fig.set_size_inches(14, 10)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "baseline_irfs_all.png"), dpi=300)
plt.close()


# -----------------------------
# Save individual WTI shock response plots
# -----------------------------

for response_var in selected_cols:
    fig = irf.plot(
        impulse=shock_var,
        response=response_var,
        orth=False,
    )

    fig.set_size_inches(8, 5)
    plt.title(f"Response of {response_var} to a Shock in {shock_var}")
    plt.tight_layout()

    filename = f"irf_{shock_var}_to_{response_var}.png"
    filename = filename.replace(" ", "_").replace("/", "_")

    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    plt.close()


# -----------------------------
# Save IRF values to CSV
# -----------------------------

irf_values = []

shock_index = selected_cols.index(shock_var)

for h in range(horizon + 1):
    for response_var in selected_cols:
        response_index = selected_cols.index(response_var)

        irf_values.append({
            "horizon": h,
            "impulse": shock_var,
            "response": response_var,
            "irf_value": irf.irfs[h, response_index, shock_index],
        })

irf_df = pd.DataFrame(irf_values)
irf_df.to_csv(os.path.join(OUTPUT_DIR, "baseline_wti_irf_values.csv"), index=False)


print("\nBaseline IRF analysis complete.")
print(f"Outputs saved in: {OUTPUT_DIR}")