# src/04_var_analysis.py

import os
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR


# -----------------------------
# Paths
# -----------------------------

INPUT_PATH = "data/commodity_returns_cleaned.csv"
OUTPUT_DIR = "outputs/transmission"
PLOTS_DIR = "outputs/plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


# -----------------------------
# Load cleaned log returns
# -----------------------------

df = pd.read_csv(INPUT_PATH)

print(df.columns.tolist())

df["Date"] = pd.to_datetime(df["Date"])
df = df.set_index("Date").sort_index()

# Use a manageable subset first.
# WTI CRUDE is the main oil shock variable.
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

var_data = df[selected_cols].dropna()


# -----------------------------
# Fit VAR model
# -----------------------------

model = VAR(var_data)

# maxlags=12 is appropriate if these are monthly returns.
# If these are daily returns, use a smaller value like 5 or 10.
results = model.fit(maxlags=12, ic="aic")

print(results.summary())
print(f"\nSelected lag order: {results.k_ar}")


# -----------------------------
# Save model summary
# -----------------------------

summary_path = os.path.join(OUTPUT_DIR, "var_summary.txt")

with open(summary_path, "w", encoding="utf-8") as f:
    f.write(str(results.summary()))
    f.write(f"\n\nSelected lag order: {results.k_ar}\n")


# -----------------------------
# Impulse Response Functions
# -----------------------------

irf_horizon = 12
irf = results.irf(irf_horizon)

fig = irf.plot(orth=False)
fig.set_size_inches(14, 10)
plt.tight_layout()

irf_plot_path = os.path.join(PLOTS_DIR, "var_impulse_responses.png")
plt.savefig(irf_plot_path, dpi=300)
plt.close()


# -----------------------------
# Forecast Error Variance Decomposition
# -----------------------------

fevd = results.fevd(irf_horizon)

fig = fevd.plot()
fig.set_size_inches(14, 10)
plt.tight_layout()

fevd_plot_path = os.path.join(PLOTS_DIR, "var_fevd.png")
plt.savefig(fevd_plot_path, dpi=300)
plt.close()


# -----------------------------
# Save FEVD table
# -----------------------------

fevd_tables = []

for i, response_variable in enumerate(var_data.columns):
    temp = pd.DataFrame(
        fevd.decomp[i],
        columns=var_data.columns,
    )

    temp.insert(0, "horizon", range(1, irf_horizon + 1))
    temp.insert(0, "response_variable", response_variable)

    fevd_tables.append(temp)

fevd_df = pd.concat(fevd_tables, ignore_index=True)

fevd_path = os.path.join(OUTPUT_DIR, "fevd_results.csv")
fevd_df.to_csv(fevd_path, index=False)


# -----------------------------
# Save cleaned VAR input
# -----------------------------

var_input_path = os.path.join(OUTPUT_DIR, "var_input_data.csv")
var_data.to_csv(var_input_path)


print("\nVAR analysis complete.")
print(f"Summary saved to: {summary_path}")
print(f"VAR input data saved to: {var_input_path}")
print(f"IRF plot saved to: {irf_plot_path}")
print(f"FEVD plot saved to: {fevd_plot_path}")
print(f"FEVD results saved to: {fevd_path}")