from pathlib import Path
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
import os

INPUT_PATH = Path("data/commodity_returns_cleaned.csv")
OUTPUT_DIR = Path("outputs/stationarity")
# OUTPUT_DATA_PATH = Path("data/processed/stationary_data.csv")

def adf_test(series):
    result = adfuller(series.dropna(), autolag="AIC")
    return result[1]  # p-value

def kpss_test(series):
    try:
        stat, p, _, _ = kpss(series.dropna(), regression="c", nlags="auto")
    except:
        p = np.nan
    return p


def run_stationarity_pipeline(df):
    results = []
    stationary_df = df.copy()

    for col in df.columns: # assess stationarity for each commodity
        series = df[col].dropna()

        adf_p = adf_test(series)
        kpss_p = kpss_test(series)

        is_stationary = (adf_p < 0.05) and (kpss_p > 0.05 if not np.isnan(kpss_p) else True)

        results.append({
            "series": col,
            "adf_pvalue": adf_p,
            "kpss_pvalue": kpss_p,
            "stationary": is_stationary
        })

        # If NOT stationary → apply rolling standardization
        if not is_stationary:
            print(f"{col} not stationary → applying rolling standardization")

            transformed = (
                series - series.rolling(20).mean()
            ) / series.rolling(20).std()

            stationary_df[col] = transformed

    return pd.DataFrame(results), stationary_df


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # os.makedirs(os.path.dirname(OUTPUT_DATA_PATH), exist_ok=True)

    # load data
    df = pd.read_csv(INPUT_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()

    # run pipeline
    results_df, stationary_df = run_stationarity_pipeline(df)

    # save outputs
    results_df.to_csv(os.path.join(OUTPUT_DIR, "stationarity_test_results.csv"), index=False)
    # stationary_df.dropna().to_csv(OUTPUT_DATA_PATH)

    print("\nDone.")
    print(f"Results saved to: {OUTPUT_DIR}")
    # print(f"Stationary dataset saved to: {OUTPUT_DATA_PATH}")

if __name__ == "__main__":
    main()