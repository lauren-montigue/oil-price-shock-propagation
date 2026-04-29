from pathlib import Path
import numpy as np
import pandas as pd


RAW_DATA_PATH = Path("data/commodity_futures.csv")
CLEANED_PRICES_PATH = Path("data/commodity_prices_cleaned.csv")
CLEANED_RETURNS_PATH = Path("data/commodity_returns_cleaned.csv")

DROP_COLUMNS = ["GASOLINE"]  # High missingness column


def clean_commodity_data(raw_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(raw_path)

    # Clean dates
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.sort_values("Date").set_index("Date")

    # Drop columns with high missingness
    df = df.drop(columns=[col for col in DROP_COLUMNS if col in df.columns])

    # WTI had one real negative value, but it breaks log returns.
    # Treat it as missing and forward fill.
    if "WTI CRUDE" in df.columns:
        df.loc[df["WTI CRUDE"] <= 0, "WTI CRUDE"] = np.nan

    # Replace any remaining nonpositive prices before log returns
    # because log(price) is undefined for price <= 0.
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].where(df[numeric_cols] > 0, np.nan)

    # Light imputation for sparse missing values
    df = df.ffill().bfill()

    # Save cleaned price levels
    cleaned_prices = df.copy()

    # Compute log returns
    cleaned_returns = np.log(cleaned_prices).diff().dropna()

    return cleaned_prices, cleaned_returns


def main() -> None:
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Could not find raw data file: {RAW_DATA_PATH}")

    CLEANED_PRICES_PATH.parent.mkdir(parents=True, exist_ok=True)

    cleaned_prices, cleaned_returns = clean_commodity_data(RAW_DATA_PATH)

    cleaned_prices.to_csv(CLEANED_PRICES_PATH)
    cleaned_returns.to_csv(CLEANED_RETURNS_PATH)

    print("Cleaning complete.")
    print(f"Saved cleaned prices to: {CLEANED_PRICES_PATH}")
    print(f"Saved cleaned returns to: {CLEANED_RETURNS_PATH}")
    print(f"Cleaned price shape: {cleaned_prices.shape}")
    print(f"Cleaned returns shape: {cleaned_returns.shape}")


if __name__ == "__main__":
    main()