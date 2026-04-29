from pathlib import Path
import pandas as pd


DATA_PATH = Path("data/commodity_futures.csv")


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Could not find file: {path}")

    df = pd.read_csv(path)
    return df


def basic_overview(df: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("BASIC OVERVIEW")
    print("=" * 80)

    print(f"\nShape: {df.shape[0]} rows x {df.shape[1]} columns")

    print("\nColumns:")
    for col in df.columns:
        print(f"  - {col}")

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nLast 5 rows:")
    print(df.tail())


def data_types(df: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("DATA TYPES")
    print("=" * 80)

    print(df.dtypes)

    print("\nFull info:")
    df.info()


def missing_values(df: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("MISSING VALUES")
    print("=" * 80)

    missing = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_percent": df.isna().mean() * 100
    }).sort_values("missing_count", ascending=False)

    print(missing)


def duplicate_check(df: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("DUPLICATE ROWS")
    print("=" * 80)

    total_duplicates = df.duplicated().sum()
    print(f"Total duplicate rows: {total_duplicates}")

    if total_duplicates > 0:
        print("\nExample duplicate rows:")
        print(df[df.duplicated(keep=False)].head(10))


def numeric_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("NUMERIC SUMMARY")
    print("=" * 80)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    if not numeric_cols:
        print("No numeric columns detected.")
        return

    print(df[numeric_cols].describe().T)

    print("\nPotential negative values:")
    for col in numeric_cols:
        negative_count = (df[col] < 0).sum()
        if negative_count > 0:
            print(f"  {col}: {negative_count} negative values")

    print("\nPotential zero values:")
    for col in numeric_cols:
        zero_count = (df[col] == 0).sum()
        if zero_count > 0:
            print(f"  {col}: {zero_count} zero values")


def date_column_check(df: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("DATE COLUMN CHECK")
    print("=" * 80)

    possible_date_cols = [
        col for col in df.columns
        if "date" in col.lower() or "time" in col.lower()
    ]

    if not possible_date_cols:
        print("No obvious date/time column found.")
        return

    print(f"Possible date columns: {possible_date_cols}")

    for col in possible_date_cols:
        print(f"\nChecking date column: {col}")

        parsed = pd.to_datetime(df[col], errors="coerce")

        invalid_dates = parsed.isna().sum()
        print(f"Invalid/unparseable dates: {invalid_dates}")

        if invalid_dates < len(df):
            print(f"Date range: {parsed.min()} to {parsed.max()}")
            print(f"Unique dates: {parsed.nunique()}")

            duplicate_dates = parsed.duplicated().sum()
            print(f"Duplicate dates: {duplicate_dates}")

            sorted_dates = parsed.dropna().sort_values()
            date_diffs = sorted_dates.diff().value_counts().head(10)

            print("\nMost common gaps between dates:")
            print(date_diffs)


def categorical_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("CATEGORICAL / TEXT SUMMARY")
    print("=" * 80)

    object_cols = df.select_dtypes(include="object").columns.tolist()

    if not object_cols:
        print("No categorical/text columns detected.")
        return

    for col in object_cols:
        print(f"\nColumn: {col}")
        print(f"Unique values: {df[col].nunique()}")

        if df[col].nunique() <= 20:
            print(df[col].value_counts(dropna=False))
        else:
            print("Top 10 values:")
            print(df[col].value_counts(dropna=False).head(10))


def suspicious_values_check(df: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("SUSPICIOUS VALUES CHECK")
    print("=" * 80)

    for col in df.columns:
        if df[col].dtype == "object":
            blank_count = df[col].astype(str).str.strip().eq("").sum()
            if blank_count > 0:
                print(f"{col}: {blank_count} blank strings")

            weird_na_count = df[col].astype(str).str.lower().isin(
                ["na", "n/a", "null", "none", "-", "--", "?"]
            ).sum()

            if weird_na_count > 0:
                print(f"{col}: {weird_na_count} possible string-coded missing values")


def save_audit_outputs(df: pd.DataFrame) -> None:
    output_dir = Path("outputs/audit")
    output_dir.mkdir(parents=True, exist_ok=True)

    missing = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_percent": df.isna().mean() * 100
    }).sort_values("missing_count", ascending=False)

    missing.to_csv(output_dir / "missing_values_summary.csv")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        df[numeric_cols].describe().T.to_csv(output_dir / "numeric_summary.csv")

    print("\n" + "=" * 80)
    print("AUDIT OUTPUTS SAVED")
    print("=" * 80)
    print(f"Saved audit files to: {output_dir}")


def main():
    df = load_data(DATA_PATH)

    basic_overview(df)
    data_types(df)
    missing_values(df)
    duplicate_check(df)
    numeric_summary(df)
    date_column_check(df)
    categorical_summary(df)
    suspicious_values_check(df)
    save_audit_outputs(df)

    df = df.rename(columns={
    "WTI CRUDE": "oil_price"
})


if __name__ == "__main__":
    main()