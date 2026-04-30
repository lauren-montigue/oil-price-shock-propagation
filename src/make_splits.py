from pathlib import Path
import pandas as pd
import os

OUTPUT_DIR = Path("data/splits")

def train_val_test_split(df, train_size=0.70, val_size=0.15):
    """
    Chronological 70/15/15 split for time series data.
    """

    n = len(df)

    train_end = int(n * train_size)
    val_end = int(n * (train_size + val_size))

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    return train, val, test


def main():
    # Load data (relative path)
    df = pd.read_csv("data/commodity_returns_cleaned.csv")

    # Ensure chronological order if needed
    # df = df.sort_values("date")

    train, val, test = train_val_test_split(df)

    # Create output folder 
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save files
    train.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
    val.to_csv(os.path.join(OUTPUT_DIR, "val.csv"), index=False)
    test.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False)

    print("Saved splits to:", OUTPUT_DIR)
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

if __name__ == "__main__":
    main()