from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

DATASET = "debashish311601/commodity-prices"
RAW_DIR = Path("data")

def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    api.dataset_download_files(
        DATASET,
        path=RAW_DIR,
        unzip=True
    )

    print(f"Downloaded and unzipped dataset to: {RAW_DIR}")

if __name__ == "__main__":
    main()