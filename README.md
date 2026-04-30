# Oil Price Shock Propagation Across Commodity Markets

## Overview

This project investigates how shocks in oil prices propagate across other commodity markets using time series methods (ARMA, mSSA, and change point analysis).

The goal is to understand whether and how movements in oil prices influence other commodities such as metals, agriculture, and energy products.

---

## Repository Structure

```
oil-price-shock-propagation/
├── data/
│   ├── splits                              # Train/val/test split of commodity_returns_cleaned.csv
│      ├── train.csv                       
│      ├── test.csv
│      ├── val.csv
│   ├── commodity_futures.csv               # Raw dataset (Kaggle)
│   ├── commodity_prices_cleaned.csv        # Cleaned price levels
│   └── commodity_returns_cleaned.csv       # Cleaned log returns (model-ready)
│
├── src/
│   ├── download_data.py        # Download dataset from Kaggle (optional)
│   ├── data_audit.py           # Data quality checks / exploratory audit
│   └── clean_data.py           # Cleaning + preprocessing pipeline
|   └── make_splits.py          # Data Splitting (70 / 15 / 15)
|   └── stationarity_tests.py   # Stationarity Checks (ADF and KPSS tests)
│
├── notebooks/
│   └── oil-price-shock-propagation.ipynb   # Analysis notebook
│
├── outputs/
│   └── audit/               # Saved audit summaries (missingness, stats)
│
├── requirements.txt
└── README.md
```

---

## Data Source

Dataset sourced from Kaggle:

https://www.kaggle.com/datasets/debashish311601/commodity-prices

The dataset contains daily commodity prices from **2000-01-03 to 2023-08-04** across:

* Energy (WTI, Brent, Natural Gas, Diesel)
* Metals (Gold, Silver, Copper, Aluminum, Nickel, Zinc)
* Agriculture (Corn, Wheat, Soybeans, Coffee, etc.)

---

## Data Audit Summary

Key findings from `src/data_audit.py`:

* **Rows:** 6,092
* **Columns:** 24
* **Date range:** 2000–2023
* **No duplicate rows**
* **No missing dates**
* Data is already numeric and well-structured

### Missing Data

* Most variables: <1% missing
* **WTI CRUDE (oil): ~0.15% missing → negligible**
* **GASOLINE: ~25% missing → dropped from analysis**

### Notable Issues

* One **negative oil price (WTI)** observed (April 2020 crash)
* Irregular spacing (trading days only: weekends/holidays missing)

---

## Data Cleaning Pipeline

Implemented in `src/clean_data.py`.

### Steps:

1. Convert `Date` to datetime and set as index
2. Sort time series chronologically
3. Drop high-missing columns (e.g., `GASOLINE`)
4. Handle invalid values:

   * Negative oil price → treated as missing
   * Non-positive values removed before log transform
5. Impute sparse missing values using forward/backward fill
6. Generate **two datasets**:

---

## Output Datasets

### 1. `commodity_prices_cleaned.csv`

* Cleaned price levels
* Shape: **(6092, 22)**

Use for:

* visualization
* descriptive statistics
* long-term trend analysis

---

### 2. `commodity_returns_cleaned.csv`

* Log returns:

  ```
  log(P_t) - log(P_{t-1})
  ```
* Shape: **(6091, 22)**

Use for:

* ARMA modeling
* mSSA analysis
* shock detection
* change point analysis

---

## Modeling Approach

The project focuses on **returns**, not raw prices.

### Why?

* Prices are **non-stationary**
* Returns are approximately **stationary**
* Shocks are defined as **large changes**, not levels

---

## Stationarity

We use log returns for modeling, which are expected to be stationary. This assumption is validated using stationarity tests, specifically the Augmented Dickey-Fuller (ADF) test and the KPSS test (see stationarity_tests.py). If this assumption were violated, transformations such as differencing would be required to induce stationarity.

### Augmented Dickey-Fuller (ADF) Test

The ADF test is used to detect the presence of a unit root in a time series.

* **Null hypothesis (H₀)**: the series has a unit root (non-stationary) 
* **Alternative hypothesis (H₁)**: the series is stationary

**Decision rule (α = 0.05):**

* p < 0.05 → reject H₀ → evidence of stationarity 
* p ≥ 0.05 → fail to reject H₀ → non-stationarity cannot be ruled out

For robustness, we also apply the KPSS test, which reverses the hypotheses (null: the series is stationary).

### Empirical Results

All commodity return series satisfy both tests (ADF: p < 0.05, KPSS: p > 0.05). These results are consistent with the assumption that log returns are stationary in mean and justify the use of stationary time series models such as ARMA, mSSA, and related approaches.

---

## Data Splitting

We use a chronological split of 70/15/15:

- Train size: 4263  
- Validation size: 914  
- Test size: 914  

---

## Oil Price Selection

Primary oil variable:

* **WTI CRUDE**

Secondary (for robustness):

* **BRENT CRUDE**

Refined products (diesel, gasoline, etc.) are retained but treated separately.

---

## Reproducibility

### Option 1 (recommended for this repo)

Dataset is included in `data/` → no setup required.

---

### Option 2 (re-download from Kaggle)

1. Create Kaggle API key
2. Place `kaggle.json` in:

   ```
   C:\Users\<username>\.kaggle\
   ```
3. Run:

   ```
   python src/download_data.py
   ```

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

---

## Next Steps

* Define oil shock variable
* Apply ARMA models to individual commodities
* Use mSSA to identify shared latent structure
* Detect structural breaks using change point analysis

---

## Key Takeaways

* Dataset required minimal cleaning
* Missingness is low and manageable
* Returns transformation is critical for valid modeling
* Oil shocks are analyzed through **returns**, not price levels

---

## Authors

Lauren Montigue 
Sabrina Queipo 
