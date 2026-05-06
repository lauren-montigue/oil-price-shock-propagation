import pandas as pd
import numpy as np
import itertools
import os
from statsmodels.tsa.statespace.sarimax import SARIMAX

P_MAX = 2
Q_MAX = 2
MAX_SHOCK_LAG = 3

DROP_COLS = [
    "WTI CRUDE",
    "BRENT CRUDE",
    "ULS DIESEL",
    "LOW SULPHUR GAS OIL"
]


def create_shock_lags(shocks, max_lag):

    return pd.DataFrame({
        f"oil_shock_lag_{lag}": shocks.shift(lag)
        for lag in range(max_lag + 1)
    })


def select_order(y, shock_df):

    best_bic = np.inf
    best_order = None

    model_df = pd.concat([y, shock_df], axis=1).dropna()

    y_clean = model_df[y.name]
    X_clean = model_df.drop(columns=[y.name])

    for p, q in itertools.product(
        range(P_MAX + 1),
        range(Q_MAX + 1)
    ):

        try:

            fitted = SARIMAX(
                y_clean,
                exog=X_clean,
                order=(p, 0, q),
                enforce_stationarity=False,
                enforce_invertibility=False
            ).fit(disp=False)

            if fitted.bic < best_bic:

                best_bic = fitted.bic
                best_order = (p, q)

        except:
            continue

    return best_order


def fit_model(y, shock_df, order):

    model_df = pd.concat([y, shock_df], axis=1).dropna()

    y_clean = model_df[y.name]
    X_clean = model_df.drop(columns=[y.name])

    return SARIMAX(
        y_clean,
        exog=X_clean,
        order=(order[0], 0, order[1]),
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False)


def main():

    train = pd.read_csv("data/splits/train.csv")
    val = pd.read_csv("data/splits/val.csv")
    test = pd.read_csv("data/splits/test.csv")

    full_df = pd.concat([train, val, test]).reset_index(drop=True)
    trainval_df = pd.concat([train, val]).reset_index(drop=True)

    for df in [trainval_df, full_df]:

        df.drop(columns=DROP_COLS, errors="ignore", inplace=True)

        for col in df.columns:

            df[col] = pd.to_numeric(df[col], errors="coerce")

    shocks = pd.read_csv(
        "outputs/shocks/oil_shocks.csv"
    )["oil_shock"]

    shock_lags = create_shock_lags(
        shocks,
        MAX_SHOCK_LAG
    )

    results = []

    for commodity in trainval_df.columns:

        print(f"Processing {commodity}")

        try:

            # select order on train + val
            order = select_order(
                trainval_df[commodity],
                shock_lags
            )

            # refit on full dataset
            fitted = fit_model(
                full_df[commodity],
                shock_lags,
                order
            )

            row = {
                "commodity": commodity,
                "p": order[0],
                "q": order[1],
                "aic": fitted.aic,
                "bic": fitted.bic
            }

            for param in fitted.params.index:

                row[f"{param}_coef"] = fitted.params[param]

            for param in fitted.pvalues.index:

                row[f"{param}_pvalue"] = fitted.pvalues[param]

            results.append(row)

        except Exception as e:

            print(f"Failed: {commodity}")
            print(e)

    os.makedirs("outputs/transmission", exist_ok=True)

    pd.DataFrame(results).to_csv(
        "outputs/transmission/arma_transmission_results.csv",
        index=False
    )


if __name__ == "__main__":
    main()

