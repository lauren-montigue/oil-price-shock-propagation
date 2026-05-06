import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox

def main():

    # 1. Load data
    train = pd.read_csv("data/splits/train.csv")
    val = pd.read_csv("data/splits/val.csv")

    train_y = pd.to_numeric(train["WTI CRUDE"], errors="coerce").dropna()
    val_y = pd.to_numeric(val["WTI CRUDE"], errors="coerce").dropna()

    full_series = pd.concat([train_y, val_y]).reset_index(drop=True)

    # 2. Load best ARMA order
    results = pd.read_csv("outputs/shocks/sorted_by_aic.csv")

    best_row = results.loc[results["aic"].idxmin()]
    p, q = int(best_row.p), int(best_row.q)

    print(f"Using ARMA({p},{q}) for shock extraction")

    # 3. Fit ARMA model
    model = ARIMA(full_series, order=(p, 0, q)).fit()

    # 4. Extract shocks
    shocks = model.resid

    # 5. Save shocks
    os.makedirs("outputs/shocks", exist_ok=True)

    shocks_df = pd.DataFrame({
        "oil_shock": shocks
    })

    shocks_df.to_csv("outputs/shocks/oil_shocks.csv", index=False)

    # 6. Plot shocks over time
    plt.figure(figsize=(14,6))
    plt.plot(shocks, linewidth=0.8)
    plt.axhline(0, color="black", linewidth=1)
    plt.title("WTI Crude Oil Shocks (ARMA Residuals)")
    plt.xlabel("Time Index")
    plt.ylabel("Shock")
    plt.tight_layout()
    plt.savefig("outputs/shocks/oil_shocks_plot.png")
    plt.show()

    # 7. ACF diagnostic
    plt.figure(figsize=(10,5))
    plot_acf(shocks, lags=40)
    plt.title("ACF of Oil Shocks")
    plt.tight_layout()
    plt.show()

    # 8. Ljung-Box test
    lb_test = acorr_ljungbox(shocks, lags=[10, 20], return_df=True)
    print("\nLjung-Box test:")
    print(lb_test)


if __name__ == "__main__":
    main()
