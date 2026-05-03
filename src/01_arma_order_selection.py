import pandas as pd
import numpy as np
import itertools
import os
from statsmodels.tsa.arima.model import ARIMA

def run_grid_search(train_y, val_y, p_max=4, q_max=4):
    results = []

    for p, q in itertools.product(range(p_max + 1), range(q_max + 1)):
        try:
            model = ARIMA(train_y, order=(p, 0, q)).fit()

            aic = model.aic
            bic = model.bic

            forecast = model.forecast(steps=len(val_y))
            val_mse = np.mean((val_y.values[:len(forecast)] - forecast) ** 2)

            results.append({
                "p": p,
                "q": q,
                "aic": aic,
                "bic": bic,
                "val_mse": val_mse
            })

        except:
            continue

    return pd.DataFrame(results)


def main():
    # Load data
    train = pd.read_csv("data/splits/train.csv")
    val = pd.read_csv("data/splits/val.csv")

    train_y = pd.to_numeric(train["WTI CRUDE"], errors="coerce").dropna()
    val_y = pd.to_numeric(val["WTI CRUDE"], errors="coerce").dropna()

    # Run grid search
    res_df = run_grid_search(train_y, val_y)

    # Save outputs
    os.makedirs("outputs/shocks", exist_ok=True)
    
    res_df.to_csv("outputs/shocks/arma_grid_results.csv", index=False)
    res_df.sort_values("aic").to_csv("outputs/shocks/sorted_by_aic.csv", index=False)
    res_df.sort_values("bic").to_csv("outputs/shocks/sorted_by_bic.csv", index=False)
    res_df.sort_values("val_mse").to_csv("outputs/shocks/sorted_by_val_mse.csv", index=False)

    # Best models
    best_aic = res_df.loc[res_df["aic"].idxmin()]
    best_bic = res_df.loc[res_df["bic"].idxmin()]
    best_val = res_df.loc[res_df["val_mse"].idxmin()]

    print("Best AIC order:", (int(best_aic.p), int(best_aic.q)))
    print("Best BIC order:", (int(best_bic.p), int(best_bic.q)))
    print("Best VAL order:", (int(best_val.p), int(best_val.q)))

if __name__ == "__main__":
    main()