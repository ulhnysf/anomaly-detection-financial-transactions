"""Kaggle verisi olmadan projeyi denemek için sentetik demo verisi üretir.

Şema, Kaggle "Credit Card Fraud Detection" veri setiyle aynıdır
(Time, V1-V28, Amount, Class). Değerler GERÇEK DEĞİLDİR; yalnızca
boru hattını (pipeline) uçtan uca çalıştırmak içindir.
"""

import numpy as np
import pandas as pd

from .config import RANDOM_STATE, Paths


def make_demo_dataframe(n_rows: int = 20_000, fraud_rate: float = 0.004,
                        seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_fraud = max(2, int(n_rows * fraud_rate))
    n_normal = n_rows - n_fraud

    normal = rng.normal(0.0, 1.0, size=(n_normal, 28))
    # Fraud işlemleri: bazı bileşenlerde kayma ve daha geniş yayılım
    fraud = rng.normal(0.0, 2.0, size=(n_fraud, 28))
    fraud[:, [0, 2, 3, 9, 13]] += rng.normal(-3.0, 1.0, size=(n_fraud, 5))

    features = np.vstack([normal, fraud])
    df = pd.DataFrame(features, columns=[f"V{i}" for i in range(1, 29)])
    df.insert(0, "Time", rng.uniform(0, 172_800, size=n_rows))
    df["Amount"] = np.round(rng.lognormal(3.0, 1.2, size=n_rows), 2)
    df["Class"] = [0] * n_normal + [1] * n_fraud
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main(paths: Paths | None = None, n_rows: int = 20_000) -> Paths:
    paths = paths or Paths.demo()
    paths.raw_csv.parent.mkdir(parents=True, exist_ok=True)
    df = make_demo_dataframe(n_rows)
    df.to_csv(paths.raw_csv, index=False)
    print(f"Demo verisi üretildi: {paths.raw_csv} "
          f"({len(df)} satır, {int(df['Class'].sum())} fraud)")
    return paths


if __name__ == "__main__":
    main()
