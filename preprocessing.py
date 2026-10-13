"""Veri ön işleme: kontrol, sınıf dağılımı grafiği, bölme ve ölçekleme.

Önemli noktalar
- Veri, ölçeklemeden ÖNCE eğitim/test olarak bölünür (veri sızıntısı olmasın).
- Ölçekleyici yalnızca normal (Class=0) eğitim işlemlerine fit edilir.
- Eğitim kümesi yalnızca normal işlemlerden oluşur (gözetimsiz anomali tespiti);
  test kümesi hem normal hem fraud işlemleri içerir.
"""

import matplotlib

matplotlib.use("Agg")  # ekransız ortamlarda da çalışsın
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import RANDOM_STATE, SCALE_COLUMNS, TEST_SIZE, Paths

REQUIRED_COLUMNS = {"Time", "Amount", "Class"}


def load_data(path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Veri dosyası bulunamadı: {path}\n"
            "Kaggle'dan creditcard.csv dosyasını indirip data/ klasörüne koyun "
            "(bkz. data/README.md) ya da `python main.py --demo` ile demo verisini kullanın."
        )
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Veri setinde beklenen sütunlar yok: {sorted(missing)}")
    return df


def check_missing(df: pd.DataFrame) -> int:
    total = int(df.isnull().sum().sum())
    print(f"Eksik değer sayısı: {total}")
    return total


def class_distribution(df: pd.DataFrame, out_path) -> None:
    counts = df["Class"].value_counts().sort_index()
    print("\nSınıf dağılımı:")
    print(counts.to_string())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 4))
    ax = sns.countplot(x="Class", data=df)
    ax.set_yscale("log")  # 492 fraud, 284k normal: lineer ölçekte görünmez
    ax.set_xticks([0, 1], ["Normal (0)", "Fraud (1)"])
    ax.bar_label(ax.containers[0])
    plt.title("Normal vs Fraud Dağılımı (log ölçek)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def split_and_scale(df: pd.DataFrame):
    """(train_normal, test) döndürür; ikisi de ölçeklenmiş DataFrame'dir."""
    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, stratify=df["Class"], random_state=RANDOM_STATE
    )
    train_normal = train_df[train_df["Class"] == 0].copy()
    test_df = test_df.copy()

    cols = [c for c in SCALE_COLUMNS if c in df.columns]
    scaler = StandardScaler().fit(train_normal[cols])
    train_normal[cols] = scaler.transform(train_normal[cols])
    test_df[cols] = scaler.transform(test_df[cols])
    return train_normal.reset_index(drop=True), test_df.reset_index(drop=True)


def main(paths: Paths | None = None):
    paths = paths or Paths.default()
    df = load_data(paths.raw_csv)
    print(f"{len(df)} satır, {df.shape[1]} sütun yüklendi.")
    check_missing(df)
    class_distribution(df, paths.results_dir / "class_distribution.png")

    train_normal, test_df = split_and_scale(df)
    paths.processed_dir.mkdir(parents=True, exist_ok=True)
    train_normal.to_csv(paths.train_csv, index=False)
    test_df.to_csv(paths.test_csv, index=False)
    print(f"\nEğitim (yalnızca normal): {len(train_normal)} satır")
    print(f"Test: {len(test_df)} satır, {int(test_df['Class'].sum())} fraud")
    print("Ön işleme tamamlandı.")


if __name__ == "__main__":
    main()
