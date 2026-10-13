"""Kaydedilmiş tahminlerden karşılaştırma grafiği, confusion matrix ve özet tablo üretir.

Modeller yeniden eğitilmez; modeling.py'nin ürettiği predictions.csv kullanılır,
böylece grafikler ile model_scores.json her zaman birbiriyle tutarlıdır.
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from .config import Paths
from .modeling import slug


def load_results(paths: Paths) -> dict:
    if not paths.scores_json.exists() or not paths.predictions_csv.exists():
        raise FileNotFoundError("Sonuç dosyaları yok. Önce model eğitimi adımını çalıştırın.")
    with open(paths.scores_json, encoding="utf-8") as f:
        return json.load(f)


def format_scores(data: dict) -> pd.DataFrame:
    rows = []
    for model, s in data.items():
        fraud = s.get("1", {})
        rows.append({
            "Model": model,
            "Precision": fraud.get("precision", 0.0),
            "Recall": fraud.get("recall", 0.0),
            "F1-Score": fraud.get("f1-score", 0.0),
            "PR-AUC": s.get("pr_auc", 0.0),
            "ROC-AUC": s.get("roc_auc", 0.0),
        })
    return pd.DataFrame(rows)


def plot_scores(df: pd.DataFrame, out_path) -> None:
    ax = df.set_index("Model").plot(kind="bar", figsize=(10, 6), colormap="viridis")
    ax.set_title("Anomali Tespit Modelleri - Fraud Sınıfı Metrikleri (test kümesi)")
    ax.set_ylabel("Skor")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, model_name: str, results_dir) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(5, 4))
    labels = ["Normal (0)", "Fraud (1)"]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(f"{model_name} - Confusion Matrix")
    plt.ylabel("Gerçek Etiket")
    plt.xlabel("Tahmin")
    plt.tight_layout()
    plt.savefig(results_dir / f"confusion_matrix_{slug(model_name)}.png", dpi=120)
    plt.close()


def write_summary(df: pd.DataFrame, paths: Paths) -> None:
    table = df.round(4).to_markdown(index=False)
    (paths.results_dir / "summary.md").write_text(
        "# Model Karşılaştırması (fraud sınıfı, test kümesi)\n\n" + table + "\n",
        encoding="utf-8",
    )


def main(paths: Paths | None = None):
    paths = paths or Paths.default()
    data = load_results(paths)
    preds = pd.read_csv(paths.predictions_csv)

    df = format_scores(data)
    print("\nModel Performans Tablosu:\n", df.round(4).to_string(index=False))

    paths.results_dir.mkdir(parents=True, exist_ok=True)
    plot_scores(df, paths.results_dir / "model_comparison.png")
    for model in data:
        plot_confusion_matrix(preds["y_true"], preds[f"{slug(model)}_pred"], model, paths.results_dir)
    write_summary(df, paths)
    print(f"Grafikler ve özet kaydedildi: {paths.results_dir}")


if __name__ == "__main__":
    main()
