"""Üç anomali tespit modelini eğitir ve test kümesinde değerlendirir.

Modeller yalnızca normal işlemlerle eğitilir; fraud etiketi eğitimde kullanılmaz.
Değerlendirme, modelin hiç görmediği test kümesi üzerinde yapılır.
"""

import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from .config import CONTAMINATION, MAX_TRAIN_SAMPLES, RANDOM_STATE, Paths


def slug(name: str) -> str:
    return name.replace(" ", "_")


def build_models() -> dict:
    return {
        "Isolation Forest": IsolationForest(
            n_estimators=100, contamination=CONTAMINATION, random_state=RANDOM_STATE, n_jobs=-1
        ),
        # novelty=True: eğitimde görülmeyen veriye predict/decision_function uygulanabilir
        "Local Outlier Factor": LocalOutlierFactor(
            n_neighbors=20, contamination=CONTAMINATION, novelty=True
        ),
        "One-Class SVM": OneClassSVM(kernel="rbf", gamma="scale", nu=CONTAMINATION),
    }


def load_split(paths: Paths):
    train = pd.read_csv(paths.train_csv)
    test = pd.read_csv(paths.test_csv)
    # numpy dizileri: eğitim ve tahminde sütun adı uyuşmazlığı uyarısı çıkmasın
    X_train = train.drop(columns=["Class"]).to_numpy()
    X_test = test.drop(columns=["Class"]).to_numpy()
    return X_train, X_test, test["Class"]


def train_and_evaluate(X_train, X_test, y_test):
    results, predictions = {}, pd.DataFrame({"y_true": y_test.values})

    for name, model in build_models().items():
        cap = MAX_TRAIN_SAMPLES.get(name)
        if cap and len(X_train) > cap:
            idx = np.random.default_rng(RANDOM_STATE).choice(len(X_train), size=cap, replace=False)
            X_fit = X_train[idx]
        else:
            X_fit = X_train

        print(f"\nEğitiliyor: {name} ({len(X_fit)} örnek)")
        start = time.perf_counter()
        model.fit(X_fit)
        train_seconds = time.perf_counter() - start

        pred = (model.predict(X_test) == -1).astype(int)  # sklearn: -1 = anomali
        score = -model.decision_function(X_test)          # yüksek = daha anormal

        report = classification_report(y_test, pred, output_dict=True, zero_division=0)
        report["roc_auc"] = float(roc_auc_score(y_test, score))
        report["pr_auc"] = float(average_precision_score(y_test, score))
        report["train_seconds"] = round(train_seconds, 2)
        report["train_samples"] = int(len(X_fit))
        results[name] = report

        predictions[f"{slug(name)}_pred"] = pred
        predictions[f"{slug(name)}_score"] = score

        print(classification_report(y_test, pred, zero_division=0))
        print(f"ROC-AUC: {report['roc_auc']:.4f} | PR-AUC: {report['pr_auc']:.4f}")

    return results, predictions


def save_results(results: dict, predictions: pd.DataFrame, paths: Paths) -> None:
    paths.results_dir.mkdir(parents=True, exist_ok=True)
    with open(paths.scores_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    predictions.to_csv(paths.predictions_csv, index=False)
    print(f"\nSkorlar kaydedildi: {paths.scores_json}")


def main(paths: Paths | None = None):
    paths = paths or Paths.default()
    if not paths.train_csv.exists() or not paths.test_csv.exists():
        raise FileNotFoundError("İşlenmiş veri yok. Önce ön işleme adımını çalıştırın.")
    X_train, X_test, y_test = load_split(paths)
    results, predictions = train_and_evaluate(X_train, X_test, y_test)
    save_results(results, predictions, paths)


if __name__ == "__main__":
    main()
