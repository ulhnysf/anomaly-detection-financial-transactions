"""Uçtan uca doğrulama testleri (sentetik veriyle, birkaç saniye sürer)."""

import json
import sqlite3

import pandas as pd
import pytest

from scripts import evaluation, make_demo_data, modeling, preprocessing, sql_queries
from scripts.config import Paths


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    paths = Paths.under(tmp_path_factory.mktemp("run"))
    df = make_demo_data.make_demo_dataframe(n_rows=3000, fraud_rate=0.01)
    df.to_csv(paths.raw_csv, index=False)
    preprocessing.main(paths)
    modeling.main(paths)
    evaluation.main(paths)
    return paths, df


def test_demo_data_schema():
    df = make_demo_data.make_demo_dataframe(n_rows=500)
    assert list(df.columns) == ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
    assert set(df["Class"].unique()) == {0, 1}


def test_train_contains_only_normal_and_test_has_fraud(pipeline):
    paths, df = pipeline
    train, test = pd.read_csv(paths.train_csv), pd.read_csv(paths.test_csv)
    assert (train["Class"] == 0).all()
    assert test["Class"].sum() > 0
    # eğitim (yalnızca normal) + test, tüm veriyi aşmamalı
    assert len(train) + len(test) <= len(df)


def test_scaler_fit_on_train_only(pipeline):
    paths, _ = pipeline
    train = pd.read_csv(paths.train_csv)
    assert abs(train["Amount"].mean()) < 1e-6
    assert abs(train["Amount"].std(ddof=0) - 1) < 1e-6


def test_scores_json_and_metrics_range(pipeline):
    paths, _ = pipeline
    scores = json.loads(paths.scores_json.read_text(encoding="utf-8"))
    assert set(scores) == {"Isolation Forest", "Local Outlier Factor", "One-Class SVM"}
    for s in scores.values():
        for key in ("roc_auc", "pr_auc"):
            assert -1e-9 <= s[key] <= 1 + 1e-9
        assert 0.0 <= s["1"]["recall"] <= 1.0


def test_outputs_exist_and_are_consistent(pipeline):
    paths, _ = pipeline
    for name in ["class_distribution.png", "model_comparison.png", "summary.md",
                 "confusion_matrix_Isolation_Forest.png",
                 "confusion_matrix_Local_Outlier_Factor.png",
                 "confusion_matrix_One-Class_SVM.png"]:
        assert (paths.results_dir / name).exists(), name
    preds = pd.read_csv(paths.predictions_csv)
    assert len(preds) == len(pd.read_csv(paths.test_csv))


def test_sql_counts_match_csv(pipeline):
    paths, df = pipeline
    out = sql_queries.main(paths)
    assert out["Toplam işlem sayısı"][0][0] == len(df)
    assert out["Fraud (Class = 1) işlem sayısı"][0][0] == int(df["Class"].sum())
    with sqlite3.connect(paths.db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0] == len(df)


def test_missing_data_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Kaggle"):
        preprocessing.main(Paths.under(tmp_path))
