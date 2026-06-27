from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import FeatureUnion


SEED = 42
ALPHA = 0.5
C = 1.0
THRESHOLD = 0.5
TEXT_COLUMNS = ["title", "entity_norm", "text_fragment"]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "homework_04_dataset" / "data" / "dataset_for_training.csv"
TEACHER_PATH = PROJECT_ROOT / "homework_07_postprocessing" / "reports" / "teacher_labels" / "train_teacher_prompt_v2b.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "tfidf_student_v2b.joblib"


def build_model_text(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["title"].fillna("").astype(str)
        + "\n"
        + frame["entity_norm"].fillna("").astype(str)
        + "\n"
        + frame["text_fragment"].fillna("").astype(str)
    )


def build_vectorizer() -> FeatureUnion:
    return FeatureUnion(
        [
            (
                "word_tfidf",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.9,
                    max_features=30000,
                ),
            ),
            (
                "char_tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_df=0.9,
                    max_features=30000,
                ),
            ),
        ]
    )


def fit_logreg_with_soft_labels(X, soft_y: np.ndarray) -> LogisticRegression:
    soft_y = np.clip(np.asarray(soft_y, dtype=float), 1e-4, 1 - 1e-4)
    X_aug = sparse.vstack([X, X])
    y_aug = np.concatenate(
        [
            np.ones(len(soft_y), dtype=int),
            np.zeros(len(soft_y), dtype=int),
        ]
    )
    sample_weight = np.concatenate([soft_y, 1.0 - soft_y])

    model = LogisticRegression(
        C=C,
        max_iter=2000,
        random_state=SEED,
        solver="liblinear",
    )
    model.fit(X_aug, y_aug, sample_weight=sample_weight)
    return model


def binary_metrics(y_true: np.ndarray, score: np.ndarray) -> dict[str, float]:
    pred = (score >= THRESHOLD).astype(int)
    return {
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, score),
        "pr_auc": average_precision_score(y_true, score),
    }


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    teacher_df = pd.read_csv(TEACHER_PATH)

    required_data_columns = {"sample_id", "split", "alert_flag", *TEXT_COLUMNS}
    missing_data_columns = sorted(required_data_columns - set(df.columns))
    if missing_data_columns:
        raise ValueError(f"Missing data columns: {missing_data_columns}")

    required_teacher_columns = {"sample_id", "teacher_label", "teacher_confidence"}
    missing_teacher_columns = sorted(required_teacher_columns - set(teacher_df.columns))
    if missing_teacher_columns:
        raise ValueError(f"Missing teacher columns: {missing_teacher_columns}")

    df["model_text"] = build_model_text(df)
    train_df = df[df["split"] == "train"].copy()
    valid_df = df[df["split"] == "valid"].copy()
    test_df = df[df["split"] == "test"].copy()

    train_teacher = train_df[["sample_id", "model_text", "alert_flag"]].merge(
        teacher_df[["sample_id", "teacher_label", "teacher_confidence"]],
        on="sample_id",
        how="inner",
        validate="one_to_one",
    )
    train_teacher = train_teacher[
        train_teacher["teacher_label"].isin([0, 1])
        & train_teacher["teacher_confidence"].notna()
    ].copy()

    teacher_label = train_teacher["teacher_label"].astype(int).to_numpy()
    teacher_confidence = train_teacher["teacher_confidence"].astype(float).clip(0.0, 1.0).to_numpy()
    teacher_soft_y = np.where(teacher_label == 1, teacher_confidence, 1.0 - teacher_confidence)
    teacher_soft_y = np.clip(teacher_soft_y, 1e-4, 1 - 1e-4)

    y_train_hard = train_teacher["alert_flag"].astype(float).to_numpy()
    soft_y = np.clip((1 - ALPHA) * y_train_hard + ALPHA * teacher_soft_y, 1e-4, 1 - 1e-4)

    vectorizer = build_vectorizer()
    X_train = vectorizer.fit_transform(train_teacher["model_text"])
    model = fit_logreg_with_soft_labels(X_train, soft_y)

    valid_score = model.predict_proba(vectorizer.transform(valid_df["model_text"]))[:, 1]
    test_score = model.predict_proba(vectorizer.transform(test_df["model_text"]))[:, 1]
    valid_metrics = binary_metrics(valid_df["alert_flag"].astype(int).to_numpy(), valid_score)
    test_metrics = binary_metrics(test_df["alert_flag"].astype(int).to_numpy(), test_score)

    artifact = {
        "model_name": "tfidf_student_v2b",
        "model_family": "Combined TF-IDF + LogisticRegression",
        "teacher_source": "teacher_prompt_v2b",
        "alpha": ALPHA,
        "C": C,
        "threshold": THRESHOLD,
        "text_columns": TEXT_COLUMNS,
        "vectorizer": vectorizer,
        "classifier": model,
        "valid_metrics": valid_metrics,
        "test_metrics": test_metrics,
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)

    print(f"Saved model artifact: {MODEL_PATH}")
    print("Valid metrics:", {key: round(value, 6) for key, value in valid_metrics.items()})
    print("Test metrics:", {key: round(value, 6) for key, value in test_metrics.items()})


if __name__ == "__main__":
    main()
