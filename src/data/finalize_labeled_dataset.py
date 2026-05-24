"""Проверка ручной разметки и сбор финального датасета."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.features.scoring import calculate_risk_score


VALID_RISK_TYPES = {
    "no_risk",
    "operational_issue",
    "data_leak_security",
    "fraud_phishing",
    "legal_regulatory",
    "sanctions",
    "customer_complaints",
    "other_risk",
}
VALID_ENTITY_RELEVANCE = {"direct", "indirect", "mentioned_only", "unclear"}
VALID_ALERT_FLAGS = {0, 1}
VALID_LABEL_QUALITY = {"ok", "ambiguous", "need_review"}


def _not_empty(series: pd.Series) -> pd.Series:
    return series.notna() & (series.astype(str).str.strip() != "")


def validate_labels(df: pd.DataFrame) -> None:
    required = ["risk_type", "entity_relevance", "alert_flag", "label_quality"]
    errors = []

    for column in required:
        if column not in df.columns:
            errors.append(f"Нет обязательной колонки: {column}")
        elif not _not_empty(df[column]).all():
            bad = df.index[~_not_empty(df[column])].tolist()[:10]
            errors.append(f"Пустые значения в {column}, строки: {bad}")

    if errors:
        raise ValueError("\n".join(errors))

    alert_flag = pd.to_numeric(df["alert_flag"], errors="coerce")
    invalid_alert = ~alert_flag.isin(VALID_ALERT_FLAGS)
    if invalid_alert.any():
        errors.append(f"Недопустимые alert_flag, строки: {df.index[invalid_alert].tolist()[:10]}")

    checks = {
        "risk_type": VALID_RISK_TYPES,
        "entity_relevance": VALID_ENTITY_RELEVANCE,
        "label_quality": VALID_LABEL_QUALITY,
    }
    for column, valid_values in checks.items():
        invalid = ~df[column].astype(str).str.strip().isin(valid_values)
        if invalid.any():
            errors.append(f"Недопустимые значения {column}, строки: {df.index[invalid].tolist()[:10]}")

    need_reason = alert_flag.eq(1) & ~_not_empty(df.get("alert_reason", pd.Series("", index=df.index)))
    if need_reason.any():
        errors.append(f"alert_reason обязателен при alert_flag=1, строки: {df.index[need_reason].tolist()[:10]}")

    if errors:
        raise ValueError("\n".join(errors))


def _group_labels(df: pd.DataFrame) -> pd.DataFrame:
    group_df = (
        df.assign(stratum=df["risk_type"].astype(str) + "__" + df["alert_flag"].astype(str))
        .groupby("url", dropna=False)["stratum"]
        .agg(lambda values: values.mode().iat[0] if not values.mode().empty else values.iloc[0])
        .reset_index()
    )
    return group_df


def _can_stratify(labels: pd.Series, test_size: float) -> bool:
    counts = labels.value_counts()
    if len(counts) < 2 or counts.min() < 2:
        return False
    return round(len(labels) * test_size) >= len(counts)


def assign_splits(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if df.empty:
        df["split"] = pd.Series(dtype="object")
        return df

    group_df = _group_labels(df)
    urls = group_df["url"]
    labels = group_df["stratum"]
    n_rows = len(df)

    split_by_url = {}
    if len(urls) < 2:
        warnings.warn("Недостаточно уникальных url для train/test split, все строки помещены в train.")
        df["split"] = "train"
        return df

    try:
        if n_rows >= 300:
            stratify = labels if _can_stratify(labels, 0.30) else None
            if stratify is None:
                warnings.warn("Стратификация train/temp невозможна, используется group split по url.")
            train_urls, temp_urls = train_test_split(urls, test_size=0.30, random_state=42, stratify=stratify)
            temp_labels = group_df.set_index("url").loc[temp_urls, "stratum"]
            stratify_temp = temp_labels if _can_stratify(temp_labels, 0.50) else None
            if stratify_temp is None:
                warnings.warn("Стратификация valid/test невозможна, используется group split по url.")
            valid_urls, test_urls = train_test_split(
                temp_urls, test_size=0.50, random_state=42, stratify=stratify_temp
            )
            split_by_url.update({url: "train" for url in train_urls})
            split_by_url.update({url: "valid" for url in valid_urls})
            split_by_url.update({url: "test" for url in test_urls})
        else:
            stratify = labels if _can_stratify(labels, 0.20) else None
            if stratify is None:
                warnings.warn("Стратификация train/test невозможна, используется group split по url.")
            train_urls, test_urls = train_test_split(urls, test_size=0.20, random_state=42, stratify=stratify)
            split_by_url.update({url: "train" for url in train_urls})
            split_by_url.update({url: "test" for url in test_urls})
    except ValueError as exc:
        warnings.warn(f"Стратификация невозможна: {exc}. Используется обычный group split по url.")
        train_urls, test_urls = train_test_split(urls, test_size=0.20, random_state=42)
        split_by_url.update({url: "train" for url in train_urls})
        split_by_url.update({url: "test" for url in test_urls})

    df["split"] = df["url"].map(split_by_url)
    return df


def _false_match_mask(df: pd.DataFrame) -> pd.Series:
    if "review_comment" not in df.columns:
        return pd.Series(False, index=df.index)

    return df["label_quality"].astype(str).str.strip().eq("need_review") & df["review_comment"].fillna("").astype(
        str
    ).str.lower().str.contains("ложное совпадение", regex=False)


def finalize_labeled_dataset(
    input_path: str,
    output: str,
    rejected_output: str = "data/processed/rejected_after_review.csv",
) -> pd.DataFrame:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rejected_output_path = Path(rejected_output)
    rejected_output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    validate_labels(df)

    false_match_mask = _false_match_mask(df)
    rejected_df = df.loc[false_match_mask].copy()
    rejected_df.to_csv(rejected_output_path, index=False)
    df = df.loc[~false_match_mask].copy()

    df["alert_flag"] = pd.to_numeric(df["alert_flag"], errors="raise").astype(int)
    df["risk_score_v1"] = df.apply(calculate_risk_score, axis=1)
    df = assign_splits(df)
    df.to_csv(output_path, index=False)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/annotation_template.csv")
    parser.add_argument("--output", default="data/processed/news_risk_dataset_labeled.csv")
    parser.add_argument("--rejected_output", default="data/processed/rejected_after_review.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = finalize_labeled_dataset(args.input, args.output, args.rejected_output)
    print(f"Сохранено строк финального датасета: {len(df)}")
    print(f"Файл: {args.output}")
    print(f"Отклоненные после ручной проверки: {args.rejected_output}")


if __name__ == "__main__":
    main()
