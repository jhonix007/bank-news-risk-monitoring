"""Добавление новой размеченной части к финальному датасету."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.features.scoring import calculate_risk_score


FINAL_COLUMNS = [
    "sample_id",
    "dataset_part",
    "source",
    "url",
    "published_at",
    "published_year",
    "published_month",
    "title",
    "text_fragment",
    "entity_mention",
    "entity_norm",
    "risk_type_candidate",
    "found_risk_keywords",
    "risk_type",
    "entity_relevance",
    "alert_flag",
    "alert_reason",
    "label_quality",
    "risk_score_v1",
    "split",
    "review_status",
    "review_comment",
]
KEY_COLUMNS = ["url", "entity_norm", "text_fragment"]
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
VALID_ALERT_FLAGS = {"0", "1"}
VALID_LABEL_QUALITY = {"ok", "ambiguous", "need_review"}
VALID_REVIEW_STATUS = {"manual_checked", "need_second_review", "rejected", "need_manual_review"}


def _clean(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _has_value(value) -> bool:
    return _clean(value) != ""


def _score(row) -> int:
    risk_type_weights = {
        "no_risk": 0,
        "customer_complaints": 20,
        "operational_issue": 25,
        "fraud_phishing": 30,
        "legal_regulatory": 30,
        "sanctions": 35,
        "data_leak_security": 40,
        "other_risk": 20,
    }
    entity_relevance_weights = {"direct": 30, "indirect": 10, "mentioned_only": 0, "unclear": 0}
    alert_flag_weights = {1: 30, 0: 0}
    try:
        risk_type = _clean(row.get("risk_type"))
        entity_relevance = _clean(row.get("entity_relevance"))
        alert_flag = int(float(row.get("alert_flag")))
    except (TypeError, ValueError):
        return 0
    return min(
        100,
        risk_type_weights.get(risk_type, 0)
        + entity_relevance_weights.get(entity_relevance, 0)
        + alert_flag_weights.get(alert_flag, 0),
    )


def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    raw = df["published_at"].map(_clean) if "published_at" in df.columns else pd.Series("", index=df.index)
    numeric = pd.to_numeric(raw, errors="coerce")
    parsed_str = pd.to_datetime(raw, errors="coerce", utc=True)
    parsed_s = pd.to_datetime(numeric, unit="s", errors="coerce", utc=True)
    parsed_ms = pd.to_datetime(numeric, unit="ms", errors="coerce", utc=True)
    candidates = [parsed_str, parsed_s, parsed_ms]
    best = max(candidates, key=lambda series: (series.notna().sum(), -(series.dt.year.eq(1970).sum())))
    if best.dt.year.eq(1970).mean() > 0.8:
        valid_candidates = [series for series in candidates if series.notna().sum() > 0]
        if valid_candidates:
            best = min(valid_candidates, key=lambda series: series.dt.year.eq(1970).mean())
    df["published_year"] = best.dt.year.astype("Int64").astype(str).replace("<NA>", "")
    df["published_month"] = best.dt.month.astype("Int64").astype(str).replace("<NA>", "")
    return df


def _normalize_part(df: pd.DataFrame, default_dataset_part: str) -> pd.DataFrame:
    for column in FINAL_COLUMNS:
        if column not in df.columns:
            df[column] = ""
    df["dataset_part"] = df["dataset_part"].where(df["dataset_part"].map(_has_value), default_dataset_part)
    df["review_status"] = df["review_status"].where(df["review_status"].map(_has_value), "manual_checked")
    df["label_quality"] = df["label_quality"].where(df["label_quality"].map(_has_value), "ok")
    df["alert_flag"] = df["alert_flag"].map(lambda value: _clean(value).replace(".0", ""))
    return df


def _validate(df: pd.DataFrame) -> None:
    checks = {
        "risk_type": VALID_RISK_TYPES,
        "entity_relevance": VALID_ENTITY_RELEVANCE,
        "alert_flag": VALID_ALERT_FLAGS,
        "label_quality": VALID_LABEL_QUALITY,
        "review_status": VALID_REVIEW_STATUS,
    }
    issue_rows = []
    for idx, row in df.iterrows():
        issues = []
        for column, valid_values in checks.items():
            value = _clean(row.get(column))
            if value == "" or value not in valid_values:
                issues.append(f"{column}={value or '<empty>'}")
        if issues:
            issue_rows.append((idx, "; ".join(issues)))
    if issue_rows:
        print("Предупреждение: найдены некорректные значения в разметке:")
        for idx, issues in issue_rows[:20]:
            print(f"  row={idx}: {issues}")
        if len(issue_rows) > 20:
            print(f"  ... еще {len(issue_rows) - 20} строк")


def _group_labels(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.assign(stratum=df["risk_type"].astype(str) + "__" + df["alert_flag"].astype(str))
        .groupby("url", dropna=False)["stratum"]
        .agg(lambda values: values.mode().iat[0] if not values.mode().empty else values.iloc[0])
        .reset_index()
    )


def _can_stratify(labels: pd.Series, test_size: float) -> bool:
    counts = labels.value_counts()
    return len(counts) >= 2 and counts.min() >= 2 and round(len(labels) * test_size) >= len(counts)


def _assign_splits(df: pd.DataFrame) -> pd.DataFrame:
    group_df = _group_labels(df)
    urls = group_df["url"]
    labels = group_df["stratum"]
    split_by_url = {}
    try:
        if len(df) >= 300:
            stratify = labels if _can_stratify(labels, 0.30) else None
            if stratify is None:
                warnings.warn("Стратификация train/temp невозможна, используется group split по url.")
            train_urls, temp_urls = train_test_split(urls, test_size=0.30, random_state=42, stratify=stratify)
            temp_labels = group_df.set_index("url").loc[temp_urls, "stratum"]
            stratify_temp = temp_labels if _can_stratify(temp_labels, 0.50) else None
            if stratify_temp is None:
                warnings.warn("Стратификация valid/test невозможна, используется group split по url.")
            valid_urls, test_urls = train_test_split(temp_urls, test_size=0.50, random_state=42, stratify=stratify_temp)
            split_by_url.update({url: "train" for url in train_urls})
            split_by_url.update({url: "valid" for url in valid_urls})
            split_by_url.update({url: "test" for url in test_urls})
        else:
            train_urls, test_urls = train_test_split(urls, test_size=0.20, random_state=42)
            split_by_url.update({url: "train" for url in train_urls})
            split_by_url.update({url: "test" for url in test_urls})
    except ValueError as exc:
        warnings.warn(f"Стратификация невозможна: {exc}. Используется обычный group split по url.")
        train_urls, test_urls = train_test_split(urls, test_size=0.20, random_state=42)
        split_by_url.update({url: "train" for url in train_urls})
        split_by_url.update({url: "test" for url in test_urls})
    df["split"] = df["url"].map(split_by_url)
    return df


def append_labeled_part(base: str, extra: str, output: str) -> pd.DataFrame:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base_df = pd.read_csv(base, keep_default_na=False)
    extra_df = pd.read_csv(extra, keep_default_na=False)
    base_rows = len(base_df)
    extra_rows = len(extra_df)
    base_df = _normalize_part(base_df, "natural_sample")
    extra_df = _normalize_part(extra_df, "targeted_risk_from_runews")
    all_columns = list(dict.fromkeys(list(base_df.columns) + list(extra_df.columns) + FINAL_COLUMNS))
    for column in all_columns:
        if column not in base_df.columns:
            base_df[column] = ""
        if column not in extra_df.columns:
            extra_df[column] = ""
    combined = pd.concat([base_df[all_columns], extra_df[all_columns]], ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates(subset=KEY_COLUMNS, keep="first").reset_index(drop=True)
    removed = before - len(combined)
    combined = _parse_dates(combined)
    combined["alert_flag"] = combined["alert_flag"].map(lambda value: _clean(value).replace(".0", ""))
    _validate(combined)
    combined["risk_score_v1"] = combined.apply(_score, axis=1)
    combined = _assign_splits(combined)
    combined["sample_id"] = [f"sample_{idx:06d}" for idx in range(1, len(combined) + 1)]
    combined = combined[FINAL_COLUMNS]
    combined.to_csv(output_path, index=False)
    years = pd.to_numeric(combined["published_year"], errors="coerce")
    print("\nОтчет append_labeled_part")
    print(f"Строк в base: {base_rows}")
    print(f"Строк в extra: {extra_rows}")
    print(f"Дублей удалено: {removed}")
    print(f"Строк в итоговом файле: {len(combined)}")
    print("\nРаспределение dataset_part:")
    print(combined["dataset_part"].value_counts(dropna=False).to_string())
    print("\nРаспределение risk_type:")
    print(combined["risk_type"].value_counts(dropna=False).to_string())
    print("\nРаспределение alert_flag:")
    print(combined["alert_flag"].value_counts(dropna=False).to_string())
    print("\nРаспределение entity_relevance:")
    print(combined["entity_relevance"].value_counts(dropna=False).to_string())
    print("\nРаспределение label_quality:")
    print(combined["label_quality"].value_counts(dropna=False).to_string())
    if years.notna().any():
        print(f"\nmin/max published_year: {int(years.min())}/{int(years.max())}")
    print(f"Количество строк с published_year = 1970: {int(combined['published_year'].eq('1970').sum())}")
    print(f"Путь к итоговому файлу: {output_path}")
    return combined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="data/processed/news_risk_dataset_labeled.csv")
    parser.add_argument("--extra", default="data/processed/annotation_template_targeted_risk_assisted.csv")
    parser.add_argument("--output", default="data/processed/news_risk_dataset_labeled_v2.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    append_labeled_part(args.base, args.extra, args.output)


if __name__ == "__main__":
    main()
