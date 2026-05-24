"""Формирование targeted-risk выборки по узким риск-словам."""

from __future__ import annotations

import argparse
import math
import re
import warnings
from pathlib import Path

import pandas as pd


TARGETED_KEYWORDS = {
    "data_leak_security": ["утечка", "персональные данные", "данные клиентов", "взлом", "кибератака"],
    "operational_issue": [
        "сбой",
        "не работает",
        "недоступно",
        "не открывается приложение",
        "проблемы с переводами",
        "не проходят платежи",
    ],
    "fraud_phishing": ["мошенники", "мошенничество", "фишинг", "злоумышленники", "хищение"],
    "customer_complaints": ["жалобы клиентов", "клиенты жалуются", "пожаловались", "массовые жалобы"],
}

TARGETED_COLUMNS = [
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
    "found_targeted_keywords",
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


def _keyword_pattern(keyword: str) -> re.Pattern[str]:
    parts = [re.escape(part) for part in keyword.split()]
    pattern = r"\s+".join(parts)
    return re.compile(rf"(?<![\w-]){pattern}(?![\w-])", flags=re.IGNORECASE | re.UNICODE)


def _find_targeted(text: str) -> tuple[str, list[str]]:
    found_by_type = {}
    for risk_type, keywords in TARGETED_KEYWORDS.items():
        found = [keyword for keyword in keywords if _keyword_pattern(keyword).search(text)]
        if found:
            found_by_type[risk_type] = found
    for risk_type in TARGETED_KEYWORDS:
        if risk_type in found_by_type:
            return risk_type, found_by_type[risk_type]
    return "", []


def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    raw = df["published_at"].fillna("").astype(str).str.strip()
    numeric = pd.to_numeric(raw, errors="coerce")
    parsed_str = pd.to_datetime(raw, errors="coerce", utc=True)
    parsed_s = pd.to_datetime(numeric, unit="s", errors="coerce", utc=True)
    parsed_ms = pd.to_datetime(numeric, unit="ms", errors="coerce", utc=True)
    candidates = [parsed_str, parsed_s, parsed_ms]
    best = max(candidates, key=lambda series: (series.notna().sum(), -(series.dt.year.eq(1970).sum())))
    df["published_year"] = best.dt.year.astype("Int64").astype(str).replace("<NA>", "")
    df["published_month"] = best.dt.month.astype("Int64").astype(str).replace("<NA>", "")
    return df


def _exclude_existing(df: pd.DataFrame, existing_path: str) -> pd.DataFrame:
    existing = pd.read_csv(existing_path, keep_default_na=False)
    existing_keys = set(map(tuple, existing[KEY_COLUMNS].fillna("").astype(str).to_numpy()))
    current_keys = df[KEY_COLUMNS].fillna("").astype(str).apply(tuple, axis=1)
    return df.loc[~current_keys.isin(existing_keys)].copy().reset_index(drop=True)


def _quota_sample(df: pd.DataFrame, rows: int, seed: int) -> pd.DataFrame:
    selected_indices: list[int] = []
    selected_global: set[int] = set()
    entity_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    entity_cap = max(1, math.ceil(rows * 0.25))
    source_cap = max(1, math.ceil(rows * 0.30))

    def can_take(index: int, use_caps: bool) -> bool:
        if index in selected_global:
            return False
        if not use_caps:
            return True
        entity = str(df.at[index, "entity_norm"])
        source = str(df.at[index, "source"])
        return entity_counts.get(entity, 0) < entity_cap and source_counts.get(source, 0) < source_cap

    def take(index: int) -> None:
        selected_indices.append(index)
        selected_global.add(index)
        entity = str(df.at[index, "entity_norm"])
        source = str(df.at[index, "source"])
        entity_counts[entity] = entity_counts.get(entity, 0) + 1
        source_counts[source] = source_counts.get(source, 0) + 1

    per_type_quota = max(1, rows // len(TARGETED_KEYWORDS))
    for offset, risk_type in enumerate(TARGETED_KEYWORDS):
        bucket = df[df["risk_type_candidate"] == risk_type].sample(frac=1, random_state=seed + offset)
        if len(bucket) < per_type_quota:
            warnings.warn(f"Для {risk_type} найдено меньше {per_type_quota} строк: {len(bucket)}.")
        for index in bucket.index:
            if len([idx for idx in selected_indices if df.at[idx, "risk_type_candidate"] == risk_type]) >= per_type_quota:
                break
            if can_take(index, use_caps=True):
                take(index)

    if len(selected_indices) < rows:
        leftovers = df.drop(index=selected_indices, errors="ignore").sample(frac=1, random_state=seed + 100)
        for index in leftovers.index:
            if len(selected_indices) >= rows:
                break
            if can_take(index, use_caps=True):
                take(index)

    if len(selected_indices) < rows:
        warnings.warn("Не удалось добрать rows с лимитами entity/source, добираем без лимитов.")
        leftovers = df.drop(index=selected_indices, errors="ignore").sample(frac=1, random_state=seed + 200)
        for index in leftovers.index:
            if len(selected_indices) >= rows:
                break
            if can_take(index, use_caps=False):
                take(index)

    return df.loc[selected_indices].sample(frac=1, random_state=seed + 300).reset_index(drop=True)


def build_targeted_sample(input_path: str, existing_path: str, output: str, rows: int, seed: int) -> pd.DataFrame:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_path, keep_default_na=False)
    df = _exclude_existing(df, existing_path)

    text = (df["title"].fillna("") + " " + df["text_fragment"].fillna("")).astype(str)
    matches = text.map(_find_targeted)
    df["risk_type_candidate"] = matches.map(lambda item: item[0])
    df["found_targeted_keywords"] = matches.map(lambda item: "; ".join(item[1]))
    df = df[df["risk_type_candidate"] != ""].copy()
    df["found_risk_keywords"] = df["found_targeted_keywords"]

    found_counts = df["risk_type_candidate"].value_counts()
    sampled = _quota_sample(df, rows, seed) if len(df) > rows else df.sample(frac=1, random_state=seed).reset_index(drop=True)
    if len(df) <= rows:
        warnings.warn("Targeted-risk кандидатов меньше или равно rows, используется весь доступный пул.")

    sampled = _parse_dates(sampled)
    sampled.insert(0, "sample_id", [f"tr_{idx:06d}" for idx in range(1, len(sampled) + 1)])
    sampled.insert(1, "dataset_part", "targeted_risk_sample")
    for column in ["risk_type", "entity_relevance", "alert_flag", "alert_reason", "label_quality", "risk_score_v1", "split", "review_status", "review_comment"]:
        sampled[column] = ""
    for column in TARGETED_COLUMNS:
        if column not in sampled.columns:
            sampled[column] = ""
    sampled = sampled[TARGETED_COLUMNS]
    sampled.to_csv(output_path, index=False)

    _print_report(sampled, found_counts, output_path)
    return sampled


def _print_report(sampled: pd.DataFrame, found_counts: pd.Series, output_path: Path) -> None:
    years = pd.to_numeric(sampled["published_year"], errors="coerce")
    print("\nTargeted-risk report")
    print("Найдено строк по targeted risk_type_candidate:")
    print(found_counts.reindex(list(TARGETED_KEYWORDS), fill_value=0).to_string())
    print(f"\nСтрок вошло в {output_path}: {len(sampled)}")
    print("\nРаспределение entity_norm:")
    print(sampled["entity_norm"].value_counts().to_string())
    print("\nРаспределение source:")
    print(sampled["source"].value_counts().to_string())
    if years.notna().any():
        print(f"\nmin/max published_year: {int(years.min())}/{int(years.max())}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/interim/bank_news_candidates_pool.csv")
    parser.add_argument("--existing", default="data/processed/news_risk_dataset_labeled.csv")
    parser.add_argument("--output", default="data/processed/annotation_template_targeted_risk.csv")
    parser.add_argument("--rows", type=int, default=200)
    parser.add_argument("--seed", type=int, default=44)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_targeted_sample(args.input, args.existing, args.output, args.rows, args.seed)


if __name__ == "__main__":
    main()
