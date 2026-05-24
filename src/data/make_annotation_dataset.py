"""Формирование шаблона для ручной разметки."""

from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path

import pandas as pd

from src.features.risk_keywords import detect_risk_type_candidate, find_risk_keywords


ANNOTATION_COLUMNS = [
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
    "found_alias",
    "match_type",
    "match_confidence",
    "risk_type_candidate",
    "found_risk_keywords",
    "risk_type",
    "entity_relevance",
    "alert_flag_candidate",
    "alert_flag",
    "alert_reason",
    "label_quality",
    "risk_score_v1",
    "split",
]

RISK_ENRICHED_TYPES = [
    "sanctions",
    "legal_regulatory",
    "fraud_phishing",
    "data_leak_security",
    "operational_issue",
    "customer_complaints",
    "other_risk",
]


def _round_robin_sample(
    df: pd.DataFrame,
    target_rows: int,
    seed: int,
    selected_global: set[int],
    entity_counts: dict[str, int],
    source_counts: dict[str, int],
    entity_cap: int,
    source_cap: int,
) -> tuple[list[int], list[str]]:
    if target_rows <= 0 or df.empty:
        return [], []

    warnings_list = []
    strata_columns = ["risk_type_candidate", "entity_norm", "source"]
    if {"published_year", "published_month"}.issubset(df.columns) and df[["published_year", "published_month"]].notna().any().any():
        strata_columns.extend(["published_year", "published_month"])

    shuffled = df.sample(frac=1, random_state=seed)
    buckets = [group.index.tolist() for _, group in shuffled.groupby(strata_columns, dropna=False, sort=False)]
    selected = []

    def can_take(index: int, use_caps: bool) -> bool:
        if index in selected_global:
            return False
        if not use_caps:
            return True
        entity = str(df.at[index, "entity_norm"])
        source = str(df.at[index, "source"])
        return entity_counts.get(entity, 0) < entity_cap and source_counts.get(source, 0) < source_cap

    def take(index: int) -> None:
        selected.append(index)
        selected_global.add(index)
        entity = str(df.at[index, "entity_norm"])
        source = str(df.at[index, "source"])
        entity_counts[entity] = entity_counts.get(entity, 0) + 1
        source_counts[source] = source_counts.get(source, 0) + 1

    while len(selected) < target_rows and any(buckets):
        progressed = False
        for bucket in buckets:
            while bucket:
                index = bucket.pop(0)
                if can_take(index, use_caps=True):
                    take(index)
                    progressed = True
                    break
            if len(selected) >= target_rows:
                break
        if not progressed:
            break

    if len(selected) < target_rows:
        warnings_list.append("Не удалось полностью соблюсти ограничения по entity/source, добираем строки без этих лимитов.")
        leftovers = shuffled.index.tolist()
        for index in leftovers:
            if len(selected) >= target_rows:
                break
            if can_take(index, use_caps=False):
                take(index)

    return selected, warnings_list


def _diverse_sample(df: pd.DataFrame, annotation_rows: int, seed: int) -> pd.DataFrame:
    if len(df) <= annotation_rows:
        warnings.warn("Данных меньше или равно annotation_rows, используется весь доступный пул.")
        return df.sample(frac=1, random_state=seed).reset_index(drop=True)

    target_rows = min(annotation_rows, len(df))
    no_risk_df = df[df["risk_type_candidate"] == "no_risk"]
    risk_df = df[df["risk_type_candidate"] != "no_risk"]
    target_no_risk = int(round(target_rows * 0.55))
    target_no_risk = min(target_no_risk, len(no_risk_df))
    target_risk = target_rows - target_no_risk
    if len(risk_df) < target_risk:
        warnings.warn("Risk candidates меньше целевой квоты 40-50%, добираем доступные no_risk.")
        target_risk = len(risk_df)
        target_no_risk = min(target_rows - target_risk, len(no_risk_df))
    if len(no_risk_df) < target_no_risk:
        warnings.warn("no_risk меньше целевой квоты 50-60%, добираем risk candidates.")
        target_no_risk = len(no_risk_df)
        target_risk = min(target_rows - target_no_risk, len(risk_df))

    entity_cap = max(1, math.ceil(target_rows * 0.25))
    source_cap = max(1, math.ceil(target_rows * 0.30))
    selected_global: set[int] = set()
    entity_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}

    risk_indices, risk_warnings = _round_robin_sample(
        risk_df, target_risk, seed, selected_global, entity_counts, source_counts, entity_cap, source_cap
    )
    no_risk_indices, no_risk_warnings = _round_robin_sample(
        no_risk_df, target_no_risk, seed + 1, selected_global, entity_counts, source_counts, entity_cap, source_cap
    )
    for warning_text in sorted(set(risk_warnings + no_risk_warnings)):
        warnings.warn(warning_text)

    selected_indices = risk_indices + no_risk_indices
    if len(selected_indices) < target_rows:
        warnings.warn("После квотного отбора строк меньше annotation_rows, добираем из остатка случайно.")
        leftovers = df.drop(index=selected_indices, errors="ignore").sample(frac=1, random_state=seed + 2)
        selected_indices.extend(leftovers.index.tolist()[: target_rows - len(selected_indices)])

    sampled = df.loc[selected_indices].sample(frac=1, random_state=seed + 3).reset_index(drop=True)

    max_entity_share = sampled["entity_norm"].value_counts(normalize=True).max() if not sampled.empty else 0
    max_source_share = sampled["source"].value_counts(normalize=True).max() if not sampled.empty else 0
    if max_entity_share > 0.25 and df["entity_norm"].nunique() > 1:
        warnings.warn("Не удалось удержать долю одного entity_norm в пределах 20-25%.")
    if max_source_share > 0.30 and df["source"].nunique() > 1:
        warnings.warn("Не удалось удержать долю одного source в пределах 30%.")

    return sampled


def _exclude_existing_rows(df: pd.DataFrame, exclude_existing: str | None) -> pd.DataFrame:
    if not exclude_existing:
        return df

    existing_path = Path(exclude_existing)
    if not existing_path.exists():
        warnings.warn(f"Файл exclude_existing не найден: {exclude_existing}. Исключение строк пропущено.")
        return df

    existing = pd.read_csv(existing_path)
    key_columns = ["url", "entity_norm", "text_fragment"]
    if not set(key_columns).issubset(existing.columns):
        warnings.warn("В exclude_existing нет ключевых колонок url/entity_norm/text_fragment. Исключение строк пропущено.")
        return df

    existing_keys = set(map(tuple, existing[key_columns].fillna("").astype(str).to_numpy()))
    current_keys = df[key_columns].fillna("").astype(str).apply(tuple, axis=1)
    filtered = df.loc[~current_keys.isin(existing_keys)].copy()
    removed = len(df) - len(filtered)
    if removed:
        print(f"Исключено уже существующих строк: {removed}")
    return filtered.reset_index(drop=True)


def _risk_enriched_sample(df: pd.DataFrame, annotation_rows: int, seed: int, exclude_existing: str | None) -> pd.DataFrame:
    df = _exclude_existing_rows(df, exclude_existing)
    risk_df = df[(df["risk_type_candidate"] != "no_risk") | (df["found_risk_keywords"].fillna("").astype(str).str.strip() != "")]
    risk_df = risk_df.copy()

    if risk_df.empty:
        warnings.warn("После фильтрации risk_enriched не осталось риск-кандидатов.")
        return risk_df
    if len(risk_df) <= annotation_rows:
        warnings.warn("Risk-enriched кандидатов меньше или равно annotation_rows, используется весь доступный пул.")
        return risk_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    target_rows = min(annotation_rows, len(risk_df))
    entity_cap = max(1, math.ceil(target_rows * 0.25))
    source_cap = max(1, math.ceil(target_rows * 0.30))
    selected_global: set[int] = set()
    entity_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    selected_indices: list[int] = []

    base_quota = max(1, math.ceil(target_rows / len(RISK_ENRICHED_TYPES)))
    for offset, risk_type in enumerate(RISK_ENRICHED_TYPES):
        bucket = risk_df[risk_df["risk_type_candidate"] == risk_type]
        if bucket.empty:
            warnings.warn(f"В пуле нет кандидатов для risk_type_candidate={risk_type}.")
            continue
        remaining = target_rows - len(selected_indices)
        if remaining <= 0:
            break
        quota = min(base_quota, remaining)
        indices, sample_warnings = _round_robin_sample(
            bucket,
            quota,
            seed + offset,
            selected_global,
            entity_counts,
            source_counts,
            entity_cap,
            source_cap,
        )
        selected_indices.extend(indices)
        for warning_text in sorted(set(sample_warnings)):
            warnings.warn(warning_text)

    if len(selected_indices) < target_rows:
        leftovers = risk_df.drop(index=selected_indices, errors="ignore")
        indices, sample_warnings = _round_robin_sample(
            leftovers,
            target_rows - len(selected_indices),
            seed + 100,
            selected_global,
            entity_counts,
            source_counts,
            entity_cap,
            source_cap,
        )
        selected_indices.extend(indices)
        for warning_text in sorted(set(sample_warnings)):
            warnings.warn(warning_text)

    if len(selected_indices) < target_rows:
        warnings.warn("Не удалось набрать annotation_rows после лимитов, добираем случайно из остатка.")
        leftovers = risk_df.drop(index=selected_indices, errors="ignore").sample(frac=1, random_state=seed + 200)
        selected_indices.extend(leftovers.index.tolist()[: target_rows - len(selected_indices)])

    sampled = risk_df.loc[selected_indices].sample(frac=1, random_state=seed + 300).reset_index(drop=True)
    if sampled["entity_norm"].value_counts(normalize=True).max() > 0.25 and risk_df["entity_norm"].nunique() > 1:
        warnings.warn("В risk_enriched не удалось удержать долю одного entity_norm в пределах 25%.")
    if sampled["source"].value_counts(normalize=True).max() > 0.30 and risk_df["source"].nunique() > 1:
        warnings.warn("В risk_enriched не удалось удержать долю одного source в пределах 30%.")

    return sampled


def _prepare_pool(input_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    for column in ["published_year", "published_month", "match_type", "match_confidence"]:
        if column not in df.columns:
            df[column] = ""
    df = df.drop_duplicates(subset=["url", "entity_norm", "text_fragment"]).reset_index(drop=True)

    full_text = (df["title"].fillna("") + " " + df["text_fragment"].fillna("")).astype(str)
    df["risk_type_candidate"] = full_text.map(detect_risk_type_candidate)
    df["found_risk_keywords"] = full_text.map(lambda text: "; ".join(find_risk_keywords(text)))
    df["alert_flag_candidate"] = (df["risk_type_candidate"] != "no_risk").astype(int)
    return df


def make_annotation_dataset(
    input_path: str,
    output: str,
    annotation_rows: int,
    seed: int,
    mode: str,
    exclude_existing: str | None,
) -> pd.DataFrame:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = _prepare_pool(input_path)
    if mode == "natural":
        sampled = _diverse_sample(df, annotation_rows, seed)
        dataset_part = "natural_sample"
    elif mode == "risk_enriched":
        sampled = _risk_enriched_sample(df, annotation_rows, seed, exclude_existing)
        dataset_part = "risk_enriched_sample"
    else:
        raise ValueError(f"Неизвестный mode: {mode}")

    sampled = sampled.reset_index(drop=True)
    prefix = "rn" if mode == "natural" else "re"
    sampled.insert(0, "sample_id", [f"{prefix}_{idx:06d}" for idx in range(1, len(sampled) + 1)])
    sampled.insert(1, "dataset_part", dataset_part)

    for column in ["risk_type", "entity_relevance", "alert_flag", "alert_reason", "label_quality", "risk_score_v1", "split"]:
        sampled[column] = ""

    sampled = sampled[ANNOTATION_COLUMNS]
    sampled.to_csv(output_path, index=False)
    return sampled


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/interim/bank_news_candidates_pool.csv")
    parser.add_argument("--output", default="data/processed/annotation_template.csv")
    parser.add_argument("--annotation_rows", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["natural", "risk_enriched"], default="natural")
    parser.add_argument("--exclude_existing", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = make_annotation_dataset(
        args.input,
        args.output,
        args.annotation_rows,
        args.seed,
        args.mode,
        args.exclude_existing,
    )
    print(f"Сохранено строк для разметки: {len(df)}")
    print(f"Файл: {args.output}")


if __name__ == "__main__":
    main()
