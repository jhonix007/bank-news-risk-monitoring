"""Targeted-risk добор редких риск-классов напрямую из RuNews."""

from __future__ import annotations

import argparse
import math
import re
import time
import warnings
from pathlib import Path

import pandas as pd
from datasets import load_dataset

from src.features.bank_aliases import find_bank_mentions
from src.utils.text_utils import extract_fragment_around_mention, normalize_text


TARGETED_KEYWORDS = {
    "data_leak_security": [
        "утечка",
        "утечка данных",
        "персональные данные",
        "данные клиентов",
        "база клиентов",
        "клиентская база",
        "взлом",
        "взломали",
        "кибератака",
        "хакеры",
        "хакерская атака",
        "информационная безопасность",
        "скомпрометированы данные",
        "украли данные",
        "слив данных",
    ],
    "operational_issue": [
        "сбой",
        "технический сбой",
        "массовый сбой",
        "не работает",
        "не работал",
        "недоступно",
        "недоступен",
        "недоступна",
        "недоступно приложение",
        "не открывается приложение",
        "проблемы с приложением",
        "проблемы с переводами",
        "не проходят платежи",
        "не проходят переводы",
        "ошибка при переводе",
        "зависли платежи",
        "перебой",
        "перебои в работе",
        "приложение банка не работает",
    ],
    "fraud_phishing": [
        "мошенники",
        "мошенничество",
        "фишинг",
        "фишинговый",
        "фишинговая атака",
        "злоумышленники",
        "хищение",
        "похитили деньги",
        "украли деньги",
        "обманули клиентов",
        "телефонные мошенники",
        "банковские мошенники",
        "поддельный сайт",
        "поддельное приложение",
        "от имени банка",
        "представлялись сотрудниками банка",
    ],
    "customer_complaints": [
        "жалобы клиентов",
        "клиенты жалуются",
        "пожаловались клиенты",
        "массовые жалобы",
        "пожаловались на банк",
        "недовольны клиенты",
        "возмущены клиенты",
        "клиенты не могут",
        "клиенты банка не могут",
        "проблемы у клиентов",
        "обращения клиентов",
        "претензии клиентов",
        "жалобы на обслуживание",
        "жалобы на списания",
        "незаконные списания",
        "необоснованные списания",
    ],
}

OUTPUT_COLUMNS = [
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


KEYWORD_PATTERNS = {
    risk_type: [(keyword, _keyword_pattern(keyword)) for keyword in keywords]
    for risk_type, keywords in TARGETED_KEYWORDS.items()
}


def _find_targeted_keywords(text: str) -> tuple[str, list[str]]:
    found_by_type: dict[str, list[str]] = {}
    for risk_type, patterns in KEYWORD_PATTERNS.items():
        found = [keyword for keyword, pattern in patterns if pattern.search(text)]
        if found:
            found_by_type[risk_type] = found
    for risk_type in TARGETED_KEYWORDS:
        if risk_type in found_by_type:
            return risk_type, found_by_type[risk_type]
    return "", []


def _parse_year_month(timestamp) -> tuple[object, object]:
    raw = "" if pd.isna(timestamp) else str(timestamp).strip()
    numeric = pd.to_numeric(pd.Series([raw]), errors="coerce").iloc[0]
    candidates = [
        pd.to_datetime(pd.Series([raw]), errors="coerce", utc=True).iloc[0],
        pd.to_datetime(pd.Series([numeric]), unit="s", errors="coerce", utc=True).iloc[0],
        pd.to_datetime(pd.Series([numeric]), unit="ms", errors="coerce", utc=True).iloc[0],
    ]
    valid = [candidate for candidate in candidates if not pd.isna(candidate)]
    if not valid:
        return "", ""
    best = min(valid, key=lambda candidate: candidate.year == 1970)
    return int(best.year), int(best.month)


def _existing_keys(path: str) -> set[tuple[str, str, str]]:
    if not Path(path).exists():
        return set()
    df = pd.read_csv(path, keep_default_na=False)
    if not set(KEY_COLUMNS).issubset(df.columns):
        return set()
    return set(map(tuple, df[KEY_COLUMNS].fillna("").astype(str).to_numpy()))


def _load_stream(seed: int):
    dataset = load_dataset("IlyaGusev/ru_news", split="train", streaming=True, trust_remote_code=True)
    try:
        return dataset.shuffle(buffer_size=10000, seed=seed)
    except (AttributeError, NotImplementedError, TypeError) as exc:
        print(f"Предупреждение: streaming shuffle недоступен ({exc}). Читаем поток последовательно.")
        return dataset


def _can_take(row: dict, rows: list[dict], max_rows: int, use_caps: bool) -> bool:
    if not use_caps:
        return True
    entity_cap = max(1, math.ceil(max_rows * 0.25))
    source_cap = max(1, math.ceil(max_rows * 0.30))
    entity_count = sum(item["entity_norm"] == row["entity_norm"] for item in rows)
    source_count = sum(item["source"] == row["source"] for item in rows)
    return entity_count < entity_cap and source_count < source_cap


def build_targeted_from_runews(max_rows: int, max_scanned_rows: int, existing: str, output: str, seed: int) -> pd.DataFrame:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing_keys = _existing_keys(existing)
    seen_keys = set(existing_keys)
    rows: list[dict] = []
    found_counts = {risk_type: 0 for risk_type in TARGETED_KEYWORDS}
    excluded_existing = 0
    per_class_quota = max(1, math.ceil(max_rows / len(TARGETED_KEYWORDS)))
    class_counts = {risk_type: 0 for risk_type in TARGETED_KEYWORDS}
    scanned = 0
    start = time.monotonic()

    for item in _load_stream(seed):
        scanned += 1
        title = normalize_text(item.get("title", ""))
        text = normalize_text(item.get("text", ""))
        searchable_text = normalize_text(f"{title}\n{text}")
        risk_type, keywords = _find_targeted_keywords(searchable_text)
        if not risk_type:
            if scanned >= max_scanned_rows:
                break
            continue

        found_counts[risk_type] += 1
        mentions = find_bank_mentions(searchable_text)
        if not mentions:
            if scanned >= max_scanned_rows:
                break
            continue

        title_offset = len(title) + 1
        published_year, published_month = _parse_year_month(item.get("timestamp", ""))
        for mention in mentions:
            mention_start_in_text = max(0, mention["start"] - title_offset)
            text_fragment = extract_fragment_around_mention(title, text, mention_start_in_text)
            key = (str(item.get("url", "")), mention["entity_norm"], text_fragment)
            if key in seen_keys:
                excluded_existing += 1
                continue
            candidate = {
                "source": item.get("source", ""),
                "url": item.get("url", ""),
                "published_at": item.get("timestamp", ""),
                "published_year": published_year,
                "published_month": published_month,
                "title": title,
                "text_fragment": text_fragment,
                "entity_mention": mention["entity_mention"],
                "entity_norm": mention["entity_norm"],
                "risk_type_candidate": risk_type,
                "found_risk_keywords": "; ".join(keywords),
                "found_targeted_keywords": "; ".join(keywords),
            }
            if class_counts[risk_type] >= per_class_quota and len(rows) < max_rows:
                continue
            if not _can_take(candidate, rows, max_rows, use_caps=True):
                continue
            rows.append(candidate)
            seen_keys.add(key)
            class_counts[risk_type] += 1
            if len(rows) >= max_rows:
                break

        if scanned % 10000 == 0:
            elapsed = max(time.monotonic() - start, 0.001)
            print(f"Прогресс: просмотрено={scanned}, сохранено={len(rows)}, исключено_existing={excluded_existing}, скорость={scanned / elapsed:.1f} новостей/сек")

        if len(rows) >= max_rows or scanned >= max_scanned_rows:
            break

    if len(rows) < max_rows:
        for risk_type, count in class_counts.items():
            if count < per_class_quota:
                warnings.warn(f"Для {risk_type} сохранено меньше целевой квоты {per_class_quota}: {count}.")

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df.insert(0, "sample_id", [f"trr_{idx:06d}" for idx in range(1, len(df) + 1)])
    df.insert(1, "dataset_part", "targeted_risk_from_runews")
    for column in ["risk_type", "entity_relevance", "alert_flag", "alert_reason", "label_quality", "risk_score_v1", "split", "review_status", "review_comment"]:
        df[column] = ""
    for column in OUTPUT_COLUMNS:
        if column not in df.columns:
            df[column] = ""
    df = df[OUTPUT_COLUMNS]
    df.to_csv(output_path, index=False)
    _print_report(df, found_counts, excluded_existing, output_path)
    return df


def _print_report(df: pd.DataFrame, found_counts: dict[str, int], excluded_existing: int, output_path: Path) -> None:
    years = pd.to_numeric(df["published_year"], errors="coerce")
    print("\nTargeted-risk-from-RuNews report")
    print("Найдено строк по risk_type_candidate:")
    for risk_type in TARGETED_KEYWORDS:
        print(f"{risk_type}: {found_counts.get(risk_type, 0)}")
    print(f"\nСтрок вошло в {output_path}: {len(df)}")
    print("\nРаспределение entity_norm:")
    print(df["entity_norm"].value_counts().to_string() if not df.empty else "")
    print("\nРаспределение source:")
    print(df["source"].value_counts().to_string() if not df.empty else "")
    if years.notna().any():
        print(f"\nmin/max published_year: {int(years.min())}/{int(years.max())}")
    print(f"Исключено как уже существующие: {excluded_existing}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_rows", type=int, default=200)
    parser.add_argument("--max_scanned_rows", type=int, default=1000000)
    parser.add_argument("--existing", default="data/processed/news_risk_dataset_labeled.csv")
    parser.add_argument("--output", default="data/processed/annotation_template_targeted_from_runews.csv")
    parser.add_argument("--seed", type=int, default=45)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_targeted_from_runews(args.max_rows, args.max_scanned_rows, args.existing, args.output, args.seed)


if __name__ == "__main__":
    main()
