"""Сбор кандидатов банковских новостей из потокового RuNews."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from datasets import load_dataset

from src.features.bank_aliases import find_bank_mentions, find_rejected_bank_mentions
from src.utils.text_utils import extract_fragment_around_mention, normalize_text


ALLOWED_MATCH_CONFIDENCE = {"high", "medium"}
CANDIDATE_COLUMNS = [
    "scanned_row_number",
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
]
REJECTED_COLUMNS = ["source", "url", "title", "entity_norm", "entity_mention", "reason"]


def _extract_year_month(timestamp) -> tuple[object, object]:
    parsed = pd.to_datetime(timestamp, errors="coerce", utc=True)
    if pd.isna(parsed):
        return "", ""
    return int(parsed.year), int(parsed.month)


def _load_stream(seed: int):
    dataset = load_dataset(
        "IlyaGusev/ru_news",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )
    try:
        return dataset.shuffle(buffer_size=10000, seed=seed)
    except (AttributeError, NotImplementedError, TypeError) as exc:
        print(f"Предупреждение: shuffle для streaming dataset недоступен ({exc}). Читаем поток последовательно.")
        return dataset


def build_candidates(
    max_candidates: int,
    max_scanned_rows: int,
    seed: int,
    output: str,
    rejected_output: str = "data/interim/rejected_bank_mentions.csv",
) -> pd.DataFrame:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rejected_output_path = Path(rejected_output)
    rejected_output_path.parent.mkdir(parents=True, exist_ok=True)

    stream = _load_stream(seed)
    rows = []
    rejected_rows = []
    seen_keys = set()
    start_time = time.monotonic()
    scanned_rows = 0

    for item in stream:
        scanned_rows += 1
        title = normalize_text(item.get("title", ""))
        text = normalize_text(item.get("text", ""))
        searchable_text = normalize_text(f"{title}\n{text}")
        mentions = find_bank_mentions(searchable_text)
        rejected_mentions = find_rejected_bank_mentions(searchable_text)
        for rejected in rejected_mentions:
            rejected_rows.append(
                {
                    "source": item.get("source", ""),
                    "url": item.get("url", ""),
                    "title": title,
                    "entity_norm": rejected["entity_norm"],
                    "entity_mention": rejected["entity_mention"],
                    "reason": rejected["reason"],
                }
            )

        if not mentions:
            continue

        title_offset = len(title) + 1
        seen_entities = set()
        for mention in mentions:
            if mention["match_confidence"] not in ALLOWED_MATCH_CONFIDENCE:
                continue

            entity_norm = mention["entity_norm"]
            if entity_norm in seen_entities:
                continue
            seen_entities.add(entity_norm)

            mention_start_in_text = mention["start"] - title_offset
            if mention_start_in_text < 0:
                mention_start_in_text = 0

            text_fragment = extract_fragment_around_mention(title, text, mention_start_in_text)
            row_key = (item.get("url", ""), entity_norm, text_fragment)
            if row_key in seen_keys:
                continue
            seen_keys.add(row_key)
            published_year, published_month = _extract_year_month(item.get("timestamp", ""))

            rows.append(
                {
                    "scanned_row_number": scanned_rows,
                    "source": item.get("source", ""),
                    "url": item.get("url", ""),
                    "published_at": item.get("timestamp", ""),
                    "published_year": published_year,
                    "published_month": published_month,
                    "title": title,
                    "text_fragment": text_fragment,
                    "entity_mention": mention["entity_mention"],
                    "entity_norm": entity_norm,
                    "found_alias": mention["alias"],
                    "match_type": mention["match_type"],
                    "match_confidence": mention["match_confidence"],
                }
            )

            if len(rows) >= max_candidates:
                break

        if scanned_rows % 10000 == 0:
            elapsed = max(time.monotonic() - start_time, 0.001)
            speed = scanned_rows / elapsed
            print(
                "Прогресс: "
                f"просмотрено новостей={scanned_rows}, "
                f"найдено кандидатов={len(rows)}, "
                f"отклонено неоднозначных алиасов={len(rejected_rows)}, "
                f"скорость={speed:.1f} новостей/сек"
            )

        if len(rows) >= max_candidates or scanned_rows >= max_scanned_rows:
            break

    df = pd.DataFrame(rows, columns=CANDIDATE_COLUMNS)
    if not df.empty:
        df = df.drop_duplicates(subset=["url", "entity_norm", "text_fragment"]).reset_index(drop=True)
    df.to_csv(output_path, index=False)
    rejected_df = pd.DataFrame(rejected_rows, columns=REJECTED_COLUMNS)
    if not rejected_df.empty:
        rejected_df = rejected_df.drop_duplicates().reset_index(drop=True)
    rejected_df.to_csv(rejected_output_path, index=False)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_rows", type=int, default=None, help="Устаревший alias для --max_candidates.")
    parser.add_argument("--max_candidates", type=int, default=5000)
    parser.add_argument("--max_scanned_rows", type=int, default=300000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/interim/bank_news_candidates_pool.csv")
    parser.add_argument("--rejected_output", default="data/interim/rejected_bank_mentions.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_candidates = args.max_rows if args.max_rows is not None else args.max_candidates
    df = build_candidates(
        max_candidates=max_candidates,
        max_scanned_rows=args.max_scanned_rows,
        seed=args.seed,
        output=args.output,
        rejected_output=args.rejected_output,
    )
    print(f"Сохранено строк: {len(df)}")
    print(f"Файл: {args.output}")
    print(f"Отклоненные ambiguous совпадения: {args.rejected_output}")


if __name__ == "__main__":
    main()
