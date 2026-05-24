"""Черновая предразметка шаблона для последующей ручной проверки."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SUGGESTED_COLUMNS = [
    "risk_type_suggested",
    "entity_relevance_suggested",
    "alert_flag_suggested",
    "alert_reason_suggested",
    "label_source",
    "review_status",
    "review_comment",
]


def _clean(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _has_value(value) -> bool:
    return _clean(value) != ""


def _risk_type_suggested(row: pd.Series) -> str:
    candidate = _clean(row.get("risk_type_candidate"))
    if candidate and candidate != "no_risk":
        return candidate
    return "no_risk"


def _entity_relevance_suggested(row: pd.Series, risk_type: str) -> str:
    if risk_type == "no_risk":
        return "mentioned_only"

    text_fragment = _clean(row.get("text_fragment")).lower()
    entity_mention = _clean(row.get("entity_mention")).lower()
    found_keywords = _clean(row.get("found_risk_keywords"))

    if entity_mention and entity_mention in text_fragment:
        return "direct"
    if found_keywords:
        return "unclear"
    return "unclear"


def _alert_reason_suggested(row: pd.Series, risk_type: str, alert_flag: int) -> str:
    if alert_flag != 1:
        return ""

    entity_norm = _clean(row.get("entity_norm"))
    found_keywords = _clean(row.get("found_risk_keywords")) or "риск-слова не указаны"
    return f"В публикации найден риск типа {risk_type} для {entity_norm}: {found_keywords}."


def _fill_if_empty(df: pd.DataFrame, column: str, values) -> None:
    if column not in df.columns:
        df[column] = ""
    mask = ~df[column].map(_has_value)
    df.loc[mask, column] = pd.Series(values, index=df.index).loc[mask]


def assist_manual_annotation(input_path: str, output: str, prefill_final: bool = False) -> pd.DataFrame:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    risk_suggested = df.apply(_risk_type_suggested, axis=1)
    relevance_suggested = [
        _entity_relevance_suggested(row, risk_type)
        for (_, row), risk_type in zip(df.iterrows(), risk_suggested, strict=False)
    ]
    alert_suggested = [
        1 if risk_type != "no_risk" and relevance == "direct" else 0
        for risk_type, relevance in zip(risk_suggested, relevance_suggested, strict=False)
    ]
    reason_suggested = [
        _alert_reason_suggested(row, risk_type, alert_flag)
        for (_, row), risk_type, alert_flag in zip(df.iterrows(), risk_suggested, alert_suggested, strict=False)
    ]

    df["risk_type_suggested"] = risk_suggested
    df["entity_relevance_suggested"] = relevance_suggested
    df["alert_flag_suggested"] = alert_suggested
    df["alert_reason_suggested"] = reason_suggested
    df["label_source"] = "rule_assisted"
    df["review_status"] = "need_manual_review"
    df["review_comment"] = ""

    for column in ["risk_type", "entity_relevance", "alert_flag", "alert_reason", "label_quality"]:
        if column not in df.columns:
            df[column] = ""
        else:
            df[column] = df[column].where(df[column].map(_has_value), "")

    if prefill_final:
        _fill_if_empty(df, "risk_type", risk_suggested)
        _fill_if_empty(df, "entity_relevance", relevance_suggested)
        _fill_if_empty(df, "alert_flag", alert_suggested)
        _fill_if_empty(df, "alert_reason", reason_suggested)
        _fill_if_empty(df, "label_quality", ["ok"] * len(df))

    if "risk_score_v1" not in df.columns:
        df["risk_score_v1"] = ""
    else:
        df["risk_score_v1"] = ""

    df.to_csv(output_path, index=False)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/annotation_template.csv")
    parser.add_argument("--output", default="data/processed/annotation_template_assisted.csv")
    parser.add_argument("--prefill-final", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = assist_manual_annotation(args.input, args.output, args.prefill_final)
    print(f"Сохранено строк с предразметкой: {len(df)}")
    print(f"Файл: {args.output}")


if __name__ == "__main__":
    main()
