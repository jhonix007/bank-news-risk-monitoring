"""Объединение и очистка размеченных natural/risk-enriched выборок."""

from __future__ import annotations

import argparse
import os
import shutil
import warnings
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from src.features.scoring import calculate_risk_score
except ImportError:  # pragma: no cover
    calculate_risk_score = None


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
MANUAL_COLUMNS = ["risk_type", "entity_relevance", "alert_flag", "alert_reason", "label_quality", "review_status", "review_comment"]

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


def _clean(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _has_value(value) -> bool:
    return _clean(value) != ""


def _fallback_score(row) -> int:
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
    if risk_type not in risk_type_weights or entity_relevance not in entity_relevance_weights:
        return 0
    return min(100, risk_type_weights[risk_type] + entity_relevance_weights[entity_relevance] + alert_flag_weights.get(alert_flag, 0))


def _read_part(path: str, dataset_part: str) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=False)
    if "dataset_part" not in df.columns:
        df["dataset_part"] = dataset_part
    else:
        df["dataset_part"] = df["dataset_part"].where(df["dataset_part"].map(_has_value), dataset_part)
    return df


def _overlay_assisted_labels(df: pd.DataFrame, original_path: str) -> pd.DataFrame:
    path = Path(original_path)
    assisted_path = path.with_name(f"{path.stem}_assisted{path.suffix}")
    if not assisted_path.exists() or not set(KEY_COLUMNS).issubset(df.columns):
        return df

    needs_labels = any(column not in df.columns or not df[column].map(_has_value).all() for column in MANUAL_COLUMNS[:5])
    if not needs_labels:
        return df

    assisted = pd.read_csv(assisted_path, keep_default_na=False)
    if not set(KEY_COLUMNS + MANUAL_COLUMNS).issubset(assisted.columns):
        return df

    print(f"Найдены проверенные поля в assisted-файле, подтягиваю разметку: {assisted_path}")
    assisted_labels = assisted[KEY_COLUMNS + MANUAL_COLUMNS].drop_duplicates(subset=KEY_COLUMNS)
    merged = df.merge(assisted_labels, on=KEY_COLUMNS, how="left", suffixes=("", "_assisted"))
    for column in MANUAL_COLUMNS:
        assisted_column = f"{column}_assisted"
        if column not in merged.columns:
            merged[column] = ""
        if assisted_column in merged.columns:
            mask = ~merged[column].map(_has_value) & merged[assisted_column].map(_has_value)
            merged.loc[mask, column] = merged.loc[mask, assisted_column]
            merged = merged.drop(columns=[assisted_column])
    return merged


def _align_columns(left: pd.DataFrame, right: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_columns = list(dict.fromkeys(list(left.columns) + list(right.columns)))
    for column in all_columns:
        if column not in left.columns:
            left[column] = ""
        if column not in right.columns:
            right[column] = ""
    return left[all_columns], right[all_columns]


def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    raw = df["published_at"].map(_clean) if "published_at" in df.columns else pd.Series("", index=df.index)
    numeric = pd.to_numeric(raw, errors="coerce")

    parsed_str = pd.to_datetime(raw, errors="coerce", utc=True)
    parsed_s = pd.to_datetime(numeric, unit="s", errors="coerce", utc=True)
    parsed_ms = pd.to_datetime(numeric, unit="ms", errors="coerce", utc=True)

    candidates = [parsed_str, parsed_s, parsed_ms]
    best = max(candidates, key=lambda series: (series.notna().sum(), -(series.dt.year.eq(1970).sum())))
    if best.dt.year.eq(1970).mean() > 0.8:
        alternatives = [series for series in candidates if series.notna().sum() > 0]
        best = min(alternatives, key=lambda series: series.dt.year.eq(1970).mean()) if alternatives else best

    df["published_year"] = best.dt.year.astype("Int64").astype(str).replace("<NA>", "")
    df["published_month"] = best.dt.month.astype("Int64").astype(str).replace("<NA>", "")

    share_1970 = df["published_year"].eq("1970").mean() if len(df) else 0
    if share_1970 > 0.8:
        warnings.warn("После исправления более 80% строк имеют published_year=1970.")
    years = pd.to_numeric(df["published_year"], errors="coerce").dropna()
    if not years.empty:
        print(f"Даты исправлены: min year={int(years.min())}, max year={int(years.max())}")
    return df


def _validate(df: pd.DataFrame, issues_path: Path) -> pd.DataFrame:
    issues = []
    checks = {
        "risk_type": VALID_RISK_TYPES,
        "entity_relevance": VALID_ENTITY_RELEVANCE,
        "alert_flag": VALID_ALERT_FLAGS,
        "label_quality": VALID_LABEL_QUALITY,
    }
    for idx, row in df.iterrows():
        row_issues = []
        for column, valid_values in checks.items():
            value = _clean(row.get(column))
            if value == "" or value not in valid_values:
                row_issues.append(f"{column}={value or '<empty>'}")
        if row_issues:
            issue_row = row.to_dict()
            issue_row["validation_issues"] = "; ".join(row_issues)
            issues.append(issue_row)

    issues_df = pd.DataFrame(issues)
    if not issues_df.empty:
        issues_df.to_csv(issues_path, index=False)
        print(f"Найдены строки с проблемами валидации: {len(issues_df)}")
        print(f"Файл проблемных строк: {issues_path}")
    elif issues_path.exists():
        issues_path.unlink()
    return issues_df


def _fill_scores(df: pd.DataFrame) -> pd.DataFrame:
    if "risk_score_v1" not in df.columns:
        df["risk_score_v1"] = ""
    score_func = calculate_risk_score or _fallback_score
    missing_score = ~df["risk_score_v1"].map(_has_value)
    df.loc[missing_score, "risk_score_v1"] = df.loc[missing_score].apply(score_func, axis=1)
    return df


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
    if "split" in df.columns and df["split"].map(_has_value).all():
        return df
    if len(df) < 2:
        df["split"] = "train"
        return df

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


def _archive_files(natural: str, risk_enriched: str, output: str, archive_dir: str) -> None:
    archive_path = Path(archive_dir)
    archive_path.mkdir(parents=True, exist_ok=True)
    output_path = Path(output).resolve()
    issues_path = output_path.with_name("rows_with_validation_issues.csv").resolve()

    for source in [Path(natural), Path(risk_enriched)]:
        if source.exists():
            shutil.copy2(source, archive_path / source.name)

    processed_dir = output_path.parent
    patterns = ["*_backup.csv", "*_assisted.csv", "*_before_*.csv", "annotation_template.before_manual_fill.csv", "annotation_template_combined.csv", "annotation_template_500_current_backup.csv"]
    candidates: set[Path] = set()
    for pattern in patterns:
        candidates.update(processed_dir.glob(pattern))

    for path in sorted(candidates):
        resolved = path.resolve()
        if not path.exists() or resolved in {output_path, issues_path} or archive_path in path.parents:
            continue
        target = archive_path / path.name
        if target.exists():
            target = archive_path / f"{path.stem}.archived{path.suffix}"
        try:
            shutil.move(str(path), str(target))
        except PermissionError:
            shutil.copy2(path, target)
            try:
                os.chmod(path, 0o666)
                path.unlink()
            except PermissionError:
                warnings.warn(f"Файл скопирован в archive, но не перенесен из-за блокировки Windows: {path}")


def merge_and_clean(natural: str, risk_enriched: str, output: str, archive_dir: str) -> pd.DataFrame:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    issues_path = output_path.with_name("rows_with_validation_issues.csv")

    natural_df = _read_part(natural, "natural_sample")
    risk_df = _read_part(risk_enriched, "risk_enriched_sample")
    risk_df = _overlay_assisted_labels(risk_df, risk_enriched)
    natural_rows = len(natural_df)
    risk_rows = len(risk_df)

    natural_df, risk_df = _align_columns(natural_df, risk_df)
    combined = pd.concat([natural_df, risk_df], ignore_index=True)
    before_dedup = len(combined)
    combined = combined.drop_duplicates(subset=KEY_COLUMNS, keep="first").reset_index(drop=True)
    duplicates_removed = before_dedup - len(combined)

    for column in FINAL_COLUMNS:
        if column not in combined.columns:
            combined[column] = ""

    combined = _parse_dates(combined)
    combined["alert_flag"] = combined["alert_flag"].map(lambda value: _clean(value).replace(".0", ""))
    _validate(combined, issues_path)
    combined = _fill_scores(combined)
    combined = _assign_splits(combined)
    combined["sample_id"] = [f"sample_{idx:06d}" for idx in range(1, len(combined) + 1)]
    combined = combined[FINAL_COLUMNS]
    combined.to_csv(output_path, index=False)

    _archive_files(natural, risk_enriched, output, archive_dir)
    _print_report(combined, natural_rows, risk_rows, duplicates_removed, output_path)
    return combined


def _print_report(df: pd.DataFrame, natural_rows: int, risk_rows: int, duplicates_removed: int, output_path: Path) -> None:
    years = pd.to_numeric(df["published_year"], errors="coerce")
    print("\nОтчет merge_and_clean_labeled_dataset")
    print(f"Строк в natural_sample: {natural_rows}")
    print(f"Строк в risk_enriched_sample: {risk_rows}")
    print(f"Дублей удалено: {duplicates_removed}")
    print(f"Строк в финальном датасете: {len(df)}")
    print("\nРаспределение dataset_part:")
    print(df["dataset_part"].value_counts(dropna=False).to_string())
    print("\nРаспределение risk_type:")
    print(df["risk_type"].value_counts(dropna=False).to_string())
    print("\nРаспределение alert_flag:")
    print(df["alert_flag"].value_counts(dropna=False).to_string())
    print("\nРаспределение entity_relevance:")
    print(df["entity_relevance"].value_counts(dropna=False).to_string())
    print("\nРаспределение label_quality:")
    print(df["label_quality"].value_counts(dropna=False).to_string())
    if years.notna().any():
        print(f"\nmin/max published_year: {int(years.min())}/{int(years.max())}")
    print(f"Количество строк с published_year = 1970: {int(df['published_year'].eq('1970').sum())}")
    print(f"Путь к финальному файлу: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--natural", default="data/processed/annotation_template.csv")
    parser.add_argument("--risk-enriched", default="data/processed/annotation_template_risk_enriched.csv")
    parser.add_argument("--output", default="data/processed/news_risk_dataset_labeled.csv")
    parser.add_argument("--archive-dir", default="data/processed/archive")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merge_and_clean(args.natural, args.risk_enriched, args.output, args.archive_dir)


if __name__ == "__main__":
    main()
