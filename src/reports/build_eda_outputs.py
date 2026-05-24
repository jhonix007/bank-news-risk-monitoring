"""Построение финальных EDA-артефактов для ДЗ №4."""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


STOP_WORDS = {
    "и",
    "в",
    "во",
    "на",
    "с",
    "со",
    "к",
    "ко",
    "по",
    "из",
    "за",
    "от",
    "до",
    "для",
    "что",
    "как",
    "это",
    "или",
    "но",
    "а",
    "о",
    "об",
    "у",
    "не",
    "он",
    "она",
    "они",
    "мы",
    "вы",
    "его",
    "ее",
    "их",
    "был",
    "была",
    "были",
    "будет",
    "также",
}


def _save_counts(df: pd.DataFrame, column: str, output_dir: Path, filename: str) -> pd.DataFrame:
    counts = df[column].fillna("").astype(str).value_counts(dropna=False).rename_axis(column).reset_index(name="count")
    counts.to_csv(output_dir / filename, index=False)
    return counts


def _save_bar(counts: pd.DataFrame, label_col: str, output_dir: Path, filename: str, top_n: int | None = None) -> None:
    plot_df = counts.head(top_n) if top_n else counts
    if plot_df.empty:
        return
    plt.figure(figsize=(10, 5))
    plot_df.set_index(label_col)["count"].sort_values().plot(kind="barh")
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=160)
    plt.close()


def _save_named_bar(
    counts: pd.DataFrame,
    label_col: str,
    output_dir: Path,
    filename: str,
    title: str,
    log_scale: bool = False,
) -> None:
    if counts.empty:
        return
    plt.figure(figsize=(10, 5))
    ax = counts.set_index(label_col)["count"].sort_values().plot(kind="barh", logx=log_scale)
    ax.set_title(title)
    ax.set_xlabel("Количество строк")
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=160)
    plt.close()


def _word_count(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.findall(r"[A-Za-zА-Яа-яЁё]+").str.len()


def _length_stats(df: pd.DataFrame, column: str) -> dict:
    chars = df[column].fillna("").astype(str).str.len()
    words = _word_count(df[column])
    return {
        "column": column,
        "char_mean": chars.mean(),
        "char_median": chars.median(),
        "char_min": chars.min(),
        "char_max": chars.max(),
        "char_p25": chars.quantile(0.25),
        "char_p75": chars.quantile(0.75),
        "word_mean": words.mean(),
        "word_median": words.median(),
        "word_min": words.min(),
        "word_max": words.max(),
        "word_p25": words.quantile(0.25),
        "word_p75": words.quantile(0.75),
    }


def _group_length_stats(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    work = df.copy()
    work["text_fragment_chars"] = work["text_fragment"].fillna("").astype(str).str.len()
    work["text_fragment_words"] = _word_count(work["text_fragment"])
    rows = []
    for value, group in work.groupby(group_col, dropna=False):
        chars = group["text_fragment_chars"]
        words = group["text_fragment_words"]
        rows.append(
            {
                group_col: value,
                "rows": len(group),
                "char_mean": chars.mean(),
                "char_median": chars.median(),
                "char_min": chars.min(),
                "char_max": chars.max(),
                "char_p25": chars.quantile(0.25),
                "char_p75": chars.quantile(0.75),
                "word_mean": words.mean(),
                "word_median": words.median(),
                "word_min": words.min(),
                "word_max": words.max(),
                "word_p25": words.quantile(0.25),
                "word_p75": words.quantile(0.75),
            }
        )
    return pd.DataFrame(rows).sort_values("rows", ascending=False)


def _parse_keywords(series: pd.Series) -> Counter:
    counter: Counter = Counter()
    for value in series.fillna("").astype(str):
        if not value.strip():
            continue
        parts = re.split(r"[;,|]", value)
        for part in parts:
            keyword = part.strip().lower()
            if keyword:
                counter[keyword] += 1
    return counter


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[A-Za-zА-Яа-яЁё]+", str(text).lower())
    return [token for token in tokens if len(token) >= 3 and token not in STOP_WORDS]


def _top_words(series: pd.Series, top_n: int = 30) -> pd.DataFrame:
    counter: Counter = Counter()
    for text in series.fillna("").astype(str):
        counter.update(_tokenize(text))
    return pd.DataFrame(counter.most_common(top_n), columns=["word", "count"])


def _save_word_plot(df: pd.DataFrame, output_dir: Path, filename: str, title: str) -> None:
    if df.empty:
        return
    plt.figure(figsize=(10, 6))
    ax = df.set_index("word")["count"].sort_values().plot(kind="barh")
    ax.set_title(title)
    ax.set_xlabel("Частота")
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=160)
    plt.close()


def build_eda_outputs(input_path: str, output_dir: str) -> pd.DataFrame:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_path, keep_default_na=False)

    duplicate_count = int(df.duplicated(subset=["url", "entity_norm", "text_fragment"]).sum())
    years = pd.to_numeric(df["published_year"], errors="coerce")
    alert = pd.to_numeric(df["alert_flag"], errors="coerce")
    summary = pd.DataFrame(
        [
            {"metric": "total_rows", "value": len(df)},
            {"metric": "total_sources", "value": df["source"].nunique()},
            {"metric": "total_banks", "value": df["entity_norm"].nunique()},
            {"metric": "total_urls", "value": df["url"].nunique()},
            {"metric": "duplicates_count", "value": duplicate_count},
            {"metric": "alert_positive_count", "value": int(alert.eq(1).sum())},
            {"metric": "alert_positive_share", "value": round(float(alert.eq(1).mean()), 4)},
            {"metric": "no_risk_count", "value": int(df["risk_type"].eq("no_risk").sum())},
            {"metric": "risk_count", "value": int(df["risk_type"].ne("no_risk").sum())},
            {"metric": "min_year", "value": int(years.min()) if years.notna().any() else ""},
            {"metric": "max_year", "value": int(years.max()) if years.notna().any() else ""},
        ]
    )
    summary.to_csv(out / "summary_counts.csv", index=False)

    distributions = {
        "dataset_part": "dataset_part_distribution.csv",
        "source": "source_distribution.csv",
        "entity_norm": "entity_distribution.csv",
        "risk_type": "risk_type_distribution.csv",
        "alert_flag": "alert_flag_distribution.csv",
        "entity_relevance": "entity_relevance_distribution.csv",
        "label_quality": "label_quality_distribution.csv",
        "split": "split_distribution.csv",
        "published_year": "published_year_distribution.csv",
    }
    saved_counts = {column: _save_counts(df, column, out, filename) for column, filename in distributions.items()}
    saved_counts["risk_type"].to_csv(out / "risk_type_distribution_full.csv", index=False)
    risk_without_no = (
        df[df["risk_type"] != "no_risk"]["risk_type"]
        .fillna("")
        .astype(str)
        .value_counts(dropna=False)
        .rename_axis("risk_type")
        .reset_index(name="count")
    )
    risk_without_no.to_csv(out / "risk_type_distribution_without_no_risk.csv", index=False)
    saved_counts["alert_flag"].to_csv(out / "alert_flag_distribution.csv", index=False)
    dataset_part_vs_alert = pd.crosstab(df["dataset_part"], df["alert_flag"]).reset_index()
    dataset_part_vs_alert.to_csv(out / "dataset_part_vs_alert_flag.csv", index=False)

    scores = pd.to_numeric(df["risk_score_v1"], errors="coerce").dropna()
    scores.describe().rename_axis("metric").reset_index(name="value").to_csv(out / "risk_score_distribution.csv", index=False)

    missing = (df.astype(str).apply(lambda col: col.str.strip().eq("")).sum()).rename_axis("column").reset_index(name="missing_count")
    missing.to_csv(out / "missing_values.csv", index=False)

    pd.DataFrame(
        [
            {"metric": "duplicate_full_rows", "value": int(df.duplicated().sum())},
            {"metric": "duplicate_url_entity_fragment_rows", "value": duplicate_count},
        ]
    ).to_csv(out / "duplicate_report.csv", index=False)

    text_stats = [_length_stats(df, column) for column in ["title", "text_fragment"]]
    pd.DataFrame(text_stats).to_csv(out / "text_length_stats.csv", index=False)
    by_risk_type = _group_length_stats(df, "risk_type")
    by_alert_flag = _group_length_stats(df, "alert_flag")
    by_dataset_part = _group_length_stats(df, "dataset_part")
    by_risk_type.to_csv(out / "text_length_by_risk_type.csv", index=False)
    by_alert_flag.to_csv(out / "text_length_by_alert_flag.csv", index=False)
    by_dataset_part.to_csv(out / "text_length_by_dataset_part.csv", index=False)

    keyword_counter = _parse_keywords(df["found_risk_keywords"]) if "found_risk_keywords" in df.columns else Counter()
    if keyword_counter:
        keyword_df = pd.DataFrame(keyword_counter.most_common(30), columns=["keyword", "count"])
    else:
        print("Предупреждение: found_risk_keywords пустой или не распарсился.")
        keyword_df = pd.DataFrame(columns=["keyword", "count"])
    keyword_df.to_csv(out / "risk_keyword_distribution.csv", index=False)

    top_words_all = _top_words(df["text_fragment"])
    top_words_alert_1 = _top_words(df[df["alert_flag"].astype(str) == "1"]["text_fragment"])
    top_words_alert_0 = _top_words(df[df["alert_flag"].astype(str) == "0"]["text_fragment"])
    top_words_risk_only = _top_words(df[df["risk_type"] != "no_risk"]["text_fragment"])
    top_words_all.to_csv(out / "top_words_all.csv", index=False)
    top_words_alert_1.to_csv(out / "top_words_alert_1.csv", index=False)
    top_words_alert_0.to_csv(out / "top_words_alert_0.csv", index=False)
    top_words_risk_only.to_csv(out / "top_words_risk_only.csv", index=False)

    _save_bar(saved_counts["dataset_part"], "dataset_part", out, "dataset_part_distribution.png")
    _save_bar(saved_counts["risk_type"], "risk_type", out, "risk_type_distribution.png")
    _save_named_bar(
        saved_counts["risk_type"],
        "risk_type",
        out,
        "risk_type_distribution_full.png",
        "Распределение risk_type: полный датасет",
    )
    _save_named_bar(
        risk_without_no,
        "risk_type",
        out,
        "risk_type_distribution_without_no_risk.png",
        "Распределение риск-классов без no_risk",
    )
    _save_named_bar(
        saved_counts["risk_type"],
        "risk_type",
        out,
        "risk_type_distribution_log_scale.png",
        "Распределение risk_type в логарифмической шкале",
        log_scale=True,
    )
    _save_named_bar(
        saved_counts["alert_flag"],
        "alert_flag",
        out,
        "alert_flag_distribution.png",
        "Бинарная целевая метка alert_flag",
    )
    _save_bar(saved_counts["entity_relevance"], "entity_relevance", out, "entity_relevance_distribution.png")
    _save_bar(saved_counts["entity_norm"], "entity_norm", out, "entity_distribution.png", top_n=20)
    _save_bar(saved_counts["published_year"], "published_year", out, "published_year_distribution.png")

    plt.figure(figsize=(8, 5))
    df["title"].fillna("").astype(str).str.len().hist(bins=30)
    plt.xlabel("Символы")
    plt.ylabel("Количество строк")
    plt.title("Длина title")
    plt.tight_layout()
    plt.savefig(out / "title_length_hist.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    df["text_fragment"].fillna("").astype(str).str.len().hist(bins=30)
    plt.xlabel("Символы")
    plt.ylabel("Количество строк")
    plt.title("Длина text_fragment")
    plt.tight_layout()
    plt.savefig(out / "text_fragment_length_hist.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    _word_count(df["text_fragment"]).hist(bins=30)
    plt.xlabel("Слова")
    plt.ylabel("Количество строк")
    plt.title("Количество слов в text_fragment")
    plt.tight_layout()
    plt.savefig(out / "text_fragment_word_count_hist.png", dpi=160)
    plt.close()

    for group_df, label_col, filename, title in [
        (by_risk_type, "risk_type", "text_length_by_risk_type.png", "Длина text_fragment по risk_type"),
        (by_alert_flag, "alert_flag", "text_length_by_alert_flag.png", "Длина text_fragment по alert_flag"),
    ]:
        if not group_df.empty:
            plt.figure(figsize=(10, 5))
            ax = group_df.set_index(label_col)["char_median"].sort_values().plot(kind="barh")
            ax.set_xlabel("Медианная длина, символы")
            ax.set_title(title)
            plt.tight_layout()
            plt.savefig(out / filename, dpi=160)
            plt.close()

    if not keyword_df.empty:
        plt.figure(figsize=(10, 6))
        ax = keyword_df.set_index("keyword")["count"].sort_values().plot(kind="barh")
        ax.set_title("Топ-30 риск-слов")
        ax.set_xlabel("Частота")
        plt.tight_layout()
        plt.savefig(out / "risk_keyword_distribution_top30.png", dpi=160)
        plt.close()

    _save_word_plot(top_words_all, out, "top_words_all.png", "Топ-30 слов: весь датасет")
    _save_word_plot(top_words_alert_1, out, "top_words_alert_1.png", "Топ-30 слов: alert_flag = 1")
    _save_word_plot(top_words_alert_0, out, "top_words_alert_0.png", "Топ-30 слов: alert_flag = 0")
    _save_word_plot(top_words_risk_only, out, "top_words_risk_only.png", "Топ-30 слов: risk_type != no_risk")

    plt.figure(figsize=(8, 5))
    scores.hist(bins=20)
    plt.xlabel("risk_score_v1")
    plt.ylabel("Количество строк")
    plt.tight_layout()
    plt.savefig(out / "risk_score_hist.png", dpi=160)
    plt.close()

    if not dataset_part_vs_alert.empty:
        plot_df = dataset_part_vs_alert.set_index("dataset_part")
        plt.figure(figsize=(10, 5))
        plot_df.plot(kind="bar", stacked=False, ax=plt.gca())
        plt.title("Распределение alert_flag по dataset_part")
        plt.xlabel("dataset_part")
        plt.ylabel("Количество строк")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(out / "dataset_part_vs_alert_flag.png", dpi=160)
        plt.close()

    print("EDA report")
    print(f"Строк: {len(df)}")
    print(f"Банков: {df['entity_norm'].nunique()}")
    print(f"Источников: {df['source'].nunique()}")
    print(f"Alert positive: {int(alert.eq(1).sum())} ({alert.eq(1).mean():.2%})")
    print(f"Дубли url+entity+fragment: {duplicate_count}")
    if years.notna().any():
        print(f"Годы: {int(years.min())}-{int(years.max())}")
    print(f"Файлы EDA: {out}")
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/news_risk_dataset_labeled.csv")
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", default="homework_04_dataset/eda_outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_eda_outputs(args.input, args.output_dir)


if __name__ == "__main__":
    main()
