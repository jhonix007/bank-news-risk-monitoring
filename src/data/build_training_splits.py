"""Подготовка файлов для обучения baseline-моделей."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_training_splits(input_path: str, output_dir: str, negative_ratio: int, seed: int) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_path, keep_default_na=False)
    filtered = df[(df["review_status"] == "manual_checked") & (df["label_quality"] != "need_review")].copy()

    train = filtered[filtered["split"] == "train"].copy()
    valid = filtered[filtered["split"] == "valid"].copy()
    test = filtered[filtered["split"] == "test"].copy()

    train.to_csv(out / "news_risk_full_train.csv", index=False)
    valid.to_csv(out / "news_risk_full_valid.csv", index=False)
    test.to_csv(out / "news_risk_full_test.csv", index=False)

    positives = train[train["alert_flag"].astype(str) == "1"]
    negatives = train[train["alert_flag"].astype(str) == "0"]
    max_negatives = min(len(negatives), len(positives) * negative_ratio)
    sampled_negatives = negatives.sample(n=max_negatives, random_state=seed) if max_negatives > 0 else negatives.head(0)
    balanced = pd.concat([positives, sampled_negatives], ignore_index=True).sample(frac=1, random_state=seed)
    balanced.to_csv(out / "news_risk_train_balanced_alert.csv", index=False)

    print("Training splits report")
    print(f"Строк в исходном датасете: {len(df)}")
    print(f"Строк после фильтрации: {len(filtered)}")
    print("\nAlert distribution:")
    print(filtered["alert_flag"].value_counts().to_string())
    print("\nRisk type distribution:")
    print(filtered["risk_type"].value_counts().to_string())
    print(f"\nTrain/valid/test: {len(train)}/{len(valid)}/{len(test)}")
    print(f"Balanced train: {len(balanced)}")
    print("Balanced train alert_flag:")
    print(balanced["alert_flag"].value_counts().to_string())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/news_risk_dataset_labeled.csv")
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", default="data/processed/modeling")
    parser.add_argument("--negative-ratio", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_training_splits(args.input, args.output_dir, args.negative_ratio, args.seed)


if __name__ == "__main__":
    main()
