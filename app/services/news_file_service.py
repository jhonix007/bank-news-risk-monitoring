from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
from fastapi import UploadFile
from sqlalchemy.orm import Session

from app.config import settings
from app.database.models import NewsBatch, NewsItem, User
from app.services.entity_matching_service import infer_entity_norm


REQUIRED_COLUMNS = ["title", "text_fragment"]
OPTIONAL_COLUMNS = ["source", "url", "published_at", "entity_norm"]


def read_and_validate_csv(content: bytes) -> pd.DataFrame:
    if not content:
        raise ValueError("Файл пустой.")
    try:
        dataframe = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise ValueError(f"Не удалось прочитать CSV-файл: {exc}") from exc
    if dataframe.empty:
        raise ValueError("CSV-файл не содержит строк с новостями.")
    missing = [column for column in REQUIRED_COLUMNS if column not in dataframe.columns]
    if missing:
        raise ValueError("В CSV отсутствуют обязательные колонки: " + ", ".join(missing) + ".")
    valid = dataframe.dropna(subset=REQUIRED_COLUMNS).copy()
    for column in REQUIRED_COLUMNS:
        valid = valid[valid[column].astype(str).str.strip() != ""]
    if valid.empty:
        raise ValueError("CSV-файл не содержит валидных строк с заполненными обязательными колонками.")
    return valid.reset_index(drop=True)


def save_upload_file(batch_id: int, filename: str, content: bytes) -> Path:
    settings.ensure_dirs()
    suffix = Path(filename or "news.csv").suffix or ".csv"
    stored_path = settings.upload_dir / f"batch_{batch_id}{suffix}"
    stored_path.write_bytes(content)
    return stored_path


def create_news_batch_from_upload(db: Session, user: User, upload: UploadFile, content: bytes) -> NewsBatch:
    if Path(upload.filename or "").suffix.lower() != ".csv":
        raise ValueError("Можно загрузить только CSV-файл.")
    dataframe = read_and_validate_csv(content)
    batch = NewsBatch(
        user_id=user.id,
        original_filename=upload.filename or "news.csv",
        stored_path="",
        status="uploaded",
        total_items=len(dataframe),
        processed_items=0,
    )
    db.add(batch)
    db.flush()
    stored_path = save_upload_file(batch.id, batch.original_filename, content)
    batch.stored_path = str(stored_path)
    for _, row in dataframe.fillna("").iterrows():
        title = str(row["title"])
        text_fragment = str(row["text_fragment"])
        provided_entity_norm = str(row.get("entity_norm", "")).strip()
        entity_norm = provided_entity_norm or infer_entity_norm(title, text_fragment) or "Не определено"
        db.add(
            NewsItem(
                batch_id=batch.id,
                user_id=user.id,
                source=str(row.get("source", "")) or None,
                url=str(row.get("url", "")) or None,
                published_at=str(row.get("published_at", "")) or None,
                title=title,
                text_fragment=text_fragment,
                entity_norm=entity_norm,
            )
        )
    batch.status = "queued"
    db.flush()
    return batch
