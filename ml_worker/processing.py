from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.database.models import NewsBatch, NewsItem, PredictionResult
from app.services.prediction_service import predict_news_items


logger = logging.getLogger(__name__)
PROCESSING_CHUNK_SIZE = max(1, settings.inference_batch_size)


def process_batch_sync(db: Session, batch_id: int) -> NewsBatch:
    started_at = time.perf_counter()
    batch = db.get(NewsBatch, batch_id)
    if batch is None:
        raise ValueError(f"Batch {batch_id} не найден.")
    try:
        logger.info("Starting batch %s processing.", batch_id)
        batch.status = "processing"
        batch.error_message = None
        batch.processed_items = 0
        db.query(PredictionResult).filter(PredictionResult.batch_id == batch_id).delete(synchronize_session=False)
        db.commit()

        load_started_at = time.perf_counter()
        items = list(db.scalars(select(NewsItem).where(NewsItem.batch_id == batch_id).order_by(NewsItem.id)))
        total_items = len(items)
        logger.info(
            "Loaded %s news items for batch %s in %.2fs.",
            total_items,
            batch_id,
            time.perf_counter() - load_started_at,
        )
        for index in range(0, len(items), PROCESSING_CHUNK_SIZE):
            chunk_started_at = time.perf_counter()
            chunk = items[index : index + PROCESSING_CHUNK_SIZE]
            predict_news_items(db, chunk)
            batch.processed_items += len(chunk)
            db.commit()
            logger.info(
                "Processed batch %s chunk: %s/%s items in %.2fs.",
                batch_id,
                batch.processed_items,
                total_items,
                time.perf_counter() - chunk_started_at,
            )

        batch.status = "completed"
        batch.processed_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(batch)
        logger.info(
            "Completed batch %s: %s items in %.2fs.",
            batch_id,
            total_items,
            time.perf_counter() - started_at,
        )
        return batch
    except Exception as exc:
        batch.status = "failed"
        batch.error_message = str(exc)
        batch.processed_at = datetime.now(timezone.utc)
        db.commit()
        logger.exception("Failed batch %s after %.2fs.", batch_id, time.perf_counter() - started_at)
        raise
