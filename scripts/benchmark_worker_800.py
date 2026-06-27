from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from uuid import uuid4

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.database.base import Base
from app.database.models import NewsBatch, NewsItem, PredictionResult, User
from ml_worker.processing import process_batch_sync


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        user = User(email=f"benchmark-{uuid4().hex}@example.com", hashed_password="benchmark")
        db.add(user)
        db.flush()

        batch = NewsBatch(
            user_id=user.id,
            original_filename="benchmark_800.csv",
            stored_path="storage/uploads/benchmark_800.csv",
            status="queued",
            total_items=800,
            processed_items=0,
        )
        db.add(batch)
        db.flush()

        db.add_all(
            [
                NewsItem(
                    batch_id=batch.id,
                    user_id=user.id,
                    title=f"Новость {index} о ВТБ",
                    text_fragment="Клиенты сообщили о мошенничестве, проверке операций и спорных списаниях.",
                    entity_norm="ВТБ",
                )
                for index in range(800)
            ]
        )
        db.commit()

        started_at = time.perf_counter()
        processed_batch = process_batch_sync(db, batch.id)
        elapsed = time.perf_counter() - started_at

        prediction_count = db.scalar(
            select(func.count(PredictionResult.id)).where(PredictionResult.batch_id == batch.id)
        )
        assert processed_batch.status == "completed"
        assert processed_batch.processed_items == 800
        assert prediction_count == 800

        print(
            f"Processed batch_id={batch.id}: "
            f"status={processed_batch.status}, "
            f"processed_items={processed_batch.processed_items}, "
            f"prediction_results={prediction_count}, "
            f"elapsed={elapsed:.2f}s"
        )
    finally:
        db.close()


if __name__ == "__main__":
    main()
