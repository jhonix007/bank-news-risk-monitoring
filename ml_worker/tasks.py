from __future__ import annotations

from app.database.session import SessionLocal

try:
    from ml_worker.processing import process_batch_sync
except ModuleNotFoundError:
    from processing import process_batch_sync


def process_batch_task(batch_id: int) -> None:
    db = SessionLocal()
    try:
        process_batch_sync(db, batch_id)
    finally:
        db.close()
