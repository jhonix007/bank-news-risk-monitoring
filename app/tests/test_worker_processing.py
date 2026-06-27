from __future__ import annotations

from unittest.mock import patch

from sqlalchemy import func, select

from app.database.models import PredictionResult
from ml_worker.processing import process_batch_sync


def test_worker_process_batch_success(db_session, uploaded_batch):
    batch = process_batch_sync(db_session, uploaded_batch.id)

    assert batch.status == "completed"


def test_worker_creates_prediction_results_for_all_items(db_session, uploaded_batch):
    process_batch_sync(db_session, uploaded_batch.id)

    prediction_count = db_session.scalar(select(func.count(PredictionResult.id)).where(PredictionResult.batch_id == uploaded_batch.id))
    assert prediction_count == uploaded_batch.total_items


def test_worker_updates_processed_items(db_session, uploaded_batch):
    batch = process_batch_sync(db_session, uploaded_batch.id)

    assert batch.processed_items == batch.total_items


def test_worker_sets_batch_completed(db_session, uploaded_batch):
    batch = process_batch_sync(db_session, uploaded_batch.id)

    assert batch.status == "completed"
    assert batch.processed_at is not None


def test_worker_sets_batch_failed_on_prediction_error(db_session, uploaded_batch):
    with patch("ml_worker.processing.predict_news_items", side_effect=RuntimeError("model failed")):
        try:
            process_batch_sync(db_session, uploaded_batch.id)
        except RuntimeError:
            pass

    db_session.refresh(uploaded_batch)
    assert uploaded_batch.status == "failed"
    assert "model failed" in uploaded_batch.error_message


def test_worker_does_not_duplicate_predictions_on_retry(db_session, uploaded_batch):
    process_batch_sync(db_session, uploaded_batch.id)
    process_batch_sync(db_session, uploaded_batch.id)

    prediction_count = db_session.scalar(select(func.count(PredictionResult.id)).where(PredictionResult.batch_id == uploaded_batch.id))
    assert prediction_count == uploaded_batch.total_items
