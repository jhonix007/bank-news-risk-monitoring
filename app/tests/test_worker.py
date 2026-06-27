from unittest.mock import patch

from sqlalchemy import func, select

from app.database.models import NewsBatch, NewsItem, PredictionResult, User
from app.tests.conftest import active_user_headers
from app.tests.test_news_upload import CSV_CONTENT
from ml_worker.processing import process_batch_sync


def test_worker_processes_batch_and_saves_predictions(client, db_session):
    headers = active_user_headers(client)
    with patch("app.routes.news.enqueue_batch"):
        response = client.post(
            "/news/upload",
            headers=headers,
            files={"file": ("news.csv", CSV_CONTENT.encode("utf-8"), "text/csv")},
        )
    batch_id = response.json()["batch_id"]
    queued_batch = db_session.get(NewsBatch, batch_id)
    assert queued_batch.status == "queued"
    assert queued_batch.processed_items == 0

    batch = process_batch_sync(db_session, batch_id)
    assert batch.status == "completed"
    assert batch.processed_items == 3

    predictions = list(db_session.scalars(select(PredictionResult).where(PredictionResult.batch_id == batch_id)))
    assert len(predictions) == 3
    assert all(0 <= pred.risk_score <= 1 for pred in predictions)


def test_worker_sets_batch_to_processing_before_inference(client, db_session):
    headers = active_user_headers(client)
    with patch("app.routes.news.enqueue_batch"):
        response = client.post(
            "/news/upload",
            headers=headers,
            files={"file": ("news.csv", CSV_CONTENT.encode("utf-8"), "text/csv")},
        )
    batch_id = response.json()["batch_id"]

    original_predict = process_batch_sync.__globals__["predict_news_items"]

    def assert_processing_status(db, items):
        batch = db.get(NewsBatch, batch_id)
        assert batch.status == "processing"
        return original_predict(db, items)

    with patch("ml_worker.processing.predict_news_items", side_effect=assert_processing_status):
        batch = process_batch_sync(db_session, batch_id)

    assert batch.status == "completed"
    assert batch.processed_items == batch.total_items


def test_worker_processes_many_items_in_batches(db_session):
    user = User(email="many-items@example.com", hashed_password="hashed")
    db_session.add(user)
    db_session.flush()
    batch = NewsBatch(
        user_id=user.id,
        original_filename="many.csv",
        stored_path="storage/uploads/many.csv",
        status="queued",
        total_items=250,
        processed_items=0,
    )
    db_session.add(batch)
    db_session.flush()
    db_session.add_all(
        [
            NewsItem(
                batch_id=batch.id,
                user_id=user.id,
                title=f"Новость {index} о ВТБ",
                text_fragment="Клиенты сообщили о мошенничестве и проверке операций.",
                entity_norm="ВТБ",
            )
            for index in range(250)
        ]
    )
    db_session.commit()

    chunk_sizes: list[int] = []
    original_predict = process_batch_sync.__globals__["predict_news_items"]

    def record_chunk_size(db, items):
        chunk_sizes.append(len(items))
        return original_predict(db, items)

    with patch("ml_worker.processing.PROCESSING_CHUNK_SIZE", 50), patch(
        "ml_worker.processing.predict_news_items", side_effect=record_chunk_size
    ):
        processed_batch = process_batch_sync(db_session, batch.id)

    prediction_count = db_session.scalar(
        select(func.count(PredictionResult.id)).where(PredictionResult.batch_id == batch.id)
    )
    assert processed_batch.status == "completed"
    assert processed_batch.processed_items == 250
    assert prediction_count == 250
    assert chunk_sizes == [50, 50, 50, 50, 50]
