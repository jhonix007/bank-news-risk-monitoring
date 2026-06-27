from __future__ import annotations

from unittest.mock import patch

from sqlalchemy import func, select

from app.database.models import NewsBatch, NewsItem, PredictionResult


def test_upload_news_csv_success(client, user_with_subscription, sample_news_csv_file):
    with patch("app.routes.news.enqueue_batch") as mock_enqueue:
        response = client.post(
            "/news/upload",
            headers=user_with_subscription,
            files={"file": ("news.csv", sample_news_csv_file.encode("utf-8"), "text/csv")},
        )

    assert response.status_code == 200
    assert response.json()["status"] == "queued"
    assert response.json()["total_items"] == 2
    mock_enqueue.assert_called_once_with(response.json()["batch_id"])


def test_upload_news_without_auth_returns_401(client, sample_news_csv_file):
    response = client.post(
        "/news/upload",
        files={"file": ("news.csv", sample_news_csv_file.encode("utf-8"), "text/csv")},
    )

    assert response.status_code == 401


def test_upload_news_without_subscription_returns_error(client, auth_headers, sample_news_csv_file):
    response = client.post(
        "/news/upload",
        headers=auth_headers,
        files={"file": ("news.csv", sample_news_csv_file.encode("utf-8"), "text/csv")},
    )

    assert response.status_code == 403


def test_upload_news_without_title_column_returns_error(client, user_with_subscription):
    response = client.post(
        "/news/upload",
        headers=user_with_subscription,
        files={"file": ("news.csv", b"text_fragment\nText\n", "text/csv")},
    )

    assert response.status_code == 400
    assert "title" in response.json()["detail"]


def test_upload_news_without_text_fragment_column_returns_error(client, user_with_subscription):
    response = client.post(
        "/news/upload",
        headers=user_with_subscription,
        files={"file": ("news.csv", b"title\nTitle\n", "text/csv")},
    )

    assert response.status_code == 400
    assert "text_fragment" in response.json()["detail"]


def test_upload_news_without_entity_norm_is_allowed(client, db_session, user_with_subscription):
    with patch("app.routes.news.enqueue_batch"):
        response = client.post(
            "/news/upload",
            headers=user_with_subscription,
            files={"file": ("news.csv", b"title,text_fragment\nVTB news,VTB risk text\n", "text/csv")},
        )

    assert response.status_code == 200
    item = db_session.scalar(select(NewsItem))
    assert item is not None
    assert item.entity_norm


def test_upload_empty_csv_returns_error(client, user_with_subscription):
    response = client.post(
        "/news/upload",
        headers=user_with_subscription,
        files={"file": ("news.csv", b"", "text/csv")},
    )

    assert response.status_code == 400


def test_upload_non_csv_file_returns_error(client, user_with_subscription):
    response = client.post(
        "/news/upload",
        headers=user_with_subscription,
        files={"file": ("news.txt", b"title,text_fragment\nTitle,Text\n", "text/plain")},
    )

    assert response.status_code == 400
    assert "CSV" in response.json()["detail"]


def test_upload_saves_all_news_items(client, db_session, user_with_subscription, sample_news_csv_file):
    with patch("app.routes.news.enqueue_batch"):
        response = client.post(
            "/news/upload",
            headers=user_with_subscription,
            files={"file": ("news.csv", sample_news_csv_file.encode("utf-8"), "text/csv")},
        )

    assert response.status_code == 200
    assert db_session.scalar(select(func.count(NewsItem.id))) == 2


def test_upload_creates_queued_batch(client, db_session, user_with_subscription, sample_news_csv_file):
    with patch("app.routes.news.enqueue_batch"):
        response = client.post(
            "/news/upload",
            headers=user_with_subscription,
            files={"file": ("news.csv", sample_news_csv_file.encode("utf-8"), "text/csv")},
        )

    batch = db_session.get(NewsBatch, response.json()["batch_id"])
    assert batch.status == "queued"
    assert batch.processed_items == 0


def test_upload_does_not_run_inference_synchronously(client, user_with_subscription, sample_news_csv_file):
    with patch("app.routes.news.enqueue_batch"), patch("app.services.prediction_service.predict_news_items") as mock_predict:
        response = client.post(
            "/news/upload",
            headers=user_with_subscription,
            files={"file": ("news.csv", sample_news_csv_file.encode("utf-8"), "text/csv")},
        )

    assert response.status_code == 200
    mock_predict.assert_not_called()


def test_upload_does_not_create_prediction_results_before_worker(client, db_session, user_with_subscription, sample_news_csv_file):
    with patch("app.routes.news.enqueue_batch"):
        response = client.post(
            "/news/upload",
            headers=user_with_subscription,
            files={"file": ("news.csv", sample_news_csv_file.encode("utf-8"), "text/csv")},
        )

    assert response.status_code == 200
    assert db_session.scalar(select(func.count(PredictionResult.id))) == 0


def test_upload_queue_failure_returns_503(client, db_session, user_with_subscription, sample_news_csv_file):
    with patch("app.routes.news.enqueue_batch", side_effect=RuntimeError("RabbitMQ недоступен")):
        response = client.post(
            "/news/upload",
            headers=user_with_subscription,
            files={"file": ("news.csv", sample_news_csv_file.encode("utf-8"), "text/csv")},
        )

    assert response.status_code == 503
    batch = db_session.scalar(select(NewsBatch))
    assert batch.status == "failed"
