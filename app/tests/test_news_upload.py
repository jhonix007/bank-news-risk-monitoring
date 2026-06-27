from unittest.mock import patch

from sqlalchemy import func, select

from app.database.models import NewsBatch, NewsItem, PredictionResult
from app.tests.conftest import active_user_headers, register_and_login


CSV_CONTENT = (
    "source,url,published_at,title,text_fragment\n"
    "РБК,u,2026,ВТБ сообщил о кибератаке,Клиенты банка ВТБ сообщили о мошенничестве\n"
    "ТАСС,u,2026,Сбербанк открыл офис,Нейтральная новость для клиентов Сбера\n"
    "Интерфакс,u,2026,Банк улучшил условия по вкладам,Новые ставки доступны для розничных клиентов\n"
)


def test_user_without_subscription_cannot_upload(client):
    headers = register_and_login(client)
    response = client.post(
        "/news/upload",
        headers=headers,
        files={"file": ("news.csv", CSV_CONTENT.encode("utf-8"), "text/csv")},
    )
    assert response.status_code == 403


def test_user_with_subscription_uploads_csv_and_news_are_saved(client, db_session):
    headers = active_user_headers(client)
    with patch("app.routes.news.enqueue_batch") as mock_enqueue:
        response = client.post(
            "/news/upload",
            headers=headers,
            files={"file": ("news.csv", CSV_CONTENT.encode("utf-8"), "text/csv")},
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "queued"
    assert payload["total_items"] == 3
    mock_enqueue.assert_called_once_with(payload["batch_id"])

    batch = db_session.get(NewsBatch, payload["batch_id"])
    assert batch is not None
    assert batch.total_items == 3
    assert batch.processed_items == 0
    assert batch.status == "queued"
    items = list(db_session.scalars(select(NewsItem).where(NewsItem.batch_id == batch.id).order_by(NewsItem.id)))
    assert len(items) == 3
    assert [item.entity_norm for item in items] == ["ВТБ", "Сбербанк", "Не определено"]
    prediction_count = db_session.scalar(select(func.count(PredictionResult.id)).where(PredictionResult.batch_id == batch.id))
    assert prediction_count == 0


def test_csv_with_provided_entity_norm_keeps_value(client, db_session):
    headers = active_user_headers(client)
    csv_content = (
        "title,text_fragment,entity_norm\n"
        "Новость без алиаса,В тексте нет названия банка,Газпромбанк\n"
    )
    with patch("app.routes.news.enqueue_batch"):
        response = client.post(
            "/news/upload",
            headers=headers,
            files={"file": ("news.csv", csv_content.encode("utf-8"), "text/csv")},
        )
    assert response.status_code == 200

    item = db_session.scalar(select(NewsItem))
    assert item is not None
    assert item.entity_norm == "Газпромбанк"


def test_news_upload_does_not_run_inference_synchronously(client):
    headers = active_user_headers(client)
    with patch("app.routes.news.enqueue_batch"), patch("app.services.prediction_service.predict_news_items") as mock_predict:
        response = client.post(
            "/news/upload",
            headers=headers,
            files={"file": ("news.csv", CSV_CONTENT.encode("utf-8"), "text/csv")},
        )

    assert response.status_code == 200
    mock_predict.assert_not_called()
