from __future__ import annotations

from unittest.mock import patch

from sqlalchemy import func, select

from app.database.models import BalanceTransaction, NewsBatch, NewsItem, PredictionResult, User
from app.tests.test_news_upload import CSV_CONTENT
from ml_worker.processing import process_batch_sync


LONG_FRAGMENT = " ".join(["Длинный фрагмент новости о банковском риске"] * 20)
LONG_CSV_CONTENT = (
    "source,url,published_at,title,text_fragment\n"
    f"РБК,u,2026,ВТБ сообщил о кибератаке,{LONG_FRAGMENT}\n"
    "ТАСС,u,2026,Сбербанк открыл офис,Нейтральная новость для клиентов Сбера\n"
    "Интерфакс,u,2026,Банк улучшил условия по вкладам,Новые ставки доступны для розничных клиентов\n"
)


def test_e2e_subscription_upload_worker_results_and_history(client, db_session):
    register = client.post("/auth/register", json={"email": "e2e@example.com", "password": "secret1"})
    assert register.status_code == 200
    headers = {"Authorization": f"Bearer {register.json()['access_token']}"}
    login = client.post("/login", data={"email": "e2e@example.com", "password": "secret1"})
    assert login.status_code == 200

    assert client.get("/billing/balance", headers=headers).json()["balance"] == 0.0
    assert (
        client.post(
            "/news/upload",
            headers=headers,
            files={"file": ("news.csv", CSV_CONTENT.encode("utf-8"), "text/csv")},
        ).status_code
        == 403
    )

    top_up = client.post("/billing/top-up", headers=headers)
    assert top_up.status_code == 200
    assert top_up.json()["balance"] == 100.0

    purchase = client.post("/subscriptions/buy", headers=headers)
    assert purchase.status_code == 200
    assert purchase.json()["subscription_active"] is True
    assert purchase.json()["balance"] == 0.0

    with patch("app.routes.news.enqueue_batch") as mock_enqueue:
        upload = client.post(
            "/news/upload",
            headers=headers,
            files={"file": ("news.csv", LONG_CSV_CONTENT.encode("utf-8"), "text/csv")},
        )
    assert upload.status_code == 200
    payload = upload.json()
    assert payload["status"] == "queued"
    assert payload["total_items"] == 3
    mock_enqueue.assert_called_once_with(payload["batch_id"])

    batch = db_session.get(NewsBatch, payload["batch_id"])
    assert batch is not None
    assert batch.status == "queued"
    assert batch.processed_items == 0
    assert db_session.scalar(select(func.count(PredictionResult.id)).where(PredictionResult.batch_id == batch.id)) == 0

    processed = process_batch_sync(db_session, batch.id)
    assert processed.status == "completed"
    assert processed.processed_items == 3

    results = client.get(f"/results/{batch.id}", headers={"accept": "application/json"})
    assert results.status_code == 200
    assert len(results.json()) == 3

    html_results = client.get(f"/results/{batch.id}", headers={"accept": "text/html"})
    assert html_results.status_code == 200
    assert '<div class="table-responsive">' in html_results.text
    assert 'class="results-table"' in html_results.text
    assert "Риск-скор" in html_results.text
    assert "Метка риска" in html_results.text
    assert "Модель" in html_results.text
    assert "Показать полностью" in html_results.text

    download = client.get(f"/results/{batch.id}/download")
    assert download.status_code == 200
    assert "risk_score" in download.text

    jobs = client.get("/jobs", headers={"accept": "application/json"})
    assert jobs.status_code == 200
    assert jobs.json()[0]["status"] == "completed"

    transactions = client.get("/billing/transactions", headers=headers)
    assert transactions.status_code == 200
    transaction_types = [item["transaction_type"] for item in transactions.json()]
    assert transaction_types.count("subscription_purchase") == 1
    assert transaction_types.count("top_up") == 1
    assert db_session.scalar(select(func.count(BalanceTransaction.id))) == 2


def test_e2e_invalid_csv_does_not_create_batch_or_change_balance(client, db_session):
    register = client.post("/auth/register", json={"email": "invalid-csv@example.com", "password": "secret1"})
    headers = {"Authorization": f"Bearer {register.json()['access_token']}"}
    client.post("/billing/top-up", headers=headers)
    client.post("/subscriptions/buy", headers=headers)

    response = client.post(
        "/news/upload",
        headers=headers,
        files={"file": ("bad.csv", b"wrong_column\nvalue\n", "text/csv")},
    )

    assert response.status_code == 400
    assert db_session.scalar(select(func.count(NewsBatch.id))) == 0
    assert client.get("/billing/balance", headers=headers).json()["balance"] == 0.0


def test_e2e_worker_failure_marks_batch_failed(client, db_session):
    register = client.post("/auth/register", json={"email": "worker-fail@example.com", "password": "secret1"})
    headers = {"Authorization": f"Bearer {register.json()['access_token']}"}
    client.post("/billing/top-up", headers=headers)
    client.post("/subscriptions/buy", headers=headers)

    with patch("app.routes.news.enqueue_batch"):
        upload = client.post(
            "/news/upload",
            headers=headers,
            files={"file": ("news.csv", CSV_CONTENT.encode("utf-8"), "text/csv")},
        )
    batch_id = upload.json()["batch_id"]

    with patch("ml_worker.processing.predict_news_items", side_effect=RuntimeError("model unavailable")):
        try:
            process_batch_sync(db_session, batch_id)
        except RuntimeError:
            pass

    batch = db_session.get(NewsBatch, batch_id)
    assert batch.status == "failed"
    assert "model unavailable" in batch.error_message


def test_results_page_renders_large_batch_inside_responsive_table(client, db_session):
    register = client.post("/auth/register", json={"email": "large-results@example.com", "password": "secret1"})
    assert register.status_code == 200
    login = client.post("/login", data={"email": "large-results@example.com", "password": "secret1"})
    assert login.status_code == 200

    user = db_session.scalar(select(User).where(User.email == "large-results@example.com"))
    batch = NewsBatch(
        user_id=user.id,
        original_filename="large.csv",
        stored_path="storage/uploads/large.csv",
        status="completed",
        total_items=805,
        processed_items=805,
    )
    db_session.add(batch)
    db_session.flush()

    items = [
        NewsItem(
            batch_id=batch.id,
            user_id=user.id,
            title=f"Длинный заголовок новости {index} о банковском событии",
            text_fragment=LONG_FRAGMENT,
            entity_norm="ВТБ" if index % 2 == 0 else "Сбербанк",
        )
        for index in range(805)
    ]
    db_session.add_all(items)
    db_session.flush()
    db_session.add_all(
        [
            PredictionResult(
                news_item_id=item.id,
                batch_id=batch.id,
                user_id=user.id,
                risk_score=0.896 if index % 2 == 0 else 0.104,
                alert_flag=1 if index % 2 == 0 else 0,
                model_name="tfidf_student_v2b",
                threshold=0.5,
            )
            for index, item in enumerate(items)
        ]
    )
    db_session.commit()

    response = client.get(f"/results/{batch.id}", headers={"accept": "text/html"})
    assert response.status_code == 200
    assert '<div class="table-responsive">' in response.text
    assert 'class="results-table"' in response.text
    assert 'id="only-alerts"' in response.text
    assert "Показать полностью" in response.text
    assert "0.896" in response.text
