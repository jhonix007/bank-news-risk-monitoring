from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api import app
from app.database.base import Base
from app.database.models import NewsBatch, NewsItem, PredictionResult, User
from app.database.session import get_db
from app.auth.security import hash_password


@pytest.fixture()
def db_session():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture()
def client(db_session):
    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    startup_handlers = list(app.router.on_startup)
    app.router.on_startup.clear()
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.router.on_startup[:] = startup_handlers
        app.dependency_overrides.clear()


def register_and_login(client: TestClient, email: str = "user@example.com") -> dict:
    response = client.post("/auth/register", json={"email": email, "password": "secret1"})
    assert response.status_code == 200
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def active_user_headers(client: TestClient) -> dict:
    headers = register_and_login(client)
    assert client.post("/billing/top-up", headers=headers).status_code == 200
    assert client.post("/subscriptions/buy", headers=headers).status_code == 200
    return headers


@pytest.fixture()
def create_user(db_session):
    def _create_user(email: str = "user@example.com", password: str = "secret1", balance: float = 0.0) -> User:
        user = User(email=email, hashed_password=hash_password(password), balance=balance)
        db_session.add(user)
        db_session.commit()
        db_session.refresh(user)
        return user

    return _create_user


@pytest.fixture()
def auth_headers(client: TestClient):
    return register_and_login(client, email="auth-user@example.com")


@pytest.fixture()
def user_with_balance(client: TestClient):
    headers = register_and_login(client, email="balance-user@example.com")
    response = client.post("/billing/top-up", headers=headers)
    assert response.status_code == 200
    return headers


@pytest.fixture()
def user_with_subscription(client: TestClient):
    headers = register_and_login(client, email="subscribed-user@example.com")
    assert client.post("/billing/top-up", headers=headers).status_code == 200
    assert client.post("/subscriptions/buy", headers=headers).status_code == 200
    return headers


@pytest.fixture()
def sample_news_csv_file():
    return (
        "source,url,published_at,title,text_fragment\n"
        "РБК,u,2026,ВТБ сообщил о кибератаке,Клиенты банка ВТБ сообщили о мошенничестве\n"
        "ТАСС,u,2026,Сбербанк открыл офис,Нейтральная новость для клиентов Сбера\n"
    )


@pytest.fixture()
def uploaded_batch(client: TestClient, db_session, user_with_subscription, sample_news_csv_file):
    from unittest.mock import patch

    with patch("app.routes.news.enqueue_batch"):
        response = client.post(
            "/news/upload",
            headers=user_with_subscription,
            files={"file": ("news.csv", sample_news_csv_file.encode("utf-8"), "text/csv")},
        )
    assert response.status_code == 200
    return db_session.get(NewsBatch, response.json()["batch_id"])


@pytest.fixture()
def processed_batch(db_session, uploaded_batch):
    from ml_worker.processing import process_batch_sync

    return process_batch_sync(db_session, uploaded_batch.id)


@pytest.fixture()
def batch_factory(db_session, create_user):
    def _batch_factory(email: str = "batch-owner@example.com", processed: bool = False) -> NewsBatch:
        user = create_user(email=email)
        batch = NewsBatch(
            user_id=user.id,
            original_filename="fixture.csv",
            stored_path="storage/uploads/fixture.csv",
            status="completed" if processed else "queued",
            total_items=1,
            processed_items=1 if processed else 0,
        )
        db_session.add(batch)
        db_session.flush()
        item = NewsItem(
            batch_id=batch.id,
            user_id=user.id,
            title="Новость о ВТБ",
            text_fragment="Клиенты сообщили о мошенничестве.",
            entity_norm="ВТБ",
        )
        db_session.add(item)
        db_session.flush()
        if processed:
            db_session.add(
                PredictionResult(
                    news_item_id=item.id,
                    batch_id=batch.id,
                    user_id=user.id,
                    risk_score=0.896,
                    alert_flag=1,
                    model_name="tfidf_student_v2b",
                    threshold=0.5,
                )
            )
        db_session.commit()
        db_session.refresh(batch)
        return batch

    return _batch_factory
