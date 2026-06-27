from __future__ import annotations

from sqlalchemy import select

from app.database.models import BalanceTransaction, Subscription


def test_buy_subscription_success(client, user_with_balance):
    response = client.post("/subscriptions/buy", headers=user_with_balance)

    assert response.status_code == 200
    assert response.json()["subscription_active"] is True
    assert response.json()["balance"] == 0.0


def test_buy_subscription_without_auth_returns_401(client):
    response = client.post("/subscriptions/buy")

    assert response.status_code == 401


def test_buy_subscription_insufficient_balance_returns_error(client, auth_headers):
    response = client.post("/subscriptions/buy", headers=auth_headers)

    assert response.status_code == 400
    assert "Недостаточно средств" in response.json()["detail"]


def test_buy_subscription_insufficient_balance_does_not_create_subscription(client, db_session, auth_headers):
    response = client.post("/subscriptions/buy", headers=auth_headers)

    assert response.status_code == 400
    assert db_session.scalar(select(Subscription)) is None


def test_buy_subscription_creates_transaction(client, db_session, user_with_balance):
    response = client.post("/subscriptions/buy", headers=user_with_balance)

    assert response.status_code == 200
    transaction = db_session.scalar(select(BalanceTransaction).where(BalanceTransaction.transaction_type == "subscription_purchase"))
    assert transaction is not None
    assert transaction.amount == -100.0


def test_get_subscription_status_active(client, user_with_subscription):
    response = client.get("/subscriptions/status", headers=user_with_subscription)

    assert response.status_code == 200
    assert response.json()["subscription_active"] is True


def test_get_subscription_status_inactive(client, auth_headers):
    response = client.get("/subscriptions/status", headers=auth_headers)

    assert response.status_code == 200
    assert response.json()["subscription_active"] is False


def test_get_subscription_status_without_auth_returns_401(client):
    response = client.get("/subscriptions/status")

    assert response.status_code == 401
