from __future__ import annotations


def test_get_balance_success(client, auth_headers):
    response = client.get("/billing/balance", headers=auth_headers)

    assert response.status_code == 200
    assert response.json()["balance"] == 0.0


def test_get_balance_without_auth_returns_401(client):
    response = client.get("/billing/balance")

    assert response.status_code == 401


def test_top_up_balance_success(client, auth_headers):
    response = client.post("/billing/top-up", headers=auth_headers)

    assert response.status_code == 200
    assert response.json()["balance"] == 100.0


def test_top_up_negative_amount_returns_error(client, auth_headers):
    response = client.post("/billing/top-up", headers=auth_headers, json={"amount": -10})

    assert response.status_code == 400
    assert "больше нуля" in response.json()["detail"]


def test_top_up_zero_amount_returns_error(client, auth_headers):
    response = client.post("/billing/top-up", headers=auth_headers, json={"amount": 0})

    assert response.status_code == 400
    assert "больше нуля" in response.json()["detail"]


def test_top_up_without_auth_returns_401(client):
    response = client.post("/billing/top-up")

    assert response.status_code == 401


def test_transactions_history_contains_top_up(client, auth_headers):
    client.post("/billing/top-up", headers=auth_headers)

    response = client.get("/billing/transactions", headers=auth_headers)

    assert response.status_code == 200
    transactions = response.json()
    assert len(transactions) == 1
    assert transactions[0]["transaction_type"] == "top_up"
    assert transactions[0]["amount"] == 100.0


def test_transactions_history_without_auth_returns_401(client):
    response = client.get("/billing/transactions")

    assert response.status_code == 401


def test_transactions_history_is_user_scoped(client):
    first = client.post("/auth/register", json={"email": "tx-first@example.com", "password": "secret1"}).json()
    second = client.post("/auth/register", json={"email": "tx-second@example.com", "password": "secret1"}).json()
    first_headers = {"Authorization": f"Bearer {first['access_token']}"}
    second_headers = {"Authorization": f"Bearer {second['access_token']}"}
    client.post("/billing/top-up", headers=first_headers)
    client.post("/billing/top-up", headers=second_headers, json={"amount": 50})

    response = client.get("/billing/transactions", headers=first_headers)

    assert response.status_code == 200
    transactions = response.json()
    assert len(transactions) == 1
    assert transactions[0]["amount"] == 100.0
