from app.tests.conftest import register_and_login


def test_user_can_top_up_balance(client):
    headers = register_and_login(client)
    response = client.post("/billing/top-up", headers=headers)
    assert response.status_code == 200
    assert response.json()["balance"] == 100.0


def test_buy_subscription_decreases_balance_and_sets_active_status(client):
    headers = register_and_login(client)
    client.post("/billing/top-up", headers=headers)
    response = client.post("/subscriptions/buy", headers=headers)
    assert response.status_code == 200
    assert response.json()["balance"] == 0.0
    status = client.get("/subscriptions/status", headers=headers)
    assert status.json()["subscription_active"] is True


def test_subscription_purchase_requires_balance(client):
    headers = register_and_login(client)
    response = client.post("/subscriptions/buy", headers=headers)
    assert response.status_code == 400
    assert "Недостаточно средств" in response.json()["detail"]
