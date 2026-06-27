from __future__ import annotations


def test_register_user_success(client):
    response = client.post("/auth/register", json={"email": "new-user@example.com", "password": "secret1"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["access_token"]
    assert payload["token_type"] == "bearer"
    assert payload["user"]["email"] == "new-user@example.com"


def test_register_duplicate_email_returns_error(client):
    client.post("/auth/register", json={"email": "duplicate@example.com", "password": "secret1"})

    response = client.post("/auth/register", json={"email": "duplicate@example.com", "password": "secret1"})

    assert response.status_code == 400
    assert "существует" in response.json()["detail"]


def test_register_invalid_email_returns_error(client):
    response = client.post("/auth/register", json={"email": "bad-email", "password": "secret1"})

    assert response.status_code == 422


def test_register_empty_password_returns_error(client):
    response = client.post("/auth/register", json={"email": "empty-password@example.com", "password": ""})

    assert response.status_code == 422


def test_login_success(client):
    client.post("/auth/register", json={"email": "login@example.com", "password": "secret1"})

    response = client.post("/auth/login", json={"email": "login@example.com", "password": "secret1"})

    assert response.status_code == 200
    assert response.json()["access_token"]
    assert response.json()["user"]["email"] == "login@example.com"


def test_login_wrong_password_returns_error(client):
    client.post("/auth/register", json={"email": "wrong-password@example.com", "password": "secret1"})

    response = client.post("/auth/login", json={"email": "wrong-password@example.com", "password": "wrong11"})

    assert response.status_code == 401


def test_login_unknown_user_returns_error(client):
    response = client.post("/auth/login", json={"email": "unknown@example.com", "password": "secret1"})

    assert response.status_code == 401


def test_login_missing_fields_returns_error(client):
    response = client.post("/auth/login", json={"email": "missing-password@example.com"})

    assert response.status_code == 422


def test_get_me_success(client, auth_headers):
    response = client.get("/auth/me", headers=auth_headers)

    assert response.status_code == 200
    assert response.json()["email"] == "auth-user@example.com"


def test_get_me_without_token_returns_401(client):
    response = client.get("/auth/me")

    assert response.status_code == 401


def test_get_me_with_invalid_token_returns_401(client):
    response = client.get("/auth/me", headers={"Authorization": "Bearer invalid-token"})

    assert response.status_code == 401
