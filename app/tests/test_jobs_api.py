from __future__ import annotations

from app.auth.security import create_access_token


def test_get_jobs_success(client, uploaded_batch, user_with_subscription):
    response = client.get("/jobs", headers={**user_with_subscription, "accept": "application/json"})

    assert response.status_code == 200
    assert response.json()[0]["batch_id"] == uploaded_batch.id


def test_get_jobs_without_auth_returns_401(client):
    response = client.get("/jobs", headers={"accept": "application/json"})

    assert response.status_code == 401


def test_get_jobs_returns_only_current_user_batches(client, batch_factory):
    own_batch = batch_factory(email="jobs-owner@example.com")
    batch_factory(email="jobs-foreign@example.com")
    headers = {"Authorization": f"Bearer {create_access_token(own_batch.user_id)}", "accept": "application/json"}

    response = client.get("/jobs", headers=headers)

    assert response.status_code == 200
    batch_ids = [item["batch_id"] for item in response.json()]
    assert batch_ids == [own_batch.id]


def test_get_job_by_id_success(client, batch_factory):
    batch = batch_factory(email="job-owner@example.com")
    headers = {"Authorization": f"Bearer {create_access_token(batch.user_id)}", "accept": "application/json"}

    response = client.get(f"/jobs/{batch.id}", headers=headers)

    assert response.status_code == 200
    assert response.json()["batch_id"] == batch.id


def test_get_job_by_id_without_auth_returns_401(client, batch_factory):
    batch = batch_factory(email="job-no-auth@example.com")

    response = client.get(f"/jobs/{batch.id}", headers={"accept": "application/json"})

    assert response.status_code == 401


def test_get_job_by_id_not_found_returns_404(client, auth_headers):
    response = client.get("/jobs/9999", headers={**auth_headers, "accept": "application/json"})

    assert response.status_code == 404


def test_get_job_by_id_of_another_user_returns_404(client, batch_factory, auth_headers):
    foreign_batch = batch_factory(email="job-foreign@example.com")

    response = client.get(f"/jobs/{foreign_batch.id}", headers={**auth_headers, "accept": "application/json"})

    assert response.status_code == 404


def test_get_job_by_invalid_id_returns_error(client, auth_headers):
    response = client.get("/jobs/not-an-id", headers={**auth_headers, "accept": "application/json"})

    assert response.status_code == 422
