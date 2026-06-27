from __future__ import annotations

from app.auth.security import create_access_token


def test_get_results_success_after_worker(client, processed_batch, user_with_subscription):
    response = client.get(f"/results/{processed_batch.id}", headers={**user_with_subscription, "accept": "application/json"})

    assert response.status_code == 200
    result = response.json()[0]
    assert {"title", "text_fragment", "entity_norm", "risk_score", "alert_flag", "model_name", "threshold"} <= set(result)


def test_get_results_before_worker_returns_empty(client, uploaded_batch, user_with_subscription):
    response = client.get(f"/results/{uploaded_batch.id}", headers={**user_with_subscription, "accept": "application/json"})

    assert response.status_code == 200
    assert response.json() == []


def test_get_results_without_auth_returns_401(client, batch_factory):
    batch = batch_factory(email="results-no-auth@example.com", processed=True)

    response = client.get(f"/results/{batch.id}", headers={"accept": "application/json"})

    assert response.status_code == 401


def test_get_results_for_unknown_batch_returns_404(client, auth_headers):
    response = client.get("/results/9999", headers={**auth_headers, "accept": "application/json"})

    assert response.status_code == 404


def test_get_results_for_another_user_returns_404(client, batch_factory, auth_headers):
    foreign_batch = batch_factory(email="results-foreign@example.com", processed=True)

    response = client.get(f"/results/{foreign_batch.id}", headers={**auth_headers, "accept": "application/json"})

    assert response.status_code == 404


def test_download_results_csv_success(client, processed_batch, user_with_subscription):
    response = client.get(f"/results/{processed_batch.id}/download", headers=user_with_subscription)

    assert response.status_code == 200
    assert "text/csv" in response.headers["content-type"]
    assert "risk_score" in response.text


def test_download_results_csv_without_auth_returns_401(client, batch_factory):
    batch = batch_factory(email="download-no-auth@example.com", processed=True)

    response = client.get(f"/results/{batch.id}/download")

    assert response.status_code == 401


def test_download_results_csv_unknown_batch_returns_404(client, auth_headers):
    response = client.get("/results/9999/download", headers=auth_headers)

    assert response.status_code == 404


def test_download_results_csv_for_another_user_returns_404(client, batch_factory, auth_headers):
    foreign_batch = batch_factory(email="download-foreign@example.com", processed=True)

    response = client.get(f"/results/{foreign_batch.id}/download", headers=auth_headers)

    assert response.status_code == 404


def test_download_results_csv_contains_expected_columns(client, processed_batch, user_with_subscription):
    response = client.get(f"/results/{processed_batch.id}/download", headers=user_with_subscription)

    assert response.status_code == 200
    header = response.text.splitlines()[0]
    assert header == "title,entity_norm,text_fragment,risk_score,alert_flag,model_name,threshold"


def test_results_invalid_batch_id_returns_error(client, auth_headers):
    response = client.get("/results/not-an-id", headers={**auth_headers, "accept": "application/json"})

    assert response.status_code == 422
