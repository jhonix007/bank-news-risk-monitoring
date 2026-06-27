from __future__ import annotations


def test_login_bad_credentials_renders_html_error(client):
    response = client.post(
        "/login",
        data={"email": "missing@example.com", "password": "wrong-password"},
        follow_redirects=False,
    )

    assert response.status_code == 401
    assert "text/html" in response.headers["content-type"]
    assert "Неверный email или пароль." in response.text
    assert 'value="missing@example.com"' in response.text
    assert '"detail"' not in response.text
    assert 'value="wrong-password"' not in response.text


def test_register_duplicate_renders_html_error(client):
    assert client.post(
        "/register",
        data={"email": "dupe@example.com", "password": "secret1"},
        follow_redirects=False,
    ).status_code == 303

    response = client.post(
        "/register",
        data={"email": "dupe@example.com", "password": "another1"},
        follow_redirects=False,
    )

    assert response.status_code == 400
    assert "text/html" in response.headers["content-type"]
    assert "Пользователь с таким email уже существует." in response.text
    assert 'value="dupe@example.com"' in response.text
    assert '"detail"' not in response.text


def test_unauthenticated_web_pages_redirect_to_login(client):
    for path in ["/dashboard", "/billing", "/upload", "/jobs", "/results/1"]:
        response = client.get(path, follow_redirects=False)

        assert response.status_code == 303
        assert response.headers["location"] == "/login"


def test_upload_page_has_only_csv_file_input(client):
    client.post(
        "/register",
        data={"email": "upload-form@example.com", "password": "secret1"},
        follow_redirects=False,
    )

    response = client.get("/upload")

    assert response.status_code == 200
    assert 'input name="file" type="file"' in response.text
    assert 'accept=".csv"' in response.text
    for forbidden in [
        'name="title"',
        'name="text_fragment"',
        'name="entity_norm"',
        'name="source"',
        'name="url"',
        'name="published_at"',
        'name="bank"',
        'name="risk_type"',
    ]:
        assert forbidden not in response.text
