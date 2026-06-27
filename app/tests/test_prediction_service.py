from app.models.inference import inference_model


def test_inference_returns_valid_prediction():
    prediction = inference_model.predict(
        title="Хакеры атаковали клиентов банка",
        entity_norm="ВТБ",
        text_fragment="Клиенты сообщили о мошеннических списаниях.",
    )
    assert 0 <= prediction["risk_score"] <= 1
    assert prediction["alert_flag"] in {0, 1}
    assert prediction["threshold"] == 0.5
    assert prediction["model_name"]
