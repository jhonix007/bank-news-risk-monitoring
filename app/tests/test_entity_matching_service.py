from app.services.entity_matching_service import infer_entity_norm


def test_infer_entity_norm_detects_sberbank():
    assert infer_entity_norm("Сбербанк предупредил клиентов", "Сообщение от Сбера") == "Сбербанк"


def test_infer_entity_norm_detects_vtb():
    assert infer_entity_norm("Банк ВТБ открыл офис", "Новость для предпринимателей") == "ВТБ"


def test_infer_entity_norm_returns_none_when_bank_not_found():
    assert infer_entity_norm("Банк улучшил условия", "Новые ставки доступны клиентам") is None
