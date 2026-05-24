"""Rule-based предразметка риск-категорий."""

from __future__ import annotations

import re


RISK_KEYWORDS = {
    "operational_issue": [
        "сбой",
        "не работает",
        "недоступ",
        "ошибка",
        "перебой",
        "проблемы с переводами",
        "не проходят платежи",
        "не открывается приложение",
    ],
    "data_leak_security": [
        "утечка",
        "персональные данные",
        "взлом",
        "кибератака",
        "информационная безопасность",
        "данные клиентов",
    ],
    "fraud_phishing": ["мошенник", "мошенничество", "фишинг", "обман", "хищение", "злоумышленники"],
    "legal_regulatory": [
        "суд",
        "иск",
        "штраф",
        "расследование",
        "проверка",
        "регулятор",
        "предписание",
        "нарушение",
    ],
    "sanctions": ["санкции", "ограничения", "блокирующие санкции", "sdn", "заморозка активов"],
    "customer_complaints": [
        "жалобы",
        "пожаловались",
        "клиенты жалуются",
        "массовые жалобы",
        "недовольны клиенты",
    ],
}

RISK_PRIORITY = [
    "data_leak_security",
    "sanctions",
    "fraud_phishing",
    "legal_regulatory",
    "operational_issue",
    "customer_complaints",
]


def _keyword_pattern(keyword: str) -> re.Pattern[str]:
    parts = [re.escape(part) for part in keyword.split()]
    pattern = r"\s+".join(parts)
    return re.compile(rf"(?<![\w-]){pattern}(?![\w-])", flags=re.IGNORECASE | re.UNICODE)


def find_risk_keywords(text: str) -> list[str]:
    """Возвращает уникальные найденные риск-слова в порядке появления."""
    if not text:
        return []

    found = []
    seen = set()
    for keywords in RISK_KEYWORDS.values():
        for keyword in keywords:
            for match in _keyword_pattern(keyword).finditer(text):
                key = keyword.lower()
                if key not in seen:
                    seen.add(key)
                    found.append((match.start(), keyword))

    return [keyword for _, keyword in sorted(found, key=lambda item: item[0])]


def detect_risk_type_candidate(text: str) -> str:
    """Определяет приоритетную риск-категорию по словарю ключевых слов."""
    if not text:
        return "no_risk"

    matched_categories = set()
    for category, keywords in RISK_KEYWORDS.items():
        if any(_keyword_pattern(keyword).search(text) for keyword in keywords):
            matched_categories.add(category)

    for category in RISK_PRIORITY:
        if category in matched_categories:
            return category
    return "no_risk"
