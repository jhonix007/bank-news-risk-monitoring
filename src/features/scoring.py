"""Расчет объяснимого risk_score_v1."""

from __future__ import annotations


RISK_TYPE_WEIGHTS = {
    "no_risk": 0,
    "customer_complaints": 20,
    "operational_issue": 25,
    "fraud_phishing": 30,
    "legal_regulatory": 30,
    "sanctions": 35,
    "data_leak_security": 40,
    "other_risk": 20,
}

ENTITY_RELEVANCE_WEIGHTS = {
    "direct": 30,
    "indirect": 10,
    "mentioned_only": 0,
    "unclear": 0,
}

ALERT_FLAG_WEIGHTS = {1: 30, 0: 0}


def calculate_risk_score(row) -> int:
    """Считает risk_score_v1; при пустых или некорректных значениях возвращает 0."""
    try:
        risk_type = str(row.get("risk_type", "")).strip()
        entity_relevance = str(row.get("entity_relevance", "")).strip()
        alert_flag = int(row.get("alert_flag"))
    except (TypeError, ValueError):
        return 0

    if (
        risk_type not in RISK_TYPE_WEIGHTS
        or entity_relevance not in ENTITY_RELEVANCE_WEIGHTS
        or alert_flag not in ALERT_FLAG_WEIGHTS
    ):
        return 0

    return min(
        100,
        RISK_TYPE_WEIGHTS[risk_type]
        + ENTITY_RELEVANCE_WEIGHTS[entity_relevance]
        + ALERT_FLAG_WEIGHTS[alert_flag],
    )
