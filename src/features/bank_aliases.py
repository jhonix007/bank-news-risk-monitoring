"""Справочник банков и поиск упоминаний в тексте."""

from __future__ import annotations

import re


BANK_CONTEXT_WORDS = [
    "банк",
    "банка",
    "банку",
    "банком",
    "банковский",
    "кредит",
    "кредиты",
    "вклад",
    "вклады",
    "карта",
    "карты",
    "счет",
    "счёт",
    "перевод",
    "переводы",
    "платеж",
    "платёж",
    "платежи",
    "платежей",
    "клиент",
    "клиенты",
    "приложение",
    "офис",
    "отделение",
    "цб",
    "центробанк",
    "лицензия",
    "финансовый",
    "финансовая",
]

STRICT_ALIASES = {
    "Сбербанк": ["сбербанк", "сбер", "сбера"],
    "ВТБ": ["втб", "банк втб"],
    "Т-Банк": ["т-банк", "тинькофф", "тинькофф банк"],
    "Альфа-Банк": ["альфа-банк", "альфа банк", "альфабанк"],
    "Газпромбанк": ["газпромбанк"],
    "Россельхозбанк": ["россельхозбанк"],
    "Совкомбанк": ["совкомбанк"],
    "Промсвязьбанк": ["промсвязьбанк", "банк псб"],
    "Банк Открытие": ["банк открытие", "фк открытие"],
    "Райффайзенбанк": ["райффайзенбанк"],
    "Росбанк": ["росбанк"],
    "МТС Банк": ["мтс банк"],
    "Почта Банк": ["почта банк"],
    "Банк Санкт-Петербург": ["банк санкт-петербург"],
    "Qiwi": ["qiwi банк", "киви банк"],
}

AMBIGUOUS_ALIASES = {
    "Газпромбанк": ["гпб"],
    "Совкомбанк": ["халва"],
    "Промсвязьбанк": ["псб"],
    "Банк Санкт-Петербург": ["бспб"],
    "Qiwi": ["киви"],
}

BANK_ALIASES = {
    entity_norm: aliases + AMBIGUOUS_ALIASES.get(entity_norm, [])
    for entity_norm, aliases in STRICT_ALIASES.items()
}

HALVA_CONTEXT_WORDS = ["карта", "банк", "совкомбанк"]
QIWI_CONTEXT_WORDS = ["банк", "кошелек", "кошелёк", "платеж", "платёж", "qiwi"]


def _alias_pattern(alias: str) -> re.Pattern[str]:
    parts = [re.escape(part) for part in alias.split()]
    pattern = r"\s+".join(parts)
    return re.compile(rf"(?<![\w-]){pattern}(?![\w-])", flags=re.IGNORECASE | re.UNICODE)


def _context_window(text: str, start: int, end: int, window: int = 80) -> str:
    return text[max(0, start - window) : min(len(text), end + window)].lower()


def _has_context(context: str, words: list[str]) -> bool:
    return any(_alias_pattern(word).search(context) for word in words)


def _context_words_for_alias(alias: str) -> list[str]:
    if alias == "халва":
        return HALVA_CONTEXT_WORDS
    if alias == "киви":
        return QIWI_CONTEXT_WORDS
    return BANK_CONTEXT_WORDS


def _build_match(entity_norm: str, alias: str, match: re.Match[str], match_type: str) -> dict:
    return {
        "entity_norm": entity_norm,
        "entity_mention": match.group(0),
        "alias": alias,
        "start": match.start(),
        "end": match.end(),
        "match_type": match_type,
        "match_confidence": "high" if match_type == "strict" else "medium",
    }


def _collect_matches(text: str) -> tuple[list[dict], list[dict]]:
    accepted: list[dict] = []
    rejected: list[dict] = []

    for entity_norm, aliases in STRICT_ALIASES.items():
        for alias in sorted(aliases, key=len, reverse=True):
            for match in _alias_pattern(alias).finditer(text):
                accepted.append(_build_match(entity_norm, alias, match, "strict"))

    for entity_norm, aliases in AMBIGUOUS_ALIASES.items():
        for alias in sorted(aliases, key=len, reverse=True):
            for match in _alias_pattern(alias).finditer(text):
                context = _context_window(text, match.start(), match.end())
                if _has_context(context, _context_words_for_alias(alias)):
                    accepted.append(_build_match(entity_norm, alias, match, "ambiguous_context"))
                else:
                    rejected.append(
                        {
                            "entity_norm": entity_norm,
                            "entity_mention": match.group(0),
                            "alias": alias,
                            "start": match.start(),
                            "end": match.end(),
                            "reason": "ambiguous_alias_without_bank_context",
                        }
                    )

    return accepted, rejected


def _select_best_match_per_entity(matches: list[dict]) -> list[dict]:
    best_matches = []
    for entity_norm in sorted({match["entity_norm"] for match in matches}):
        entity_matches = [match for match in matches if match["entity_norm"] == entity_norm]
        entity_matches.sort(
            key=lambda item: (
                item["match_confidence"] != "high",
                -(item["end"] - item["start"]),
                item["start"],
            )
        )
        best_matches.append(entity_matches[0])
    return sorted(best_matches, key=lambda item: item["start"])


def find_bank_mentions(text: str) -> list[dict]:
    """Возвращает найденные банки с нормализованным названием и позицией.

    Для каждого банка оставляется одно совпадение: самое длинное, а при равной
    длине самое раннее в тексте.
    """
    if not text:
        return []

    accepted, _ = _collect_matches(text)
    return _select_best_match_per_entity(accepted)


def find_rejected_bank_mentions(text: str) -> list[dict]:
    """Возвращает неоднозначные алиасы, отклоненные из-за отсутствия банковского контекста."""
    if not text:
        return []

    _, rejected = _collect_matches(text)
    return sorted(rejected, key=lambda item: item["start"])
