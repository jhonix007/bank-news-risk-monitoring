from __future__ import annotations


BANK_ALIASES = {
    "Сбербанк": ["сбербанк", "сбер", "сбера"],
    "ВТБ": ["втб", "банк втб"],
    "Газпромбанк": ["газпромбанк"],
    "Альфа-Банк": ["альфа-банк", "альфа банк", "альфа"],
    "Т-Банк": ["т-банк", "тинькофф", "тинькофф банк"],
    "Россельхозбанк": ["россельхозбанк", "рсхб"],
    "Росбанк": ["росбанк"],
}


def infer_entity_norm(title: str, text_fragment: str) -> str | None:
    text = f"{title or ''} {text_fragment or ''}".lower()
    matches: list[tuple[int, str]] = []

    for bank_name, aliases in BANK_ALIASES.items():
        positions = [text.find(alias) for alias in aliases if alias in text]
        if positions:
            matches.append((min(positions), bank_name))

    if not matches:
        return None
    return min(matches, key=lambda match: match[0])[1]
