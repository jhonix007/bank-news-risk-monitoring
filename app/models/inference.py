from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from app.config import settings


MODEL_THRESHOLD = settings.model_threshold


class InferenceModel:
    def __init__(self) -> None:
        self.artifact = None
        self.model_path = Path(settings.model_path)
        if self.model_path.exists():
            self.artifact = joblib.load(self.model_path)

    @property
    def model_name(self) -> str:
        if self.artifact:
            return str(self.artifact.get("model_name", "tfidf_student_v2b"))
        return "fallback_keyword_model"

    @staticmethod
    def make_model_text(title: str, entity_norm: str, text_fragment: str) -> str:
        return f"{title or ''}\n{entity_norm or ''}\n{text_fragment or ''}"

    def predict(self, title: str, entity_norm: str, text_fragment: str) -> dict:
        return self.predict_many([(title, entity_norm, text_fragment)])[0]

    def predict_many(self, rows: list[tuple[str, str | None, str]]) -> list[dict]:
        model_texts = [self.make_model_text(title, entity_norm or "", text_fragment) for title, entity_norm, text_fragment in rows]
        threshold = float(self.artifact.get("threshold", MODEL_THRESHOLD)) if self.artifact else MODEL_THRESHOLD
        if self.artifact:
            X = self.artifact["vectorizer"].transform(pd.Series(model_texts))
            scores = self.artifact["classifier"].predict_proba(X)[:, 1]
        else:
            scores = [self._fallback_score(model_text) for model_text in model_texts]
        return [
            {
                "risk_score": round(float(score), 6),
                "alert_flag": int(float(score) >= threshold),
                "threshold": threshold,
                "model_name": self.model_name,
            }
            for score in scores
        ]

    @staticmethod
    def _fallback_score(text: str) -> float:
        text = text.lower()
        weighted_terms = {
            "санкц": 0.28,
            "суд": 0.20,
            "иск": 0.17,
            "мошен": 0.25,
            "хак": 0.24,
            "уголов": 0.24,
            "штраф": 0.20,
            "банкрот": 0.22,
            "отзыв": 0.26,
            "провер": 0.12,
            "наруш": 0.18,
        }
        return min(0.99, 0.12 + sum(weight for term, weight in weighted_terms.items() if term in text))


inference_model = InferenceModel()
