from __future__ import annotations

from sqlalchemy.orm import Session

from app.database.models import NewsItem, PredictionResult
from app.models.inference import inference_model


def predict_news_item(db: Session, item: NewsItem) -> PredictionResult:
    result = build_prediction_results([item])[0]
    db.add(result)
    db.flush()
    return result


def build_prediction_results(items: list[NewsItem]) -> list[PredictionResult]:
    predictions = inference_model.predict_many(
        [(item.title, item.entity_norm, item.text_fragment) for item in items]
    )
    return [
        PredictionResult(
            news_item_id=item.id,
            batch_id=item.batch_id,
            user_id=item.user_id,
            risk_score=prediction["risk_score"],
            alert_flag=prediction["alert_flag"],
            model_name=prediction["model_name"],
            threshold=prediction["threshold"],
        )
        for item, prediction in zip(items, predictions)
    ]


def predict_news_items(db: Session, items: list[NewsItem]) -> list[PredictionResult]:
    results = build_prediction_results(items)
    db.add_all(results)
    db.flush()
    return results
