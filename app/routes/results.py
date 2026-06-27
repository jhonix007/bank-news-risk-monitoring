from __future__ import annotations

import csv
import io
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_user
from app.database.models import NewsBatch, NewsItem, PredictionResult, User
from app.database.session import get_db


router = APIRouter()


def ensure_owner(db: Session, batch_id: int, user_id: int) -> NewsBatch:
    batch = db.get(NewsBatch, batch_id)
    if not batch or batch.user_id != user_id:
        raise HTTPException(status_code=404, detail="Результаты не найдены.")
    return batch


def result_rows(db: Session, batch_id: int, user_id: int) -> list[dict]:
    ensure_owner(db, batch_id, user_id)
    rows = (
        db.query(NewsItem, PredictionResult)
        .join(PredictionResult, PredictionResult.news_item_id == NewsItem.id)
        .filter(NewsItem.batch_id == batch_id, NewsItem.user_id == user_id)
        .order_by(PredictionResult.risk_score.desc())
        .all()
    )
    return [
        {
            "title": item.title,
            "entity_norm": item.entity_norm,
            "text_fragment": item.text_fragment,
            "risk_score": pred.risk_score,
            "alert_flag": pred.alert_flag,
            "model_name": pred.model_name,
            "threshold": pred.threshold,
        }
        for item, pred in rows
    ]


@router.get("/results/{batch_id}")
def results(batch_id: int, db: Annotated[Session, Depends(get_db)], user: Annotated[User, Depends(get_current_user)]):
    return result_rows(db, batch_id, user.id)


@router.get("/results/{batch_id}/download")
def download(batch_id: int, db: Annotated[Session, Depends(get_db)], user: Annotated[User, Depends(get_current_user)]):
    rows = result_rows(db, batch_id, user.id)
    output = io.StringIO()
    fieldnames = ["title", "entity_norm", "text_fragment", "risk_score", "alert_flag", "model_name", "threshold"]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=results_batch_{batch_id}.csv"},
    )
