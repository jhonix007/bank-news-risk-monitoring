from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_user
from app.database.models import NewsBatch, User
from app.database.session import get_db


router = APIRouter()


def serialize_batch(batch: NewsBatch) -> dict:
    return {
        "id": batch.id,
        "batch_id": batch.id,
        "original_filename": batch.original_filename,
        "filename": batch.original_filename,
        "status": batch.status,
        "total_items": batch.total_items,
        "processed_items": batch.processed_items,
        "created_at": batch.created_at,
        "processed_at": batch.processed_at,
        "error_message": batch.error_message,
    }


@router.get("/jobs")
def jobs(db: Annotated[Session, Depends(get_db)], user: Annotated[User, Depends(get_current_user)]):
    batches = db.scalars(select(NewsBatch).where(NewsBatch.user_id == user.id).order_by(NewsBatch.created_at.desc()))
    return [serialize_batch(batch) for batch in batches]


@router.get("/jobs/{batch_id}")
def job(batch_id: int, db: Annotated[Session, Depends(get_db)], user: Annotated[User, Depends(get_current_user)]):
    batch = db.get(NewsBatch, batch_id)
    if not batch or batch.user_id != user.id:
        raise HTTPException(status_code=404, detail="Загрузка не найдена.")
    return serialize_batch(batch)
