from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_user
from app.database.models import User
from app.database.session import get_db
from app.services.news_file_service import create_news_batch_from_upload
from app.services.subscription_service import is_subscription_active
from app.services.task_service import enqueue_batch


router = APIRouter()


@router.post("/news/upload")
async def upload_news(
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
    file: UploadFile = File(...),
):
    if not is_subscription_active(db, user.id):
        raise HTTPException(status_code=403, detail="Для загрузки новостей нужна активная подписка.")
    content = await file.read()
    try:
        batch = create_news_batch_from_upload(db, user, file, content)
        db.commit()
        try:
            enqueue_batch(batch.id)
        except RuntimeError as exc:
            batch.status = "failed"
            batch.error_message = str(exc)
            db.commit()
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {
            "batch_id": batch.id,
            "status": batch.status,
            "total_items": batch.total_items,
            "message": "Файл загружен. Новости сохранены и поставлены в очередь на обработку.",
        }
    except ValueError as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
