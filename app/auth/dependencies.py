from __future__ import annotations

from typing import Annotated

from fastapi import Cookie, Depends, Header, HTTPException, Request
from sqlalchemy.orm import Session

from app.auth.security import decode_access_token
from app.database.models import User
from app.database.session import get_db


def get_current_user(
    db: Annotated[Session, Depends(get_db)],
    authorization: Annotated[str | None, Header()] = None,
    access_token: Annotated[str | None, Cookie()] = None,
    token: str | None = None,
) -> User:
    raw_token = token or access_token
    if authorization and authorization.startswith("Bearer "):
        raw_token = authorization.removeprefix("Bearer ").strip()
    if not raw_token:
        raise HTTPException(status_code=401, detail="Требуется авторизация.")
    user_id = decode_access_token(raw_token)
    if user_id is None:
        raise HTTPException(status_code=401, detail="Недействительный токен авторизации.")
    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=401, detail="Пользователь не найден.")
    return user


def get_optional_user(request: Request, db: Session) -> User | None:
    token = request.cookies.get("access_token")
    authorization = request.headers.get("authorization")
    if authorization and authorization.startswith("Bearer "):
        token = authorization.removeprefix("Bearer ").strip()
    if not token:
        return None
    user_id = decode_access_token(token)
    return db.get(User, user_id) if user_id else None
