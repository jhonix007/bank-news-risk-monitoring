from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, field_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_user
from app.auth.security import create_access_token, hash_password, verify_password
from app.database.models import User
from app.database.session import get_db


router = APIRouter()


class AuthPayload(BaseModel):
    email: str
    password: str

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str) -> str:
        normalized = value.strip().lower()
        if "@" not in normalized or "." not in normalized.rsplit("@", 1)[-1]:
            raise ValueError("Некорректный email.")
        return normalized

    @field_validator("password")
    @classmethod
    def validate_password(cls, value: str) -> str:
        if len(value.strip()) < 6:
            raise ValueError("Пароль должен содержать минимум 6 символов.")
        return value


def user_response(user: User) -> dict:
    return {"id": user.id, "email": user.email, "balance": user.balance}


@router.post("/auth/register")
def register(payload: AuthPayload, db: Annotated[Session, Depends(get_db)]):
    email = payload.email
    if db.scalar(select(User).where(User.email == email)):
        raise HTTPException(status_code=400, detail="Пользователь с таким email уже существует.")
    user = User(email=email, hashed_password=hash_password(payload.password))
    db.add(user)
    db.commit()
    db.refresh(user)
    return {"access_token": create_access_token(user.id), "token_type": "bearer", "user": user_response(user)}


@router.post("/auth/login")
def login(payload: AuthPayload, db: Annotated[Session, Depends(get_db)]):
    user = db.scalar(select(User).where(User.email == payload.email))
    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Неверный email или пароль.")
    return {"access_token": create_access_token(user.id), "token_type": "bearer", "user": user_response(user)}


@router.get("/auth/me")
def me(user: Annotated[User, Depends(get_current_user)]):
    return user_response(user)
