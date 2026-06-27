from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Body, Depends, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_user
from app.database.models import User
from app.database.session import get_db
from app.services.billing_service import list_transactions, top_up_balance


router = APIRouter()


class TopUpPayload(BaseModel):
    amount: float | None = None


@router.post("/billing/top-up")
def top_up(
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
    payload: Annotated[TopUpPayload | None, Body()] = None,
):
    try:
        top_up_balance(db, user, amount=payload.amount if payload and payload.amount is not None else None)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    db.commit()
    return {"message": "Демо-баланс пополнен.", "balance": user.balance}


@router.get("/billing/balance")
def balance(user: Annotated[User, Depends(get_current_user)]):
    return {"balance": user.balance}


@router.get("/billing/transactions")
def transactions(db: Annotated[Session, Depends(get_db)], user: Annotated[User, Depends(get_current_user)]):
    return [
        {"id": tx.id, "amount": tx.amount, "transaction_type": tx.transaction_type, "created_at": tx.created_at}
        for tx in list_transactions(db, user.id)
    ]


@router.post("/web/billing/top-up")
def web_top_up(db: Annotated[Session, Depends(get_db)], user: Annotated[User, Depends(get_current_user)]):
    top_up_balance(db, user)
    db.commit()
    return RedirectResponse("/billing", status_code=303)
