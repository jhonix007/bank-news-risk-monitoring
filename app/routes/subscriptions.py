from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_user
from app.config import settings
from app.database.models import User
from app.database.session import get_db
from app.services.billing_service import add_subscription_purchase_transaction
from app.services.subscription_service import buy_subscription, get_active_subscription, is_subscription_active


router = APIRouter()


@router.post("/subscriptions/buy")
def buy(db: Annotated[Session, Depends(get_db)], user: Annotated[User, Depends(get_current_user)]):
    try:
        subscription = buy_subscription(db, user)
        add_subscription_purchase_transaction(db, user, settings.subscription_price)
        db.commit()
        return {
            "message": "Подписка куплена.",
            "plan_name": subscription.plan_name,
            "subscription_active": True,
            "ends_at": subscription.ends_at,
            "balance": user.balance,
        }
    except ValueError as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/subscriptions/status")
def status(db: Annotated[Session, Depends(get_db)], user: Annotated[User, Depends(get_current_user)]):
    subscription = get_active_subscription(db, user.id)
    return {
        "subscription_active": subscription is not None,
        "plan_name": subscription.plan_name if subscription else None,
        "ends_at": subscription.ends_at if subscription else None,
        "balance": user.balance,
        "price": settings.subscription_price,
    }


@router.post("/web/subscriptions/buy")
def web_buy(db: Annotated[Session, Depends(get_db)], user: Annotated[User, Depends(get_current_user)]):
    try:
        subscription = buy_subscription(db, user)
        add_subscription_purchase_transaction(db, user, settings.subscription_price)
        db.commit()
        return RedirectResponse("/billing", status_code=303)
    except ValueError:
        return RedirectResponse("/billing?error=balance", status_code=303)
