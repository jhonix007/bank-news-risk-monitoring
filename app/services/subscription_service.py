from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.database.models import Subscription, User


PLAN_NAME = "Базовый"


def is_subscription_active(db: Session, user_id: int) -> bool:
    now = datetime.now(timezone.utc)
    return db.scalar(
        select(Subscription).where(
            Subscription.user_id == user_id,
            Subscription.starts_at <= now,
            Subscription.ends_at >= now,
        )
    ) is not None


def get_active_subscription(db: Session, user_id: int) -> Subscription | None:
    now = datetime.now(timezone.utc)
    return db.scalar(
        select(Subscription)
        .where(
            Subscription.user_id == user_id,
            Subscription.starts_at <= now,
            Subscription.ends_at >= now,
        )
        .order_by(Subscription.ends_at.desc())
    )


def buy_subscription(db: Session, user: User) -> Subscription:
    if user.balance < settings.subscription_price:
        raise ValueError("Недостаточно средств для покупки подписки.")
    now = datetime.now(timezone.utc)
    user.balance -= settings.subscription_price
    subscription = Subscription(
        user_id=user.id,
        plan_name=PLAN_NAME,
        starts_at=now,
        ends_at=now + timedelta(days=settings.subscription_days),
    )
    db.add(subscription)
    db.flush()
    return subscription
