from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database.models import BalanceTransaction, User


TOP_UP_AMOUNT = 100.0


def top_up_balance(db: Session, user: User, amount: float | None = TOP_UP_AMOUNT) -> BalanceTransaction:
    amount = TOP_UP_AMOUNT if amount is None else amount
    if amount <= 0:
        raise ValueError("Сумма пополнения должна быть больше нуля.")
    user.balance += amount
    transaction = BalanceTransaction(user_id=user.id, amount=amount, transaction_type="top_up")
    db.add(transaction)
    db.flush()
    return transaction


def add_subscription_purchase_transaction(db: Session, user: User, amount: float) -> BalanceTransaction:
    transaction = BalanceTransaction(user_id=user.id, amount=-abs(amount), transaction_type="subscription_purchase")
    db.add(transaction)
    db.flush()
    return transaction


def list_transactions(db: Session, user_id: int) -> list[BalanceTransaction]:
    return list(
        db.scalars(
            select(BalanceTransaction)
            .where(BalanceTransaction.user_id == user_id)
            .order_by(BalanceTransaction.created_at.desc())
        )
    )
