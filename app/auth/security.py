from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time

from app.config import settings


def hash_password(password: str) -> str:
    return hashlib.sha256((settings.secret_key + password).encode("utf-8")).hexdigest()


def verify_password(password: str, hashed_password: str) -> bool:
    return hmac.compare_digest(hash_password(password), hashed_password)


def create_access_token(user_id: int) -> str:
    payload = {
        "sub": user_id,
        "exp": int(time.time()) + settings.access_token_expire_minutes * 60,
    }
    raw_payload = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    encoded_payload = base64.urlsafe_b64encode(raw_payload).decode("ascii")
    signature = hmac.new(settings.secret_key.encode("utf-8"), encoded_payload.encode("ascii"), hashlib.sha256).hexdigest()
    return f"{encoded_payload}.{signature}"


def decode_access_token(token: str) -> int | None:
    try:
        encoded_payload, signature = token.split(".", 1)
        expected = hmac.new(settings.secret_key.encode("utf-8"), encoded_payload.encode("ascii"), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected):
            return None
        payload = json.loads(base64.urlsafe_b64decode(encoded_payload.encode("ascii")))
        if int(payload["exp"]) < int(time.time()):
            return None
        return int(payload["sub"])
    except Exception:
        return None
