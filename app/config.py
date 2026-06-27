from __future__ import annotations

import os
from pathlib import Path


class Settings:
    app_host: str = os.getenv("APP_HOST", "0.0.0.0")
    app_port: int = int(os.getenv("APP_PORT", "8080"))
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./storage/app.db")
    rabbitmq_url: str = os.getenv("RABBITMQ_URL", "amqp://guest:guest@rabbitmq:5672/")
    rabbitmq_queue: str = os.getenv("RABBITMQ_QUEUE", "ml_tasks")
    secret_key: str = os.getenv("SECRET_KEY", "change-me")
    access_token_expire_minutes: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))
    subscription_price: float = float(os.getenv("SUBSCRIPTION_PRICE", "100"))
    subscription_days: int = int(os.getenv("SUBSCRIPTION_DAYS", "30"))
    model_path: str = os.getenv("MODEL_PATH", "models/tfidf_student_v2b.joblib")
    model_threshold: float = float(os.getenv("MODEL_THRESHOLD", "0.50"))
    inference_batch_size: int = int(os.getenv("INFERENCE_BATCH_SIZE", "128"))
    upload_dir: Path = Path(os.getenv("UPLOAD_DIR", "storage/uploads"))
    result_dir: Path = Path(os.getenv("RESULT_DIR", "storage/results"))

    def ensure_dirs(self) -> None:
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.result_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
