from __future__ import annotations

import json

import pika

from app.config import settings


def enqueue_batch(batch_id: int) -> None:
    try:
        parameters = pika.URLParameters(settings.rabbitmq_url)
        connection = pika.BlockingConnection(parameters)
        channel = connection.channel()
        channel.queue_declare(queue=settings.rabbitmq_queue, durable=True)
        channel.basic_publish(
            exchange="",
            routing_key=settings.rabbitmq_queue,
            body=json.dumps({"batch_id": batch_id}).encode("utf-8"),
            properties=pika.BasicProperties(delivery_mode=2),
        )
        connection.close()
    except Exception as exc:
        raise RuntimeError(f"Не удалось поставить batch {batch_id} в RabbitMQ-очередь: {exc}") from exc
