from __future__ import annotations

import json
import logging
import time

import pika

from app.config import settings

try:
    from ml_worker.tasks import process_batch_task
except ModuleNotFoundError:
    from tasks import process_batch_task


logger = logging.getLogger(__name__)


def parse_batch_id(body: bytes) -> int:
    payload = json.loads(body.decode("utf-8"))
    batch_id = payload.get("batch_id")
    if not isinstance(batch_id, int):
        raise ValueError("RabbitMQ message must contain integer batch_id.")
    return batch_id


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    while True:
        try:
            parameters = pika.URLParameters(settings.rabbitmq_url)
            connection = pika.BlockingConnection(parameters)
            channel = connection.channel()
            channel.queue_declare(queue=settings.rabbitmq_queue, durable=True)
            channel.basic_qos(prefetch_count=1)

            def callback(ch, method, properties, body) -> None:
                try:
                    batch_id = parse_batch_id(body)
                    logger.info("Received batch_id=%s from RabbitMQ.", batch_id)
                    process_batch_task(batch_id)
                    ch.basic_ack(delivery_tag=method.delivery_tag)
                except Exception:
                    logger.exception("Failed to process RabbitMQ message; rejecting without requeue.")
                    ch.basic_reject(delivery_tag=method.delivery_tag, requeue=False)

            channel.basic_consume(queue=settings.rabbitmq_queue, on_message_callback=callback)
            logger.info("ML worker connected to RabbitMQ queue '%s' and waits for batch_id.", settings.rabbitmq_queue)
            channel.start_consuming()
        except KeyboardInterrupt:
            raise
        except Exception:
            logger.exception("RabbitMQ worker loop failed; reconnecting in 2s.")
            time.sleep(2)


if __name__ == "__main__":
    main()
