# Testing Report

Дата проверки: 2026-06-27.

## Область проверки

Проверялись критические пользовательские сценарии проекта `bank-news-risk-monitoring`: регистрация, вход, баланс, покупка подписки, загрузка CSV, асинхронная постановка batch-задачи, обработка worker-ом, результаты и история.

Система оплаты и подписки не менялась. Баланс используется для покупки подписки, а активная подписка открывает доступ к загрузке CSV. Дополнительное списание кредитов за ML-запрос не вводилось.

## RabbitMQ

Redis-очередь заменена на RabbitMQ:

```text
RABBITMQ_URL=amqp://guest:guest@rabbitmq:5672/
RABBITMQ_QUEUE=ml_tasks
```

FastAPI публикует persistent-сообщение с `batch_id`. Worker читает очередь RabbitMQ, использует `prefetch_count=1`, подтверждает сообщение после успешной обработки и reject без requeue при ошибке.

RabbitMQ management UI доступен в Docker Compose на `http://localhost:15672`.

## ORM

В worker и results-эндпоинтах прямые `db.execute(...)` для доменной логики заменены на SQLAlchemy ORM query API. Проверка выполняется командой:

```bash
rg -n "cursor\.execute|session\.execute|db\.execute|text\(" app ml_worker
```

## Зависимости

Зависимости в `app/requirements.txt` и `ml_worker/requirements.txt` закреплены через `==`. Добавлена проверка:

```bash
python scripts/check_requirements_pinned.py
```

Результат локальной проверки: `All requirements are pinned.`

## E2E

Добавлен тест `app/tests/test_e2e_user_balance_ml_flow.py`.

Покрытые сценарии:

* пользователь без подписки не может загрузить CSV;
* пользователь пополняет демо-баланс и покупает подписку;
* CSV сохраняется как batch/news_items и получает статус `queued`;
* inference не запускается синхронно при upload;
* worker переводит batch в `completed` и сохраняет `prediction_results`;
* результаты доступны через `/results/{batch_id}` и CSV download;
* `/jobs` показывает завершённую загрузку;
* `/billing/transactions` показывает пополнение и покупку подписки;
* невалидный CSV не создаёт batch;
* ошибка worker-а переводит batch в `failed`.

Полный локальный прогон:

```bash
python -m pytest -p no:cacheprovider
```

Результат после расширения REST API покрытия: `94 passed`.

## Покрытие REST API endpoints

Фактически найденные REST/API endpoints:

```text
GET  /health
POST /auth/register
POST /auth/login
GET  /auth/me
GET  /billing/balance
POST /billing/top-up
GET  /billing/transactions
POST /subscriptions/buy
GET  /subscriptions/status
POST /news/upload
GET  /jobs
GET  /jobs/{batch_id}
GET  /results/{batch_id}
GET  /results/{batch_id}/download
```

Отдельных endpoint-ов `POST /predict` и `GET /predictions/history` в текущем коде нет. История ML-запросов покрыта через `/jobs` и `/jobs/{batch_id}`; история предсказаний покрыта через `/results/{batch_id}` и `/results/{batch_id}/download`.

| Endpoint | Positive tests | Negative tests | Комментарий |
|---|---:|---:|---|
| GET /health | да | не требуется | Healthcheck |
| POST /auth/register | да | да | duplicate email, invalid email, empty password |
| POST /auth/login | да | да | wrong password, unknown user, missing fields |
| GET /auth/me | да | да | no token, invalid token |
| GET /billing/balance | да | да | no auth |
| POST /billing/top-up | да | да | no auth, negative amount, zero amount |
| GET /billing/transactions | да | да | no auth, user scope |
| POST /subscriptions/buy | да | да | no auth, insufficient balance, no subscription on failure |
| GET /subscriptions/status | да | да | active, inactive, no auth |
| POST /news/upload | да | да | no auth, no subscription, invalid CSV, non-CSV, queue failure |
| GET /jobs | да | да | no auth, user scope |
| GET /jobs/{batch_id} | да | да | no auth, not found, foreign batch, invalid id |
| GET /results/{batch_id} | да | да | before worker, no auth, not found, foreign batch, invalid id |
| GET /results/{batch_id}/download | да | да | CSV columns, no auth, not found, foreign batch |

## Независимость тестов

REST API проверки разнесены по отдельным файлам:

```text
app/tests/test_auth_api.py
app/tests/test_billing_api.py
app/tests/test_subscriptions_api.py
app/tests/test_news_upload_api.py
app/tests/test_jobs_api.py
app/tests/test_results_api.py
app/tests/test_worker_processing.py
```

Каждый тест получает чистую SQLite БД через fixture `db_session` и сам создаёт нужного пользователя, batch или подписку. Большой e2e flow оставлен как smoke-test и не заменяет endpoint-level проверки.

## Команды проверки

Выполнены:

```bash
python -m pytest -p no:cacheprovider
python scripts/check_requirements_pinned.py
docker compose config --quiet
docker compose build
docker compose run --rm app pytest -p no:cacheprovider
```

Результат:

```text
94 passed
All requirements are pinned.
docker compose config --quiet: ok
docker compose build: ok
docker compose run --rm app pytest -p no:cacheprovider: 94 passed
```

## Docker

Проверены команды:

```bash
docker compose config --quiet
docker compose build
docker compose up -d --scale ml_worker=3
docker compose ps
```

Результат: `app`, `db`, `rabbitmq` и три `ml_worker` запущены. RabbitMQ имеет статус `healthy`, worker-ы подключаются к очереди `ml_tasks`.

Docker предупреждает об orphan-контейнерах от старой конфигурации (`redis`, `frontend`, `worker`, `api`). Они не входят в текущий `docker-compose.yaml`; автоматическое удаление не выполнялось.

## Ограничение сравнения

Архив `ml-forecasting-service(1).rar` не найден в рабочей директории проекта, поэтому сравнение с ним не выполнялось локально. Проверка проведена по требованиям из задачи и текущему состоянию репозитория.
