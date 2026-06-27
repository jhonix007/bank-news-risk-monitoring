# Bank News Risk Monitoring

Прикладной ML-сервис для выявления риск-сигналов в банковских новостях.

Сервис принимает готовый CSV-файл со списком новостей, сохраняет каждую новость в PostgreSQL, создаёт batch/job, передаёт задачу worker-у, применяет ML-модель и показывает результаты в web-интерфейсе.

## О проекте

Банковским риск- и PR-командам нужно быстро находить новости, которые могут требовать ручной проверки: мошенничество, санкции, судебные споры, нарушения, киберинциденты, штрафы и другие риск-сигналы.

Сервис автоматизирует первичную сортировку новостей и помогает получить таблицу с `risk_score` и `alert_flag`.

## Пользовательский сценарий

```text
регистрация / вход
-> пополнение баланса
-> покупка подписки на сервис
-> загрузка CSV-файла со списком новостей
-> сохранение всех новостей из файла в PostgreSQL
-> создание batch/job
-> постановка задачи в очередь
-> worker применяет ML-модель
-> сохранение risk_score / alert_flag
-> просмотр результатов в web-интерфейсе
-> скачивание результатов CSV
```

Пользователь загружает обычный CSV со списком новостей. Обязательны только `title` и `text_fragment`. Поле `entity_norm` опционально: если оно отсутствует, сервис пытается автоматически определить банк по словарю алиасов. Такой сценарий ближе к реальному пользовательскому процессу, где пользователь не обязан заранее готовить ML-датасет.

## Архитектура

```text
app        FastAPI REST API + Jinja2 web-интерфейс
ml_worker  отдельный worker для обработки batch/job
db         PostgreSQL
rabbitmq   очередь задач RabbitMQ
```

Основные каталоги:

```text
app/          backend, REST API, web UI, SQLAlchemy, tests
ml_worker/    worker и sync processing
models/       TF-IDF model artifact
sample_data/  пример CSV для загрузки
storage/      uploads/results
```

## Доменная модель

| Таблица | Назначение |
|---|---|
| `users` | пользователи, пароль, баланс |
| `subscriptions` | подписки на сервис |
| `balance_transactions` | пополнения и покупки подписки |
| `news_batches` | batch/job загрузки CSV |
| `news_items` | отдельные новости из CSV |
| `prediction_results` | результат inference для каждой новости |

При загрузке CSV каждая валидная строка сохраняется как отдельная запись в `news_items`. После обработки worker создаёт одну запись `prediction_results` для каждой новости.

## REST API

Swagger доступен после запуска:

```text
http://localhost:8080/docs
```

Endpoints:

```text
GET  /health
POST /auth/register
POST /auth/login
GET  /auth/me
POST /billing/top-up
GET  /billing/balance
GET  /billing/transactions
POST /subscriptions/buy
GET  /subscriptions/status
POST /news/upload
GET  /jobs
GET  /jobs/{batch_id}
GET  /results/{batch_id}
GET  /results/{batch_id}/download
```

Ошибки API возвращаются на русском языке.

## Web UI

UI реализован как полноценный web-интерфейс на FastAPI + Jinja2 templates + CSS.
Streamlit не используется. React/Vite не используется как основной пользовательский интерфейс.

```text
http://localhost:8080
```

Страницы: `/`, `/register`, `/login`, `/logout`, `/dashboard`, `/billing`, `/upload`, `/jobs`, `/results/{batch_id}`.

Все тексты интерфейса, кнопки и сообщения написаны на русском языке.

Web-flow отделен от REST API:

```text
GET/POST /login
GET/POST /register
GET      /logout
GET      /dashboard
GET      /billing
GET/POST /upload
GET      /jobs
GET      /results/{batch_id}
```

Если пользователь без cookie открывает `/dashboard`, `/billing`, `/upload`, `/jobs` или `/results/{batch_id}`, сервис делает redirect на `/login`, а не возвращает JSON 401. Ошибки логина, регистрации и загрузки CSV показываются HTML-страницей на русском языке: email сохраняется, пароль очищается, raw JSON `detail` в web UI не показывается.

## Подписка и баланс

Платёжная интеграция в текущей версии не подключена; баланс и подписка реализованы как внутренняя бизнес-логика сервиса.

```text
Название: Базовый
Цена: 100 условных кредитов
Срок: 30 дней
```

Без активной подписки пользователь не может загрузить CSV-файл.

## Формат CSV

Обязательные колонки:

```text
title,text_fragment
```

Опциональные колонки:

```text
source,url,published_at,entity_norm
```

Поддерживаемый формат:

```text
source,url,published_at,title,text_fragment
```

Лишние колонки не ломают загрузку. Пустой файл или файл без обязательных колонок отклоняется с понятной ошибкой на русском языке.

Пример файла:

```text
sample_data/news_upload_sample.csv
```

Пример файла с заранее указанной сущностью:

```text
sample_data/news_upload_sample_with_entity_norm.csv
```

## Загрузка новостей

Страница `/upload` принимает только один файл:

```html
<input type="file" name="file" accept=".csv">
<button>Загрузить и обработать</button>
```

На странице нет отдельных полей `title`, `text_fragment`, `entity_norm`, `source`, `url`, `published_at`, выбора банка, источника, типа риска, ключевых слов или параметров модели.

После успешной загрузки показывается HTML-сообщение: файл загружен, сохранено новостей, ID загрузки и статус `queued`.

## Сохранение данных в БД

Endpoint `/news/upload` проверяет авторизацию и подписку, валидирует CSV, создаёт `news_batches`, сохраняет строки в `news_items`, ставит batch в RabbitMQ-очередь и возвращает статус `queued`.

## ML inference

Финальная исследовательская модель — `TF-IDF student v2b`.

```text
models/tfidf_student_v2b.joblib
```

Подготовка текста соответствует notebooks:

```python
model_text = title + "\n" + entity_norm + "\n" + text_fragment
```

Если `entity_norm` не указана во входном CSV и не найдена по алиасам, модель получает пустую строку вместо сущности.

Порог:

```text
MODEL_THRESHOLD=0.50
```

Если `.joblib`-артефакт отсутствует, сервис использует резервную keyword-based модель `fallback_keyword_model`, чтобы сохранить работоспособность inference pipeline. Финальная исследовательская модель — `TF-IDF student v2b`.

Пересборка артефакта:

```bash
python scripts/train_tfidf_student_v2b.py
```

## Исследовательские материалы

Исследовательские ноутбуки, домашние задания и эксперименты вынесены в папку `research/`, чтобы не смешивать сервисную часть с этапами подготовки данных и моделирования.

Внутри находятся:

* постановка бизнес-задачи;
* прототип продукта;
* benchmark;
* подготовка датасета;
* baseline/modeling;
* postprocessing;
* дополнительные эксперименты с `rubert-tiny2` и `Qwen2.5-0.5B + LoRA`, если они присутствовали в git history.

Сервисная часть использует финальную модельную логику `TF-IDF student v2b`; Qwen/RuBERT-ноутбуки сохранены как исследовательские эксперименты и не запускаются при старте Docker-сервиса.

## Асинхронная обработка

API не обрабатывает файл синхронно. API сохраняет новости и публикует `batch_id` в очередь RabbitMQ `ml_tasks`. Worker забирает batch и выполняет inference.

```text
CSV upload
-> news_batch status=queued
-> queue
-> ml_worker
-> status=processing
-> prediction_results
-> status=completed
```

Обработка файлов выполняется асинхронно. FastAPI-приложение только принимает CSV, сохраняет новости в БД и ставит batch-задачу в очередь. ML inference выполняется отдельным сервисом `ml_worker`. Количество worker-ов можно масштабировать командой `docker compose up --scale ml_worker=3`.

Worker обрабатывает новости чанками: размер задаётся переменной `INFERENCE_BATCH_SIZE` (`128` по умолчанию). После каждого чанка обновляется `processed_items`, поэтому для больших файлов видно прогресс обработки.

Сообщения в очереди RabbitMQ persistent (`delivery_mode=2`), а worker использует `prefetch_count=1`, чтобы несколько worker-ов равномерно разбирали независимые batch-задачи.

## Worker и масштабирование

```bash
docker compose up --build --scale ml_worker=3
```

Бизнес-логика worker вынесена в `process_batch_sync(batch_id: int)` и тестируется напрямую.

## Docker-запуск

```bash
cp .env.example .env
docker compose up --build
```

Адреса:

```text
Web UI:  http://localhost:8080
Swagger: http://localhost:8080/docs
RabbitMQ management: http://localhost:15672
```

Compose использует `env_file: .env` и hot reload:

```text
app:       uvicorn api:app --host 0.0.0.0 --port 8080 --reload
ml_worker: watchfiles "python -m worker" /worker
```

Основные bind mounts:

```text
./app:/app
./ml_worker:/worker
./models:/model_artifacts:ro
./storage:/app/storage
./storage:/worker/storage
```

## Hot reload

Изменения в `app/`, templates, static и `ml_worker/` применяются без пересборки контейнера.

Изменения в `requirements.txt`, Dockerfile, `.env` и compose требуют пересборки или перезапуска.

Если Docker предупреждает о двух compose-файлах, можно указать основной файл явно:

```bash
docker compose -f docker-compose.yaml up --build
```

## Быстрая проверка web UI

Проверить, что web-login не возвращает raw JSON:

```bash
curl -i -X POST http://localhost:8080/login \
  -H "Content-Type: application/x-www-form-urlencoded" \
  --data "email=missing@example.com&password=wrong"
```

Ожидаемо: `401`, `content-type: text/html`, текст `Неверный email или пароль.`, без JSON-поля `detail`.

Проверить redirect защищенной страницы:

```bash
curl -i http://localhost:8080/dashboard
```

Ожидаемо: `303 See Other`, `location: /login`.

Пример загрузки CSV через web-cookie:

```bash
curl -c cookies.txt -i -X POST http://localhost:8080/register \
  -H "Content-Type: application/x-www-form-urlencoded" \
  --data "email=demo@example.com&password=secret1"

curl -b cookies.txt -i -X POST http://localhost:8080/web/billing/top-up
curl -b cookies.txt -i -X POST http://localhost:8080/web/subscriptions/buy
curl -b cookies.txt -i -F "file=@sample_data/news_upload_sample.csv;type=text/csv" \
  http://localhost:8080/upload
```

## Переменные окружения

См. `.env.example`: `DATABASE_URL`, `RABBITMQ_URL`, `RABBITMQ_QUEUE`, `SECRET_KEY`, `SUBSCRIPTION_PRICE`, `SUBSCRIPTION_DAYS`, `MODEL_PATH`, `MODEL_THRESHOLD`, `INFERENCE_BATCH_SIZE`, `UPLOAD_DIR`, `RESULT_DIR`.

## Тесты

Локально:

```bash
python -m pytest -p no:cacheprovider
python scripts/check_requirements_pinned.py
```

Через Docker:

```bash
docker compose run --rm app pytest
```

Benchmark worker-а на 800 новостей:

```bash
python scripts/benchmark_worker_800.py
```

Покрыты health, auth REST API, billing REST API, история транзакций, подписка, валидация CSV, загрузка новостей, jobs/history, results/history, CSV download, inference, worker processing, web-auth и e2e smoke-сценарий от регистрации до результатов ML. Тесты REST API разделены на независимые positive/negative функции по endpoint-ам.

История ML-запросов реализована через `news_batches` и endpoint-ы `/jobs`, `/jobs/{batch_id}`. История предсказаний реализована через `prediction_results` и endpoint-ы `/results/{batch_id}`, `/results/{batch_id}/download`. Отдельных endpoint-ов `/predict` и `/predictions/history` в текущей версии сервиса нет.

Дополнительное списание кредитов за ML-запрос не вводится: баланс используется для покупки подписки, а доступ к загрузке CSV контролируется активной подпиской.

## Визуальный стиль

Интерфейс сервиса выполнен в сдержанной финансовой стилистике: светлый фон, белые карточки, графитовый текст и бордовый акцентный цвет. В hero-блоке главной страницы используется деликатный SVG-фон с тонкой сеткой и абстрактной data-line графикой.

## Как проверить сохранение новостей в БД

```bash
docker compose exec db psql -U postgres -d bank_news
```

```sql
select id, original_filename, status, total_items, processed_items from news_batches;
select id, batch_id, title, entity_norm from news_items order by id;
select id, batch_id, risk_score, alert_flag, model_name from prediction_results order by id;
```

## Ограничения текущей версии

* Текущая версия сфокусирована на пакетной обработке CSV-файлов.
* Промышленный live-scraping новостей не реализован.
* Источники мониторинга не настраиваются в интерфейсе.
* Пользователь загружает CSV-файл со списком новостей.
* Сервис не требует выбора банков в интерфейсе.
* `entity_norm` можно передать во входном файле, но это необязательно.
* Финальная исследовательская модель — `TF-IDF student v2b`.
* Если основной `.joblib`-артефакт отсутствует, используется резервная keyword-based модель.
* Нейросетевые эксперименты с `rubert-tiny2` и `Qwen2.5-0.5B + LoRA` сохранены как исследовательская часть и не используются в production inference.

## Соответствие критериям ДЗ №8

| Критерий | Реализация |
|---|---|
| Доменная модель сервиса | SQLAlchemy-модели: User, Subscription, NewsBatch, NewsItem, PredictionResult, BalanceTransaction |
| Хранение данных в СУБД | PostgreSQL через SQLAlchemy |
| REST интерфейс | FastAPI endpoints |
| Пользовательский интерфейс | Полноценный web-интерфейс на FastAPI/Jinja2 templates + CSS, без Streamlit |
| Web/API разделение | Web routes возвращают HTML/redirect, REST `/auth/*` и другие API endpoints возвращают JSON |
| Web error handling | Ошибки логина, регистрации и загрузки CSV показываются в HTML на русском языке, без raw JSON |
| Тесты критических частей | pytest-тесты в app/tests |
| Docker контейнер | app/Dockerfile + ml_worker/Dockerfile + docker-compose.yaml |
| Hot reload | uvicorn `--reload`, worker через `watchfiles`, bind mounts к `app`, `ml_worker`, `models`, `storage` |
| Масштабирование воркеров | docker compose up --scale ml_worker=3 |
