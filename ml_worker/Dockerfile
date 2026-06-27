FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/:/worker

WORKDIR /worker

COPY ml_worker/requirements.txt /tmp/requirements.txt
RUN pip install --default-timeout=120 --no-cache-dir -r /tmp/requirements.txt

COPY app /app
COPY ml_worker /worker

CMD ["watchfiles", "python -m worker", "/worker"]
