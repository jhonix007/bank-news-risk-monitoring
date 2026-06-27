FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/

WORKDIR /app

COPY app/requirements.txt /tmp/requirements.txt
RUN pip install --default-timeout=120 --no-cache-dir -r /tmp/requirements.txt

COPY app /app

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]
