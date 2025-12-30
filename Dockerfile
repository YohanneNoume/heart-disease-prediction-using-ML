FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY predict.py .
COPY models/ ./models/
COPY data/ data/

EXPOSE 9696

CMD ["gunicorn", "--workers=2", "--bind=0.0.0.0:9696", "--timeout=120", "predict:app"]