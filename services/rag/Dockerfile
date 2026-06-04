FROM python:3.13-slim

WORKDIR /app

COPY services/rag/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY services/rag/rag.py .
COPY schemas/ schemas/

CMD ["celery", "-A", "rag", "worker", "--loglevel=info", "--queues=rag"]
