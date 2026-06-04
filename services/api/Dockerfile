FROM python:3.13-slim

WORKDIR /app

COPY services/api/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY services/api/main.py .
COPY schemas/ schemas/

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
