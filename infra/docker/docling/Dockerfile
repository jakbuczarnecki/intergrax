# © Intergrax

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    libgl1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    docling[all]==2.76.0 \
    fastapi \
    uvicorn \
    python-multipart

COPY service.py /app/service.py

EXPOSE 8080

CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8080"]