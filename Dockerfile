# Laby ADK agent service — Python on Cloud Run.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080

WORKDIR /app

# Install deps first for layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code.
COPY server.py .
COPY agent ./agent

# Run as non-root.
RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin laby \
    && chown -R laby:laby /app
USER laby

EXPOSE 8080

# Cloud Run sets $PORT; uvicorn binds to it.
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-8080}"]
