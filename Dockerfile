# Laby ADK agent service — Python on Cloud Run.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/tmp \
    XDG_CACHE_HOME=/tmp/.cache \
    D10_USAGE_OUTBOX_PATH=/var/lib/dentnode-ai/d10-usage-outbox.sqlite3 \
    PORT=8080

WORKDIR /app

# Install deps first for layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code.
COPY server.py .
COPY agent ./agent
COPY scan_review ./scan_review
# server.py imports scan_qa unconditionally — omitting this kills the container
# on startup with ModuleNotFoundError, before any health check can run.
COPY scan_qa ./scan_qa

# Run as non-root.
RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin laby \
    && install -d -o laby -g laby -m 0700 /var/lib/dentnode-ai \
    && chown -R laby:laby /app
USER laby

EXPOSE 8080

# Uses the standard library so image health does not depend on curl/wget. The
# FastAPI lifespan validates required configuration before this can return 200.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD ["python", "-c", "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8080')+'/health', timeout=3).read()"]

# Cloud Run sets $PORT; uvicorn binds to it.
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-8080}"]
