# ── Stage 1: build dependencies ──────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt

# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY app/       app/
COPY pipeline/  pipeline/
COPY data/taxonomy/ data/taxonomy/
COPY scripts/   scripts/

# Pre-build the FAISS index at image build time (uses mock embeddings)
RUN python scripts/build_embeddings.py --mock

# Runtime configuration
ENV USE_MOCK_LLM=true \
    DATA_DIR=/app/data \
    PYTHONUNBUFFERED=1

# Job data lives outside the image layer so it can be mounted as a volume
VOLUME ["/app/data/jobs"]

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
