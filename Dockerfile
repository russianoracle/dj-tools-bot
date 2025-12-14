# ============================================================================
# Multi-stage Dockerfile для DJ Tools Bot
# Оптимизирован для быстрой сборки и компактного размера
# ============================================================================

# syntax=docker/dockerfile:1.4 - включить BuildKit features
ARG PYTHON_VERSION=3.12

# ============================================================================
# STAGE 1: Build Dependencies (builder image)
# ============================================================================
FROM python:${PYTHON_VERSION}-slim AS builder

WORKDIR /build

# Установка build-time зависимостей (один слой)
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libsndfile1-dev \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Копируем ТОЛЬКО production requirements (кэш layer)
COPY requirements-prod.txt .

# Собираем wheels с кэшем pip (BuildKit cache mount)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip wheel --no-cache-dir --wheel-dir /wheels -r requirements-prod.txt

# ============================================================================
# STAGE 2: Runtime Image (final minimal image)
# ============================================================================
FROM python:${PYTHON_VERSION}-slim AS runtime

WORKDIR /app

# Runtime зависимости (один слой, atomic)
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 \
        ffmpeg \
        curl \
        redis-tools \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Копируем wheels из builder и устанавливаем (один слой)
COPY --from=builder /wheels /wheels
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir --no-deps /wheels/* \
    && rm -rf /wheels

# Создаём non-root user (security best practice)
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/cache /app/downloads && \
    chown -R appuser:appuser /app

# Копируем код (Clean Architecture)
# ВАЖНО: копируем только app/, БЕЗ src/, training/, experiments/
COPY --chown=appuser:appuser app/ ./app/

# Копируем production models (если есть)
RUN mkdir -p ./models/production
COPY --chown=appuser:appuser models/production/ ./models/production/ 2>/dev/null || true

# Switch to non-root
USER appuser

# Environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import redis; r = redis.from_url('${REDIS_URL:-redis://redis:6379/0}'); r.ping()" || exit 1

# Entry point
CMD ["python", "-m", "app.main"]

# ============================================================================
# Build Instructions:
# ============================================================================
# Local build (с кэшем):
#   DOCKER_BUILDKIT=1 docker build -t mood-classifier:latest .
#
# Push to Yandex Registry:
#   docker tag mood-classifier:latest cr.yandex/YOUR_REGISTRY_ID/mood-classifier:latest
#   docker push cr.yandex/YOUR_REGISTRY_ID/mood-classifier:latest
#
# Размер итогового образа: ~800MB (vs ~15GB с tensorflow/torch)
# ============================================================================
