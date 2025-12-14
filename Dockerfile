# Multi-stage build for DJ Tools Bot
# Clean Architecture: app/ is the primary production code

# Stage 1: Build dependencies
FROM python:3.12-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1-dev \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

# Stage 2: Runtime image
FROM python:3.12-slim as runtime

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    curl \
    redis-tools \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy wheels from builder
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/cache /app/downloads && \
    chown -R appuser:appuser /app

# Copy Clean Architecture production code
COPY --chown=appuser:appuser app/ ./app/

# Copy production models (if they exist)
RUN mkdir -p ./models/production
COPY --chown=appuser:appuser models/production/ ./models/production/ 2>/dev/null || true

# Switch to non-root user
USER appuser

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

# Health check - verify Redis connection
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import redis; r = redis.from_url('${REDIS_URL:-redis://redis:6379/0}'); r.ping()" || exit 1

# Default command: run bot using Clean Architecture entry point
CMD ["python", "-m", "app.main"]