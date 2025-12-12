# Multi-stage build for mood-classifier
# Stage 1: Build dependencies
FROM python:3.12-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1-dev \
    ffmpeg \
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
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy wheels from builder
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/cache /app/downloads && \
    chown -R appuser:appuser /app

# Copy application code
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser config/ ./config/
COPY --chown=appuser:appuser models/production/ ./models/production/

# Switch to non-root user
USER appuser

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import src; print('OK')" || exit 1

# Default command (overridden by docker-compose)
CMD ["python", "-m", "src.services.bot"]
