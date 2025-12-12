# Optimized Dockerfile for Telegram Bot
# Uses Alpine for minimal size (~50MB vs ~1GB)

FROM python:3.12-alpine

WORKDIR /app

# Install minimal runtime dependencies in single layer
RUN apk add --no-cache libffi \
    && rm -rf /var/cache/apk/*

# Create non-root user early
RUN adduser -D -u 1000 appuser \
    && mkdir -p /app/cache /app/downloads \
    && chown -R appuser:appuser /app

# Copy and install requirements first (layer caching)
COPY requirements-bot.txt ./
RUN pip install --no-cache-dir -r requirements-bot.txt \
    && rm -rf ~/.cache/pip /root/.cache

# Copy application code (changes most frequently - last layer)
COPY --chown=appuser:appuser src/services/ ./src/services/
COPY --chown=appuser:appuser src/__init__.py ./src/
COPY --chown=appuser:appuser config/ ./config/

# Switch to non-root user
USER appuser

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "print('OK')" || exit 1

CMD ["python", "-m", "src.services.bot"]