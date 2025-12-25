#!/bin/bash
# Queue Quality Metrics for Production (Docker-based)
# Usage: ./scripts/queue-metrics-prod.sh [--detailed]

set -e

echo "🔍 Running queue metrics analysis..."
echo ""

# Find app container with Python
APP_CONTAINER=$(docker ps -q -f name=app)

if [ -z "$APP_CONTAINER" ]; then
    echo "❌ App container not found"
    echo "   Run: docker ps | grep app"
    exit 1
fi

# Run Python metrics script inside app container
# Redis is accessible via docker network at 'redis:6379'
docker exec -e REDIS_HOST=redis -e REDIS_PORT=6379 $APP_CONTAINER \
    python scripts/queue_metrics.py "$@"

echo ""
echo "💡 Tip: Run with --detailed for expired retry analysis"
echo "   ./scripts/queue-metrics-prod.sh --detailed"