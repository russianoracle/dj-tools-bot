#!/bin/bash
# Локальный запуск CI тестов с Redis
# Usage: ./scripts/run_ci_tests_local.sh [--keep-redis]

set -e

KEEP_REDIS=false
if [[ "$1" == "--keep-redis" ]]; then
    KEEP_REDIS=true
fi

USE_DOCKER=false
REDIS_STARTED_BY_SCRIPT=false

echo "=== Локальный запуск CI тестов ==="
echo ""

# Проверяем Python зависимости
echo "[1/4] Проверка Python зависимостей..."
python3 -c "import pytest, redis" 2>/dev/null || {
    echo "⚠️  Устанавливаем pytest и redis..."
    pip install pytest pytest-asyncio redis
}
echo "✓ Python зависимости OK"
echo ""

# Проверяем, запущен ли Docker
echo "[2/4] Проверка Redis..."
if docker info &>/dev/null; then
    echo "✓ Docker запущен, используем Docker Compose"
    USE_DOCKER=true

    # Запускаем Redis в Docker
    docker compose up -d redis 2>/dev/null || docker-compose up -d redis
    sleep 2

    # Ждём, пока Redis будет готов
    echo "Ожидание Redis в Docker..."
    for i in {1..10}; do
        if docker exec mood-classifier-redis redis-cli ping &>/dev/null; then
            echo "✓ Redis готов (Docker)"
            REDIS_STARTED_BY_SCRIPT=true
            break
        fi
        if [ $i -eq 10 ]; then
            echo "❌ Redis не ответил через 10 секунд"
            exit 1
        fi
        sleep 1
    done
else
    echo "⚠️  Docker не запущен, проверяем локальный Redis..."

    # Проверяем, запущен ли Redis локально
    if redis-cli ping &>/dev/null; then
        echo "✓ Redis уже запущен локально (порт 6379)"
    elif command -v redis-server &>/dev/null; then
        echo "⚠️  Запускаем локальный redis-server..."
        redis-server --daemonize yes --port 6379
        sleep 1
        REDIS_STARTED_BY_SCRIPT=true
        echo "✓ Redis запущен локально"
    else
        echo "❌ ERROR: Redis не найден!"
        echo ""
        echo "Установите Redis одним из способов:"
        echo "  - macOS: brew install redis && brew services start redis"
        echo "  - Linux: sudo apt install redis-server && sudo systemctl start redis"
        echo "  - Или запустите Docker Desktop и используйте docker compose"
        exit 1
    fi
fi
echo ""

# Устанавливаем переменную окружения для тестов
export REDIS_URL=redis://localhost:6379/0

# Запускаем тесты
echo "[3/4] Запуск CI тестов..."
echo ""
echo "=== test_ci_bot.py (Bot components) ==="
pytest tests/test_ci_bot.py -v --tb=short
echo ""

echo "=== test_ci_cache_redis.py (Redis integration) ==="
pytest tests/test_ci_cache_redis.py -v --tb=short
echo ""

echo "=== test_ci_pipelines.py (Real audio analysis) ==="
pytest tests/test_ci_pipelines.py -v --tb=short
echo ""

# Останавливаем Redis (если не указан --keep-redis)
cleanup_redis() {
    if [ "$KEEP_REDIS" = false ] && [ "$REDIS_STARTED_BY_SCRIPT" = true ]; then
        echo ""
        echo "[4/4] Остановка Redis..."
        if [ "$USE_DOCKER" = true ]; then
            docker compose stop redis 2>/dev/null || docker-compose stop redis
        else
            redis-cli shutdown 2>/dev/null || true
        fi
        echo "✓ Redis остановлен"
    elif [ "$KEEP_REDIS" = true ]; then
        echo ""
        echo "[4/4] Redis оставлен запущенным (--keep-redis)"
    fi
}

# Гарантированная остановка Redis даже при ошибках
trap cleanup_redis EXIT

echo ""
echo "=== ✅ Все CI тесты завершены ==="