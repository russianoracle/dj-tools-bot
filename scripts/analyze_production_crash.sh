#!/bin/bash
# Точная диагностика причины краша worker
# Собирает прямые доказательства, а не догадки
# Usage: ssh -i ~/.ssh/tender-bot-key ubuntu@158.160.122.216 'bash -s' < scripts/analyze_production_crash.sh

echo "=========================================="
echo "FORENSIC ANALYSIS: Worker Crash Root Cause"
echo "=========================================="
echo ""

WORKER_CONTAINER="mood-arq-worker"
TIMEFRAME="2025-12-23T13:44:00"  # Время краша из логов

echo "🔍 1. ПРЯМОЕ ДОКАЗАТЕЛЬСТВО: OOM Kill Events"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Проверяем kernel logs на OOM killer:"
sudo dmesg -T | grep -A 10 -B 2 "Out of memory\|oom-kill\|Killed process" | grep -A 10 -B 2 "python\|arq" || echo "✅ OOM kill НЕ найден"
echo ""

echo "🔍 2. DOCKER EVENTS: Почему контейнер перезапустился?"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Ищем события около $TIMEFRAME:"
docker events --since "$TIMEFRAME" --until "2025-12-23T13:46:00" \
  --filter container=$WORKER_CONTAINER \
  --filter type=container \
  --format 'time={{.Time}} status={{.Status}} {{.Actor.Attributes}}' 2>/dev/null || echo "⚠️  Нет доступа к Docker events (слишком старые)"
echo ""

echo "🔍 3. CONTAINER EXIT CODE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
docker inspect $WORKER_CONTAINER --format '{{json .State}}' | python3 -c "
import json, sys
state = json.load(sys.stdin)
print(f\"Status: {state.get('Status')}\")
print(f\"Exit Code: {state.get('ExitCode')} {'(normal exit)' if state.get('ExitCode') == 0 else '(abnormal exit)'}\")
print(f\"OOMKilled: {state.get('OOMKilled')} {'☠️ CONFIRMED OOM' if state.get('OOMKilled') else ''}\")
print(f\"Started At: {state.get('StartedAt')}\")
print(f\"Finished At: {state.get('FinishedAt')}\")
" 2>&1
echo ""

echo "🔍 4. ПРОВЕРКА: Применилась ли оптимизация?"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
docker exec $WORKER_CONTAINER python -c "
import inspect
from app.common.primitives.stft import _stft_numpy

src = inspect.getsource(_stft_numpy)
has_complex64 = 'astype(np.complex64)' in src

if has_complex64:
    print('✅ Оптимизация ПРИМЕНЕНА (complex64 conversion found)')
else:
    print('❌ Оптимизация НЕ ПРИМЕНЕНА')

# Показать последние строки функции
lines = [l for l in src.split('\n') if l.strip()]
print('\nПоследние 3 строки _stft_numpy():')
for line in lines[-3:]:
    print(f'  {line}')
" 2>&1
echo ""

echo "🔍 5. WORKER LOGS: Последние сообщения перед крашем"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Ищем логи около 13:45:13 - 13:45:27:"
docker logs $WORKER_CONTAINER --since "2025-12-23T13:44:00" --until "2025-12-23T13:46:00" 2>&1 | tail -50
echo ""

echo "🔍 6. HEALTHCHECK LOGS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
docker inspect $WORKER_CONTAINER --format '{{json .State.Health}}' | python3 -c "
import json, sys
try:
    health = json.load(sys.stdin)
    if health:
        print(f\"Status: {health.get('Status')}\")
        print(f\"Failing Streak: {health.get('FailingStreak')}\")
        logs = health.get('Log', [])
        if logs:
            print(f'\nПоследние {len(logs)} healthcheck результатов:')
            for log in logs[-3:]:
                print(f\"  {log.get('Start')}: {log.get('ExitCode')} - {log.get('Output', '')[:100]}\")
    else:
        print('Healthcheck не настроен')
except:
    print('Нет данных healthcheck')
" 2>&1
echo ""

echo "🔍 7. MEMORY STATS В МОМЕНТ КРАША"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
docker stats $WORKER_CONTAINER --no-stream --format "Current: {{.MemUsage}} ({{.MemPerc}})"
docker inspect $WORKER_CONTAINER --format 'Limit: {{.HostConfig.Memory}}' | python3 -c "
import sys
mem = int(sys.stdin.read().strip() or 0)
if mem > 0:
    print(f'{mem / 1024 / 1024 / 1024:.1f} GB')
else:
    print('No limit')
"
echo ""

echo "🔍 8. ПРОВЕРКА max_jobs"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
docker exec $WORKER_CONTAINER python -c "
from app.services.arq_worker import WorkerSettings
print(f'max_jobs = {WorkerSettings.max_jobs}')
print(f'job_timeout = {WorkerSettings.job_timeout} sec')

import asyncio
from arq import create_pool
from app.core.config import get_redis_host, get_redis_port

async def check():
    try:
        redis = await create_pool({'host': get_redis_host(), 'port': get_redis_port()})
        in_progress = await redis.zcard('arq:in-progress')
        queue_len = await redis.llen('arq:queue')
        print(f'Задач в обработке: {in_progress}')
        print(f'Задач в очереди: {queue_len}')
        await redis.close()
    except Exception as e:
        print(f'Ошибка проверки Redis: {e}')

asyncio.run(check())
" 2>&1
echo ""

echo "=========================================="
echo "SUMMARY"
echo "=========================================="
echo ""
echo "Проверьте разделы 1 (OOM kill) и 3 (OOMKilled flag)."
echo "Это прямые доказательства причины краша."
echo ""