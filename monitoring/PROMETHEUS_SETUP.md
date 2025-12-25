# Prometheus Setup Guide

Проброс метрик в Yandex Cloud Managed Prometheus настроен автоматически.

## Архитектура

```
Bot:8000 ──┐
           ├──> Prometheus ──> remote_write ──> Yandex Managed Prometheus
Worker:8001┘
```

## Компоненты

### 1. Prometheus Server
- **Образ**: `prom/prometheus:v2.54.1`
- **Порт**: `9090` (Web UI + API)
- **Конфигурация**: `monitoring/prometheus.yml`

### 2. Scrape Targets
- **Bot metrics**: `http://bot:8000/metrics`
- **Worker metrics**: `http://worker:8001/metrics`
- **Интервал**: 15 секунд

### 3. Remote Write
- **Endpoint**: `https://monitoring.api.cloud.yandex.net/prometheus/workspaces/monhr2m0jl118pvk9mtu/api/v1/write`
- **Авторизация**: Bearer token (API key из Lockbox)
- **Батчинг**: 2000 samples per send

## Развертывание

### Шаг 1: Обновить секреты на VM

```bash
ssh ubuntu@158.160.122.216
cd /home/ubuntu/app
sudo -E ./scripts/fetch-secrets.sh --env-file .env
```

Проверь что `PROMETHEUS_API_KEY` появился в `.env`:
```bash
grep PROMETHEUS_API_KEY .env
```

### Шаг 2: Запустить стек с Prometheus

```bash
# Локально (для тестирования)
make start

# На продакшене (через деплой)
make deploy-full
```

### Шаг 3: Проверить работу Prometheus

**Локально:**
```bash
# Проверить что контейнер запустился
docker ps | grep prometheus

# Проверить логи
docker logs mood-prometheus

# Открыть Web UI
open http://localhost:9090
```

**На сервере:**
```bash
# Проверить контейнер
ssh ubuntu@158.160.122.216 "docker ps | grep prometheus"

# Проверить логи remote_write
ssh ubuntu@158.160.122.216 "docker logs mood-prometheus 2>&1 | grep -i 'remote_write\|error'"

# Проверить что метрики собираются
ssh ubuntu@158.160.122.216 "curl -s http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | {job, health, lastError}'"
```

### Шаг 4: Проверить метрики в Yandex Cloud

1. Открой Prometheus Workspace:
   https://console.yandex.cloud/folders/b1ge0vpe8dp87vc3n73l/monitoring/prometheus

2. Перейди в раздел "Explore" или "Query"

3. Выполни тестовый запрос:
   ```promql
   arq_queue_depth{job="bot"}
   ```

4. Должны появиться данные с labels:
   - `cluster=mood-classifier`
   - `env=production`
   - `job=bot` или `job=worker`

## Создание дашбордов

### Автоматический импорт

1. Открой Yandex Monitoring:
   https://console.yandex.cloud/folders/b1ge0vpe8dp87vc3n73l/monitoring/dashboards

2. Нажми "Создать дашборд"

3. В меню дашборда выбери "Загрузить из JSON"

4. Загрузи файл `monitoring/dashboard-prometheus.json`

### Ручное создание

См. примеры запросов ниже.

## Доступные метрики

### ARQ Queue Metrics
```promql
# Queue depth (bot)
arq_queue_depth{job="bot",queue_name="default"}

# Queue depth (worker)
arq_queue_depth{job="worker",queue_name="default"}

# Tasks in progress
arq_in_progress{queue_name="default"}

# Task processing rate (tasks/sec)
rate(arq_tasks_total[5m])
```

### Memory Metrics
```promql
# Bot memory usage
memory_usage_bytes{job="bot"}

# Worker memory usage
memory_usage_bytes{job="worker"}

# Process resident memory
process_resident_memory_bytes{job="bot"}
process_resident_memory_bytes{job="worker"}
```

### CPU Metrics
```promql
# CPU usage rate (bot)
rate(process_cpu_seconds_total{job="bot"}[5m])

# CPU usage rate (worker)
rate(process_cpu_seconds_total{job="worker"}[5m])
```

### Python GC Metrics
```promql
# GC collections rate
rate(python_gc_collections_total[5m])

# GC objects collected
rate(python_gc_objects_collected_total[5m])
```

## Troubleshooting

### Prometheus не запускается

```bash
# Проверь логи
docker logs mood-prometheus

# Проверь конфигурацию
docker exec mood-prometheus promtool check config /etc/prometheus/prometheus.yml

# Проверь что API ключ установлен
docker exec mood-prometheus cat /etc/prometheus/api_key.txt
```

### Remote write errors

**401 Unauthorized:**
- Проверь что `PROMETHEUS_API_KEY` правильный
- Проверь что у сервисного аккаунта есть роль `monitoring.editor`

**404 Not Found:**
- Проверь что workspace ID правильный в `prometheus.yml`
- URL должен быть: `https://monitoring.api.cloud.yandex.net/prometheus/workspaces/monhr2m0jl118pvk9mtu/api/v1/write`

**429 Too Many Requests:**
- Увеличь `min_backoff` в `queue_config`
- Уменьши `max_samples_per_send`

### Метрики не появляются в Yandex Cloud

1. Проверь что Prometheus собирает метрики:
   ```bash
   curl http://localhost:9090/api/v1/query?query=arq_queue_depth
   ```

2. Проверь статус remote_write:
   ```bash
   curl http://localhost:9090/api/v1/status/tsdb
   ```

3. Подожди 1-2 минуты - метрики батчатся перед отправкой

## Полезные команды

### Локальная разработка

```bash
# Перезагрузить конфигурацию без перезапуска
curl -X POST http://localhost:9090/-/reload

# Проверить targets
curl http://localhost:9090/api/v1/targets | jq

# Выполнить запрос
curl 'http://localhost:9090/api/v1/query?query=up' | jq
```

### Продакшен

```bash
# Проверить health
ssh ubuntu@158.160.122.216 "curl -s http://localhost:9090/-/healthy"

# Reload конфигурации
ssh ubuntu@158.160.122.216 "curl -X POST http://localhost:9090/-/reload"

# Проверить remote_write queue
ssh ubuntu@158.160.122.216 "curl -s http://localhost:9090/api/v1/status/walreplay"
```

## Безопасность

- API ключ хранится в Lockbox
- Передается через переменную окружения `PROMETHEUS_API_KEY`
- Записывается в `/etc/prometheus/api_key.txt` с правами `600`
- НЕ коммитится в git
- НЕ логируется в stdout/stderr

## Архитектура хранения

### Prometheus TSDB
- **Volume**: `prometheus_data`
- **Retention**: по умолчанию 15 дней
- **WAL**: для надежности remote_write
- **Персистентность**: данные сохраняются при перезапуске

### Yandex Managed Prometheus
- **Retention**: 90 дней (настраивается в workspace)
- **Репликация**: автоматическая
- **Резервные копии**: встроенные
