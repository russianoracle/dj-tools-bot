# Dashboard Creation Guide

Пошаговая инструкция по созданию дашбордов в Yandex Cloud Monitoring.

---

## Быстрый старт

1. Открой [Yandex Cloud Monitoring](https://console.yandex.cloud/folders/b1ge0vpe8dp87vc3n73l/monitoring/dashboards)
2. Нажми **"Создать дашборд"**
3. Используй метрики из секций ниже

**Важно**: Все метрики имеют namespace префикс:
- `app.*` — метрики бота (порт 8000)
- `worker.*` — метрики воркера (порт 8001)
- `sys.*` — системные метрики Linux

---

## Dashboard 1: Business Metrics

### Настройки дашборда
- **Название**: DJ Tools Bot - Business Metrics
- **Описание**: Queue, cache, analysis performance, detections

### Виджеты

#### 1. ARQ Queue Depth
**Тип**: Line chart
**Позиция**: Row 1, Left
**Метрика**:
```
app.arq_queue_depth{queue_name="default"}
```
**Описание**: Текущее количество задач в очереди

---

#### 2. ARQ Tasks In Progress
**Тип**: Line chart
**Позиция**: Row 1, Right
**Метрика**:
```
app.arq_in_progress{queue_name="default"}
```
**Описание**: Количество задач в обработке

---

#### 3. Analysis Duration (p50, p95, p99)
**Тип**: Line chart
**Позиция**: Row 2, Full width
**Метрики**:
```
histogram_quantile(0.5, rate(app.analysis_duration_seconds_bucket[5m]))   # p50
histogram_quantile(0.95, rate(app.analysis_duration_seconds_bucket[5m]))  # p95
histogram_quantile(0.99, rate(app.analysis_duration_seconds_bucket[5m]))  # p99
```
**Описание**: Процентили времени анализа треков

---

#### 4. Cache Hit Rate
**Тип**: Line chart
**Позиция**: Row 3, Left
**Метрика**:
```
rate(app.cache_operations_total{operation="get",result="hit"}[5m]) / rate(app.cache_operations_total{operation="get"}[5m])
```
**Описание**: Процент попаданий в кеш
**Y-Axis**: Min: 0, Max: 1

---

#### 5. Cache Operations Rate
**Тип**: Line chart
**Позиция**: Row 3, Right
**Метрики**:
```
rate(app.cache_operations_total{result="hit"}[5m])   # Hits
rate(app.cache_operations_total{result="miss"}[5m])  # Misses
```
**Описание**: Операции с кешем в секунду

---

#### 6. Drops Detected by Zone
**Тип**: Line chart
**Позиция**: Row 4, Left
**Метрика**:
```
rate(app.drops_detected_total[5m])
```
**Описание**: Дропы по зонам энергии

---

#### 7. Transitions Detected
**Тип**: Line chart
**Позиция**: Row 4, Right
**Метрика**:
```
rate(app.transitions_detected_total[5m])
```
**Описание**: Обнаруженные переходы между зонами

---

#### 8. Processing Error Rate
**Тип**: Line chart
**Позиция**: Row 5, Left
**Метрика**:
```
rate(app.processing_errors_total[5m]) * 60
```
**Описание**: Ошибки обработки в минуту

---

#### 9. Telegram Requests Rate
**Тип**: Line chart
**Позиция**: Row 5, Right
**Метрика**:
```
rate(app.telegram_requests_total[5m])
```
**Описание**: Запросы к боту в секунду

---

## Dashboard 2: System Metrics

### Настройки дашборда
- **Название**: DJ Tools Bot - System Metrics
- **Описание**: CPU, memory, disk, network, process stats

### Виджеты

#### 1. CPU Usage
**Тип**: Line chart
**Позиция**: Row 1, Left
**Метрики**:
```
sys.cpu.usage_user      # User
sys.cpu.usage_system    # System
sys.cpu.usage_iowait    # IO Wait
```
**Описание**: Использование CPU
**Y-Axis**: Min: 0, Max: 100

---

#### 2. Memory Usage
**Тип**: Line chart
**Позиция**: Row 1, Right
**Метрики**:
```
(sys.memory.used / sys.memory.total) * 100           # Used %
(sys.memory.swap_used / sys.memory.swap_total) * 100 # Swap %
```
**Описание**: Использование памяти
**Y-Axis**: Min: 0, Max: 100

---

#### 3. Application Memory (Bot + Worker)
**Тип**: Line chart
**Позиция**: Row 2, Left
**Метрики**:
```
app.memory_usage_bytes{process="bot"}       # Bot
worker.memory_usage_bytes{process="worker"} # Worker
```
**Описание**: Память процессов бота и воркера

---

#### 4. Process Resident Memory
**Тип**: Line chart
**Позиция**: Row 2, Right
**Метрики**:
```
app.process_resident_memory_bytes    # Bot
worker.process_resident_memory_bytes # Worker
```
**Описание**: RSS память Python процессов

---

#### 5. Disk Usage
**Тип**: Line chart
**Позиция**: Row 3, Left
**Метрика**:
```
sys.storage.usage_percent
```
**Описание**: Использование диска по разделам
**Y-Axis**: Min: 0, Max: 100

---

#### 6. Disk I/O Operations
**Тип**: Line chart
**Позиция**: Row 3, Right
**Метрики**:
```
sys.io.read_ops   # Read IOPS
sys.io.write_ops  # Write IOPS
```
**Описание**: Операции чтения/записи диска

---

#### 7. Network Traffic
**Тип**: Line chart
**Позиция**: Row 4, Left
**Метрики**:
```
rate(sys.network.bytes_sent[5m])  # Sent
rate(sys.network.bytes_recv[5m])  # Received
```
**Описание**: Сетевой трафик

---

#### 8. Network Errors
**Тип**: Line chart
**Позиция**: Row 4, Right
**Метрики**:
```
rate(sys.network.errors_sent[5m])  # Errors Sent
rate(sys.network.errors_recv[5m])  # Errors Received
rate(sys.network.drops_sent[5m])   # Drops Sent
rate(sys.network.drops_recv[5m])   # Drops Received
```
**Описание**: Сетевые ошибки и потери пакетов

---

#### 9. Python Garbage Collection
**Тип**: Line chart
**Позиция**: Row 5, Left
**Метрики**:
```
rate(app.python_gc_collections_total{generation="0"}[5m])     # Bot Gen 0
rate(worker.python_gc_collections_total{generation="0"}[5m])  # Worker Gen 0
```
**Описание**: Сборка мусора Python

---

#### 10. Process CPU Time
**Тип**: Line chart
**Позиция**: Row 5, Right
**Метрики**:
```
rate(app.process_cpu_seconds_total[5m])    # Bot
rate(worker.process_cpu_seconds_total[5m]) # Worker
```
**Описание**: CPU время процессов

---

## Рекомендуемые алерты

### Critical Alerts

1. **OOM Risk**
   - Метрика: `(sys.memory.used / sys.memory.total) * 100 > 90`
   - Действие: Увеличить память VM или расследовать утечку

2. **Queue Backup**
   - Метрика: `app.arq_queue_depth{queue_name="default"} > 20`
   - Действие: Добавить воркеры или проверить медленные задачи

3. **High Error Rate**
   - Метрика: `rate(app.processing_errors_total[5m]) * 60 > 5`
   - Действие: Проверить логи, расследовать ошибки

4. **Slow Analysis**
   - Метрика: `histogram_quantile(0.95, rate(app.analysis_duration_seconds_bucket[5m])) > 300`
   - Действие: Проверить производительность

### Warning Alerts

1. **Low Cache Hit Rate**
   - Метрика: `rate(app.cache_operations_total{result="hit"}[5m]) / rate(app.cache_operations_total{operation="get"}[5m]) < 0.5`
   - Действие: Проверить конфигурацию кеша

2. **High Disk Usage**
   - Метрика: `sys.storage.usage_percent > 80`
   - Действие: Очистить старые файлы или увеличить диск

---

## Полезные ссылки

- [Metric Explorer](https://console.yandex.cloud/folders/b1ge0vpe8dp87vc3n73l/monitoring/explore)
- [Dashboards](https://console.yandex.cloud/folders/b1ge0vpe8dp87vc3n73l/monitoring/dashboards)
- [Alerts](https://console.yandex.cloud/folders/b1ge0vpe8dp87vc3n73l/monitoring/alert)

---

**Время создания дашборда**: ~10-15 минут
**Количество виджетов**: 19 (9 бизнес + 10 системных)
