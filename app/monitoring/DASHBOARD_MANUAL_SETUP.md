# Создание дашборда Prometheus вручную

JSON импорт не работает в Yandex Monitoring — создавай через UI.

## Шаг 1: Создай дашборд

1. https://monitoring.yandex.cloud/folders/b1ge0vpe8dp87vc3n73l/monitoring/dashboards
2. "Создать дашборд"
3. Название: **DJ Tools Bot - Prometheus**

## Шаг 2: Добавь виджеты

### Widget 1: ARQ Queue Depth

**Запрос:**
```promql
arq_queue_depth{job="bot",queue_name="default"}
arq_queue_depth{job="worker",queue_name="default"}
```

**Настройки:**
- Тип: Линейный график
- Название: ARQ Queue Depth
- Агрегация: Нет

### Widget 2: Tasks In Progress

**Запрос:**
```promql
arq_in_progress{queue_name="default"}
```

**Настройки:**
- Тип: Линейный график
- Название: Tasks In Progress

### Widget 3: Task Processing Rate

**Запрос:**
```promql
rate(arq_tasks_total[5m])
```

**Настройки:**
- Тип: Линейный график
- Название: Task Processing Rate (tasks/sec)

### Widget 4: Bot Memory

**Запрос:**
```promql
memory_usage_bytes{job="bot"}
```

**Настройки:**
- Тип: Линейный график
- Название: Bot Memory Usage
- Единицы: bytes

### Widget 5: Worker Memory

**Запрос:**
```promql
memory_usage_bytes{job="worker"}
```

**Настройки:**
- Тип: Линейный график
- Название: Worker Memory Usage
- Единицы: bytes

### Widget 6: CPU Usage

**Запрос:**
```promql
rate(process_cpu_seconds_total{job="bot"}[5m])
rate(process_cpu_seconds_total{job="worker"}[5m])
```

**Настройки:**
- Тип: Линейный график
- Название: CPU Usage
- Агрегация: Нет

### Widget 7: Python GC

**Запрос:**
```promql
rate(python_gc_collections_total[5m])
```

**Настройки:**
- Тип: Линейный график
- Название: Python GC Collections/sec

## Шаг 3: Сохрани

Нажми "Сохранить" — дашборд готов.

## Полезные запросы

### Memory в MB
```promql
memory_usage_bytes{job="bot"} / 1024 / 1024
```

### Task queue % заполненности (если есть лимит)
```promql
(arq_in_progress / 10) * 100
```

### Средняя длина очереди за последний час
```promql
avg_over_time(arq_queue_depth{queue_name="default"}[1h])
```

## Troubleshooting

### "Нет данных"
- Проверь что Prometheus запущен: `docker ps | grep prometheus`
- Проверь метрики: `curl http://158.160.122.216:9090/api/v1/query?query=up`
- Подожди 1-2 минуты после создания виджета

### Метрика не найдена
- Проверь workspace ID в prometheus.yml
- Проверь remote_write логи: `docker logs mood-prometheus | grep remote`

### Данные старые
- Добавь автообновление: настройки дашборда → Auto-refresh → 30s
