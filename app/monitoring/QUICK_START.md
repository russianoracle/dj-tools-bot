# Мониторинг — Быстрый старт

Инструкция как увидеть метрики в Yandex Cloud Monitoring.

---

## Как работает Yandex Cloud Monitoring

**Важно понять:**
1. Unified Agent собирает метрики с `namespace` (app, worker, sys)
2. В Yandex Cloud namespace становится **префиксом имени метрики** (`app.arq_queue_depth`)
3. Все метрики имеют `service = "custom"`
4. Имена метрик **обязательно в кавычках**

---

## Создание графика (пошагово)

### Шаг 1: Открой дашборд
https://console.yandex.cloud/folders/b1ge0vpe8dp87vc3n73l/monitoring/dashboards

Нажми **"Создать дашборд"**

---

### Шаг 2: Добавь виджет
Нажми **"Добавить виджет"** → **"График"**

---

### Шаг 3: Выбери service
В поле **"service"** ВСЕГДА выбирай: **`custom`**

(Все метрики от Unified Agent идут с service=custom)

---

### Шаг 4: Введи запрос

**С префиксом namespace И в кавычках!**

#### Метрики бота (namespace = app)
```
"app.arq_queue_depth"{queue_name="default"}
"app.arq_in_progress"{queue_name="default"}
"app.memory_usage_bytes"{process="bot"}
"app.process_resident_memory_bytes"
```

#### Метрики воркера (namespace = worker)
```
"worker.arq_queue_depth"{queue_name="default"}
"worker.memory_usage_bytes"{process="worker"}
```

#### Системные метрики (namespace = sys)
```
"sys.cpu.usage_user"
"sys.cpu.usage_system"
"sys.memory.used"
"sys.memory.total"
"sys.storage.usage_percent"
"sys.network.bytes_sent"
```

---

## Примеры готовых виджетов

### 1. ARQ Queue Depth
- **Service**: `custom`
- **Запрос**: `"app.arq_queue_depth"{queue_name="default"}`
- **Название**: ARQ Queue Depth

### 2. Память бота
- **Service**: `custom`
- **Запрос**: `"app.memory_usage_bytes"{process="bot"}`
- **Название**: Bot Memory

### 3. Память воркера
- **Service**: `custom`
- **Запрос**: `"worker.memory_usage_bytes"{process="worker"}`
- **Название**: Worker Memory

### 4. CPU Usage
- **Service**: `custom`
- **Запросы** (добавь несколько):
  ```
  "sys.cpu.usage_user"
  "sys.cpu.usage_system"
  ```
- **Название**: CPU Usage

### 5. Memory Usage
- **Service**: `custom`
- **Запрос**: `("sys.memory.used" / "sys.memory.total") * 100`
- **Название**: Memory %

---

## Полный список метрик

### Namespace: app (бот, service=custom)
```
"app.arq_queue_depth"
"app.arq_in_progress"
"app.arq_tasks_total"
"app.memory_usage_bytes"
"app.app_info"
"app.process_resident_memory_bytes"
"app.process_virtual_memory_bytes"
"app.process_cpu_seconds_total"
"app.python_gc_collections_total"
"app.python_gc_objects_collected_total"
```

### Namespace: worker (воркер, service=custom)
```
"worker.arq_queue_depth"
"worker.arq_in_progress"
"worker.arq_tasks_total"
"worker.memory_usage_bytes"
"worker.app_info"
"worker.process_resident_memory_bytes"
"worker.process_virtual_memory_bytes"
"worker.process_cpu_seconds_total"
"worker.python_gc_collections_total"
```

### Namespace: sys (система, service=custom)
```
"sys.cpu.usage_user"
"sys.cpu.usage_system"
"sys.cpu.usage_iowait"
"sys.cpu.usage_idle"
"sys.memory.total"
"sys.memory.used"
"sys.memory.available"
"sys.memory.swap_total"
"sys.memory.swap_used"
"sys.storage.usage_percent"
"sys.storage.total_bytes"
"sys.storage.used_bytes"
"sys.network.bytes_sent"
"sys.network.bytes_recv"
"sys.network.errors_sent"
"sys.network.errors_recv"
"sys.io.read_ops"
"sys.io.write_ops"
```

---

## Если метрик нет

1. **Проверь что контейнеры работают:**
   ```bash
   ssh ubuntu@158.160.122.216 "docker ps"
   ```

2. **Проверь endpoints:**
   ```bash
   ssh ubuntu@158.160.122.216 "curl http://localhost:8000/metrics | head -30"
   ssh ubuntu@158.160.122.216 "curl http://localhost:8001/metrics | head -30"
   ```

3. **Проверь Unified Agent:**
   ```bash
   ssh ubuntu@158.160.122.216 "sudo systemctl status unified-agent"
   ssh ubuntu@158.160.122.216 "sudo journalctl -u unified-agent -n 50"
   ```

4. **Подожди 1-2 минуты** — метрики собираются не моментально

---

## Troubleshooting

### "Нет данных"
- Подожди 1-2 минуты после создания виджета
- Проверь что выбран правильный service
- Проверь что временной диапазон = "Последний час"

### "Parsing error"
- Убери префикс `app.`, `worker.`, `sys.` из запроса
- Service выбирается в UI, не в запросе!

### Метрика не появляется в списке
- Введи её вручную в поле "Запрос"
- Автокомплит может не показывать все метрики

---

**Время создания первого графика:** 2 минуты
**Полный дашборд:** 10-15 минут
