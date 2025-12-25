# Готовые запросы для копирования

Скопируй и вставь в поле "Запрос" в Yandex Cloud Monitoring.
**Service всегда: `custom`**

---

## Бизнес-метрики (Bot)

### ARQ Queue Depth
```
"app.arq_queue_depth"{folderId = "b1ge0vpe8dp87vc3n73l", service = "custom", queue_name="default"}
```

### ARQ Tasks In Progress
```
"app.arq_in_progress"{folderId = "b1ge0vpe8dp87vc3n73l", service = "custom", queue_name="default"}
```

### Bot Memory Usage (MB)
```
"app.memory_usage_bytes"{folderId = "b1ge0vpe8dp87vc3n73l", service = "custom", process="bot"} / 1024 / 1024
```

### Worker Memory Usage (MB)
```
"worker.memory_usage_bytes"{folderId = "b1ge0vpe8dp87vc3n73l", service = "custom", process="worker"} / 1024 / 1024
```

### Total App Memory (MB)
```
("app.memory_usage_bytes"{folderId = "b1ge0vpe8dp87vc3n73l", service = "custom", process="bot"} + "worker.memory_usage_bytes"{folderId = "b1ge0vpe8dp87vc3n73l", service = "custom", process="worker"}) / 1024 / 1024
```

### App Info
```
"app.app_info"{folderId = "b1ge0vpe8dp87vc3n73l", service = "custom"}
```

---

## Системные метрики (VM)

### CPU User %
```
"sys.cpu.usage_user"{folderId = "b1ge0vpe8dp87vc3n73l", service = "custom"}
```

### CPU System %
```
"sys.cpu.usage_system"{folderId = "b1ge0vpe8dp87vc3n73l", service = "custom"}
```

### CPU IO Wait %
```
"sys.cpu.usage_iowait"{folderId = "b1ge0vpe8dp87vc3n73l", service = "custom"}
```

### Memory Usage %
```
("sys.memory.used"{folderId = "b1ge0vpe8dp87vc3n73l", service = "custom"} / "sys.memory.total"{folderId = "b1ge0vpe8dp87vc3n73l", service = "custom"}) * 100
```

### Memory Available (GB)
```
"sys.memory.available"{folderId = "b1ge0vpe8dp87vc3n73l", service = "custom"} / 1024 / 1024 / 1024
```

### Swap Usage %
```
("sys.memory.swap_used"{folderId = "b1ge0vpe8dp87vc3n73l", service = "custom"} / "sys.memory.swap_total"{folderId = "b1ge0vpe8dp87vc3n73l", service = "custom"}) * 100
```

### Disk Usage %
```
"sys.storage.usage_percent"{folderId = "b1ge0vpe8dp87vc3n73l", service = "custom"}
```

### Disk Free (GB)
```
("sys.storage.total_bytes"{folderId = "b1ge0vpe8dp87vc3n73l", service = "custom"} - "sys.storage.used_bytes"{folderId = "b1ge0vpe8dp87vc3n73l", service = "custom"}) / 1024 / 1024 / 1024
```

### Network Sent (MB/s)
```
"sys.network.bytes_sent"{folderId = "b1ge0vpe8dp87vc3n73l", service = "custom"} / 1024 / 1024
```

### Network Received (MB/s)
```
"sys.network.bytes_recv"{folderId = "b1ge0vpe8dp87vc3n73l", service = "custom"} / 1024 / 1024
```

### Network Errors (sent)
```
"sys.network.errors_sent"{folderId = "b1ge0vpe8dp87vc3n73l", service = "custom"}
```

### Disk Read IOPS
```
"sys.io.read_ops"{folderId = "b1ge0vpe8dp87vc3n73l", service = "custom"}
```

### Disk Write IOPS
```
"sys.io.write_ops"{folderId = "b1ge0vpe8dp87vc3n73l", service = "custom"}
```

---

## Python процессы

### Bot CPU Usage
```
"app.process_cpu_seconds_total"{folderId = "b1ge0vpe8dp87vc3n73l", service = "custom"}
```

### Worker CPU Usage
```
"worker.process_cpu_seconds_total"{folderId = "b1ge0vpe8dp87vc3n73l", service = "custom"}
```

### Bot RSS Memory (MB)
```
"app.process_resident_memory_bytes"{folderId = "b1ge0vpe8dp87vc3n73l", service = "custom"} / 1024 / 1024
```

### Worker RSS Memory (MB)
```
"worker.process_resident_memory_bytes"{folderId = "b1ge0vpe8dp87vc3n73l", service = "custom"} / 1024 / 1024
```

### Bot Virtual Memory (MB)
```
"app.process_virtual_memory_bytes"{folderId = "b1ge0vpe8dp87vc3n73l", service = "custom"} / 1024 / 1024
```

### Python GC Gen 0 (Bot)
```
"app.python_gc_collections_total"{folderId = "b1ge0vpe8dp87vc3n73l", service = "custom", generation="0"}
```

### Python GC Gen 1 (Bot)
```
"app.python_gc_collections_total"{folderId = "b1ge0vpe8dp87vc3n73l", service = "custom", generation="1"}
```

### Python GC Gen 2 (Bot)
```
"app.python_gc_collections_total"{folderId = "b1ge0vpe8dp87vc3n73l", service = "custom", generation="2"}
```

---

## Составные метрики

### Total CPU Usage (User + System)
```
"sys.cpu.usage_user"{folderId = "b1ge0vpe8dp87vc3n73l", service = "custom"} + "sys.cpu.usage_system"{folderId = "b1ge0vpe8dp87vc3n73l", service = "custom"}
```

### Free Memory (GB)
```
("sys.memory.total"{folderId = "b1ge0vpe8dp87vc3n73l", service = "custom"} - "sys.memory.used"{folderId = "b1ge0vpe8dp87vc3n73l", service = "custom"}) / 1024 / 1024 / 1024
```

### Network Total (MB/s)
```
("sys.network.bytes_sent"{folderId = "b1ge0vpe8dp87vc3n73l", service = "custom"} + "sys.network.bytes_recv"{folderId = "b1ge0vpe8dp87vc3n73l", service = "custom"}) / 1024 / 1024
```

### Total App CPU
```
"app.process_cpu_seconds_total"{folderId = "b1ge0vpe8dp87vc3n73l", service = "custom"} + "worker.process_cpu_seconds_total"{folderId = "b1ge0vpe8dp87vc3n73l", service = "custom"}
```

---

## Примеры дашбордов

### Минимальный (3 виджета)
1. **Queue Depth**: `"app.arq_queue_depth"{folderId = "b1ge0vpe8dp87vc3n73l", service = "custom", queue_name="default"}`
2. **Memory %**: `("sys.memory.used"{folderId = "b1ge0vpe8dp87vc3n73l", service = "custom"} / "sys.memory.total"{folderId = "b1ge0vpe8dp87vc3n73l", service = "custom"}) * 100`
3. **CPU %**: `"sys.cpu.usage_user"{folderId = "b1ge0vpe8dp87vc3n73l", service = "custom"} + "sys.cpu.usage_system"{folderId = "b1ge0vpe8dp87vc3n73l", service = "custom"}`
