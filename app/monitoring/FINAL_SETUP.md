# Финальная конфигурация Prometheus + Yandex Monitoring

## Текущее состояние

### ✅ Выполнено

1. **Сервисный аккаунт** `prometheus-writer` создан с ролью `monitoring.editor`
2. **API ключ** создан и сохранен в Lockbox (`PROMETHEUS_API_KEY`)
3. **Prometheus remote_write** настроен на workspace `monhr2m0jl118pvk9mtu`
4. **Prometheus remote_read** настроен с `X-Lookback-Delta: 5m`
5. **Стандартные коллекторы** добавлены (ProcessCollector, GCCollector, PlatformCollector)
6. **33 custom метрики** определены в `app/core/monitoring/metrics.py`

### Конфигурация Prometheus

```yaml
# app/monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: 'mood-classifier'
    env: 'production'

scrape_configs:
  - job_name: 'bot'
    static_configs:
      - targets: ['bot:8000']

  - job_name: 'worker'
    static_configs:
      - targets: ['worker:8001']

remote_write:
  - url: 'https://monitoring.api.cloud.yandex.net/prometheus/workspaces/monhr2m0jl118pvk9mtu/api/v1/write'
    authorization:
      type: Bearer
      credentials_file: /etc/prometheus/api_key.txt

remote_read:
  - url: 'https://monitoring.api.cloud.yandex.net/prometheus/workspaces/monhr2m0jl118pvk9mtu/api/v1/read'
    authorization:
      type: Bearer
      credentials_file: /etc/prometheus/api_key.txt
    headers:
      X-Lookback-Delta: 5m
    read_recent: true
```

### Формат запросов в дашбордах

**ПРАВИЛЬНО (PromQL для Prometheus workspace):**
```hcl
queries {
  target {
    query     = "arq_queue_depth"
    text_mode = false
  }
}
```

**НЕПРАВИЛЬНО (Yandex Monitoring Query Language - для нативных метрик):**
```hcl
queries {
  target {
    query     = "\"arq_queue_depth\"{folderId=\"...\", service=\"...\"}"
    text_mode = true
  }
}
```

## Деплой

### 1. Деплой конфигурации на сервер

```bash
make deploy
```

Это обновит:
- `app/monitoring/prometheus.yml`
- `app/core/monitoring/server.py` (с коллекторами)
- `app/core/monitoring/metrics.py`

### 2. Создание дашбордов через Terraform

```bash
cd app/monitoring/terraform
terraform init
terraform apply -auto-approve
```

Будут созданы дашборды:
- `prometheus-metrics-dashboard` (базовый, 7 виджетов)
- `dj-tools-comprehensive` (полный, 18 виджетов)
- `simple-test` (тестовый, 4 виджета)

### 3. Проверка метрик

После деплоя подождите **5-10 минут** для:
- Рестарта Prometheus
- Сбора метрик
- Отправки в workspace
- Появления в дашбордах

## Troubleshooting

### Дашборды показывают "No data"

**Причина:** Метрики еще не дошли до workspace.

**Решение:**
1. Проверьте логи Prometheus:
   ```bash
   ssh ubuntu@158.160.122.216 "docker logs mood-prometheus --tail 50 | grep -i 'remote_write\|error'"
   ```

2. Проверьте, что метрики экспортируются:
   ```bash
   curl http://158.160.122.216:8000/metrics | grep arq_queue_depth
   ```

3. Подождите 10-15 минут после рестарта

### Ошибка "variable not found"

**Причина:** Используется неправильный формат запроса (Yandex Query Language вместо PromQL).

**Решение:** Убедитесь, что `text_mode = false` и query без кавычек.

### Remote write ошибки в логах

**Причина:** Проблемы с API ключом или авторизацией.

**Решение:**
1. Проверьте, что API ключ в Lockbox актуален
2. Проверьте права сервисного аккаунта:
   ```bash
   yc iam service-account get prometheus-writer
   ```

## ✅ Установка завершена

**Статус:** Все компоненты работают корректно

### Дашборды (активны)

1. **Simple Test Dashboard** (4 виджета - базовые метрики):
   https://monitoring.yandex.cloud/folders/b1ge0vpe8dp87vc3n73l/monitoring/dashboards/fbedr8rdlmvn6gv2ftlt

2. **Prometheus Metrics Dashboard** (7 виджетов - системные метрики):
   https://monitoring.yandex.cloud/folders/b1ge0vpe8dp87vc3n73l/monitoring/dashboards/fbe6thc2rlq5006qnhl8

3. **Comprehensive Dashboard** (18 виджетов - все метрики):
   https://monitoring.yandex.cloud/folders/b1ge0vpe8dp87vc3n73l/monitoring/dashboards/fbec8eu2t9pes36mlq3c

### Что исправлено

1. ✅ ProcessCollector duplication error - убраны дублирующие регистрации (автоматически регистрируются prometheus_client)
2. ✅ Dashboard query format - все запросы используют PromQL (`text_mode = false`)
3. ✅ Prometheus remote_write - успешно отправляет метрики в workspace
4. ✅ Все targets healthy - bot, worker, prometheus

### Следующие шаги (опционально)

1. Добавить алерты для critical метрик (stuck tasks, high memory)
2. Настроить Grafana для более гибкой визуализации
3. Добавить retention policy для метрик

## Полезные команды

```bash
# Проверить статус Prometheus
docker ps | grep prometheus

# Посмотреть логи Prometheus
docker logs mood-prometheus --tail 100

# Перезапустить Prometheus
docker restart mood-prometheus

# Проверить remote_write статус
curl http://158.160.122.216:9090/api/v1/status/tsdb

# Список метрик в Prometheus
curl http://158.160.122.216:9090/api/v1/label/__name__/values
```

## Документация

- [Yandex Managed Prometheus](https://cloud.yandex.com/en/docs/monitoring/operations/prometheus/)
- [PromQL Documentation](https://prometheus.io/docs/prometheus/latest/querying/basics/)
- [Terraform yandex_monitoring_dashboard](https://terraform-provider.yandexcloud.net/resources/monitoring_dashboard)
