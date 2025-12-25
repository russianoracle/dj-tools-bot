# Terraform Dashboard Deployment

Автоматическое создание дашборда Prometheus в Yandex Monitoring через Terraform.

## Быстрый старт

```bash
cd app/monitoring/terraform

# Инициализация
terraform init

# Проверка плана
terraform plan

# Применение
terraform apply -auto-approve
```

## Что создается

- Dashboard "DJ Tools Bot - Prometheus Metrics"
- 7 виджетов с метриками:
  1. ARQ Queue Depth
  2. Tasks In Progress
  3. Task Processing Rate
  4. Bot Memory Usage
  5. Worker Memory Usage
  6. CPU Usage
  7. Python GC Collections Rate

## После создания

URL дашборда выведется в output:
```
dashboard_url = "https://monitoring.yandex.cloud/folders/b1ge0vpe8dp87vc3n73l/monitoring/dashboards/..."
```

## Обновление

Если нужно изменить дашборд:
1. Отредактируй `main.tf`
2. Запусти `terraform apply`

## Удаление

```bash
terraform destroy
```

## Требования

- Terraform >= 1.0
- Yandex Cloud CLI настроен (`yc init`)
- Права на создание дашбордов в folder `b1ge0vpe8dp87vc3n73l`

## Troubleshooting

### "Authentication failed"
```bash
yc config list  # Проверь что IAM token актуален
yc init         # Переинициализируй если нужно
```

### "Dashboard already exists"
Terraform state хранит ID дашборда. Если удалил вручную:
```bash
terraform state rm yandex_monitoring_dashboard.prometheus_metrics
terraform apply
```
