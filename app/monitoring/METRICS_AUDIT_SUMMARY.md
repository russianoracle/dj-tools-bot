# Metrics Audit Summary

## Problem Identified

Dashboard widgets показывали ошибку: `NoSuchElementException: variable 'process_resident_memory_bytes' not found, available variables: []`

## Root Cause

Использовался **неправильный формат запросов** в дашбордах:
- ❌ Yandex Monitoring Query Language: `"metric_name"{folderId="...", service="..."}`
- ✅ Prometheus Query Language (PromQL): `metric_name`

**Ключевое понимание:** Prometheus workspace метрики НЕ доступны через стандартный Yandex Monitoring Query Language. Для них нужен **обычный PromQL**.

## Changes Made

### 1. Prometheus Configuration (`app/monitoring/prometheus.yml`)

**Added labels for better organization:**
```yaml
global:
  external_labels:
    folderId: 'b1ge0vpe8dp87vc3n73l'  # Added for Yandex Cloud context

scrape_configs:
  - job_name: 'bot'
    labels:
      service: 'custom'  # Changed from 'mood-classifier-bot'
      app: 'mood-classifier-bot'  # Added for distinction
```

**Note:** Labels `folderId` и `service` добавлены для совместимости, но **не влияют на формат запросов в дашбордах**.

### 2. Metrics Server (`app/core/monitoring/server.py`)

**Added standard Prometheus collectors:**
```python
ProcessCollector(registry=REGISTRY)  # For process_cpu_seconds_total, process_resident_memory_bytes, etc.
GCCollector(registry=REGISTRY)       # For python_gc_collections_total
PlatformCollector(registry=REGISTRY) # For platform metrics
```

### 3. Dashboard Format (`app/monitoring/terraform/dashboard_simple_test.tf`)

**Corrected query format:**
```hcl
queries {
  target {
    query     = "arq_queue_depth"  # PromQL - no quotes, no labels
    text_mode = false              # IMPORTANT: must be false for PromQL
  }
}
```

## Metrics Inventory

Total: **33 custom metrics** defined in `metrics.py`

### System Metrics (auto-exported by collectors)
- `process_cpu_seconds_total` - CPU usage
- `process_resident_memory_bytes` - Memory RSS
- `python_gc_collections_total` - GC activity

### Application Metrics (by category)

**Queue & Tasks (5):**
- `arq_queue_depth` - Queue size
- `arq_in_progress` - Tasks being processed
- `arq_tasks_total` - Total tasks (success/failure)
- `stuck_tasks_total` - Tasks stuck >10min
- `oldest_task_age_seconds` - Age of oldest queued task

**Performance (3):**
- `analysis_duration_seconds` - Track analysis time
- `stft_computation_seconds` - STFT processing time
- `track_duration_seconds` - Track length distribution

**Cache (3):**
- `cache_operations_total` - Hit/miss counters
- `cache_size_bytes` - Cache size
- `cache_entries_total` - Number of entries

**Features (3):**
- `drops_detected_total` - Drop detection counts
- `transitions_detected_total` - Transition detection
- `feature_extraction_duration_seconds` - Feature extraction time

**Errors (3):**
- `processing_errors_total` - Processing failures
- `telegram_errors_total` - Bot errors
- `user_errors_total` - User-specific errors

**Business (3):**
- `sets_analyzed_total` - DJ sets processed
- `unique_users_total` - Active users (24h)
- `telegram_requests_total` - Bot requests

**User Activity (4):**
- `user_requests_total` - Requests per user
- `user_tracks_analyzed_total` - Tracks per user
- `user_active_sessions` - Active sessions count
- `user_avg_track_duration` - Track duration histogram

**Operational (4):**
- `task_timeouts_total` - Task timeout events
- `long_running_tasks` - Tasks >30min
- `worker_healthy` - Worker health status (1/0)
- `last_heartbeat_timestamp` - Last heartbeat time

## Correct Dashboard Query Patterns

### Simple Metric
```hcl
query = "arq_queue_depth"
text_mode = false
```

### With Label Filters
```hcl
query = "arq_queue_depth{process=\"bot\"}"
text_mode = false
```

### With PromQL Functions
```hcl
query = "rate(process_cpu_seconds_total[5m]) * 100"
text_mode = false
```

### Aggregations
```hcl
query = "sum by (process) (arq_queue_depth)"
text_mode = false
```

## Common Mistakes to Avoid

❌ **DON'T:**
```hcl
query = "\"metric_name\"{folderId=\"...\", service=\"...\"}"  # Yandex Query Language
text_mode = true
```

✅ **DO:**
```hcl
query = "metric_name"  # PromQL
text_mode = false
```

## Next Steps

1. **Apply Terraform changes:**
   ```bash
   cd app/monitoring/terraform
   terraform init
   terraform apply -auto-approve
   ```

2. **Wait 5-10 minutes** for metrics to populate after Prometheus restart

3. **Verify in dashboard** that widgets show data

4. **Update comprehensive dashboard** with correct PromQL format

## References

- Prometheus Query Language: https://prometheus.io/docs/prometheus/latest/querying/basics/
- Yandex Monitoring Prometheus workspace: https://cloud.yandex.com/en/docs/monitoring/operations/prometheus/
- Dashboard configuration: `app/monitoring/terraform/`
