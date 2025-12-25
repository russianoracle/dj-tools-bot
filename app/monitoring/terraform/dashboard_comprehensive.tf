resource "yandex_monitoring_dashboard" "comprehensive" {
  name  = "dj-tools-comprehensive"
  title = "DJ Tools Bot - Comprehensive Monitoring"

  # ============================================================================
  # SECTION 1: SYSTEM RESOURCES
  # ============================================================================

  # Title: System Resources
  widgets {
    title {
      text = "SYSTEM RESOURCES"
      size = "TITLE_SIZE_L"
    }
    position { x = 0; y = 0; w = 24; h = 1 }
  }

  # CPU Usage
  widgets {
    chart {
      title          = "CPU Usage %"
      chart_id       = "cpu_usage"
      display_legend = true
      queries {
        target {
          query     = "rate(process_cpu_seconds_total[5m]) * 100"
          text_mode = true
        }
        downsampling { grid_aggregation = "GRID_AGGREGATION_AVG"; max_points = 1000 }
      }
      visualization_settings {
        type        = "VISUALIZATION_TYPE_LINE"
        interpolate = "INTERPOLATE_LINEAR"
        yaxis_settings { left { min = "0"; max = "100"; unit_format = "UNIT_PERCENT" } }
      }
    }
    position { x = 0; y = 1; w = 8; h = 6 }
  }

  # Memory Usage (RSS)
  widgets {
    chart {
      title          = "Memory Usage (RSS)"
      chart_id       = "memory_rss"
      display_legend = true
      queries {
        target {
          query     = "process_resident_memory_bytes"
          text_mode = true
        }
        downsampling { grid_aggregation = "GRID_AGGREGATION_AVG"; max_points = 1000 }
      }
      visualization_settings {
        type        = "VISUALIZATION_TYPE_LINE"
        interpolate = "INTERPOLATE_LINEAR"
        yaxis_settings { left { min = "0"; unit_format = "UNIT_BYTES_SI" } }
      }
    }
    position { x = 8; y = 1; w = 8; h = 6 }
  }

  # Python GC Activity
  widgets {
    chart {
      title          = "Python GC Collections/sec"
      chart_id       = "gc_rate"
      display_legend = true
      queries {
        target {
          query     = "rate(python_gc_collections_total[1m])"
          text_mode = true
        }
        downsampling { grid_aggregation = "GRID_AGGREGATION_AVG"; max_points = 1000 }
      }
      visualization_settings {
        type        = "VISUALIZATION_TYPE_LINE"
        interpolate = "INTERPOLATE_LINEAR"
        yaxis_settings { left { min = "0"; unit_format = "UNIT_NONE" } }
      }
    }
    position { x = 16; y = 1; w = 8; h = 6 }
  }

  # ============================================================================
  # SECTION 2: BUSINESS METRICS
  # ============================================================================

  # Title: Business Metrics
  widgets {
    title {
      text = "BUSINESS METRICS"
      size = "TITLE_SIZE_L"
    }
    position { x = 0; y = 7; w = 24; h = 1 }
  }

  # Track Analysis Rate
  widgets {
    chart {
      title          = "Track Analysis Rate (tracks/min)"
      chart_id       = "analysis_rate"
      display_legend = true
      queries {
        target {
          query     = "rate(arq_tasks_total{status=\"success\"}[5m]) * 60"
          text_mode = true
        }
        downsampling { grid_aggregation = "GRID_AGGREGATION_AVG"; max_points = 1000 }
      }
      visualization_settings {
        type        = "VISUALIZATION_TYPE_LINE"
        interpolate = "INTERPOLATE_LINEAR"
        yaxis_settings { left { min = "0"; unit_format = "UNIT_NONE"; title = "tracks/min" } }
      }
    }
    position { x = 0; y = 8; w = 12; h = 6 }
  }

  # Cache Hit Rate
  widgets {
    chart {
      title          = "Cache Hit Rate %"
      chart_id       = "cache_hit_rate"
      display_legend = true
      queries {
        target {
          query     = "(rate(cache_operations_total{result=\"hit\"}[5m]) / (rate(cache_operations_total{result=\"hit\"}[5m]) + rate(cache_operations_total{result=\"miss\"}[5m]))) * 100"
          text_mode = true
        }
        downsampling { grid_aggregation = "GRID_AGGREGATION_AVG"; max_points = 1000 }
      }
      visualization_settings {
        type        = "VISUALIZATION_TYPE_LINE"
        interpolate = "INTERPOLATE_LINEAR"
        yaxis_settings { left { min = "0"; max = "100"; unit_format = "UNIT_PERCENT" } }
      }
    }
    position { x = 12; y = 8; w = 12; h = 6 }
  }

  # Analysis Duration (p95)
  widgets {
    chart {
      title          = "Analysis Duration p95 (seconds)"
      chart_id       = "analysis_p95"
      display_legend = true
      queries {
        target {
          query     = "histogram_quantile(0.95, rate(analysis_duration_seconds_bucket[5m]))"
          text_mode = true
        }
        downsampling { grid_aggregation = "GRID_AGGREGATION_AVG"; max_points = 1000 }
      }
      visualization_settings {
        type        = "VISUALIZATION_TYPE_LINE"
        interpolate = "INTERPOLATE_LINEAR"
        yaxis_settings { left { min = "0"; unit_format = "UNIT_SECONDS" } }
      }
    }
    position { x = 0; y = 14; w = 12; h = 6 }
  }

  # Error Rate
  widgets {
    chart {
      title          = "Processing Errors (per minute)"
      chart_id       = "error_rate"
      display_legend = true
      queries {
        target {
          query     = "rate(processing_errors_total[5m]) * 60"
          text_mode = true
        }
        downsampling { grid_aggregation = "GRID_AGGREGATION_AVG"; max_points = 1000 }
      }
      visualization_settings {
        type        = "VISUALIZATION_TYPE_LINE"
        interpolate = "INTERPOLATE_LINEAR"
        yaxis_settings { left { min = "0"; unit_format = "UNIT_NONE" } }
      }
    }
    position { x = 12; y = 14; w = 12; h = 6 }
  }

  # ============================================================================
  # SECTION 3: OPERATIONS (REALTIME)
  # ============================================================================

  # Title: Operations & Health
  widgets {
    title {
      text = "OPERATIONS & HEALTH (REALTIME)"
      size = "TITLE_SIZE_L"
    }
    position { x = 0; y = 20; w = 24; h = 1 }
  }

  # ARQ Queue Depth (CRITICAL)
  widgets {
    chart {
      title          = "⚠️ Queue Depth (ALERT if >10)"
      chart_id       = "queue_depth_alert"
      display_legend = true
      queries {
        target {
          query     = "arq_queue_depth{queue_name=\"default\"}"
          text_mode = true
        }
        downsampling { grid_aggregation = "GRID_AGGREGATION_MAX"; max_points = 1000 }
      }
      visualization_settings {
        type        = "VISUALIZATION_TYPE_LINE"
        interpolate = "INTERPOLATE_LINEAR"
        yaxis_settings { left { min = "0"; unit_format = "UNIT_NONE" } }
        # Add threshold line at 10
      }
    }
    position { x = 0; y = 21; w = 8; h = 6 }
  }

  # Tasks In Progress
  widgets {
    chart {
      title          = "Tasks In Progress"
      chart_id       = "tasks_progress"
      display_legend = true
      queries {
        target {
          query     = "arq_in_progress{queue_name=\"default\"}"
          text_mode = true
        }
        downsampling { grid_aggregation = "GRID_AGGREGATION_MAX"; max_points = 1000 }
      }
      visualization_settings {
        type        = "VISUALIZATION_TYPE_LINE"
        interpolate = "INTERPOLATE_LINEAR"
        yaxis_settings { left { min = "0"; unit_format = "UNIT_NONE" } }
      }
    }
    position { x = 8; y = 21; w = 8; h = 6 }
  }

  # Stuck Tasks (NEW)
  widgets {
    chart {
      title          = "🚨 Stuck Tasks (>10min)"
      chart_id       = "stuck_tasks"
      display_legend = true
      queries {
        target {
          query     = "stuck_tasks_total"
          text_mode = true
        }
        downsampling { grid_aggregation = "GRID_AGGREGATION_MAX"; max_points = 1000 }
      }
      visualization_settings {
        type        = "VISUALIZATION_TYPE_LINE"
        interpolate = "INTERPOLATE_LINEAR"
        yaxis_settings { left { min = "0"; unit_format = "UNIT_NONE" } }
      }
    }
    position { x = 16; y = 21; w = 8; h = 6 }
  }

  # Oldest Task Age (NEW)
  widgets {
    chart {
      title          = "Oldest Queued Task Age (minutes)"
      chart_id       = "oldest_task"
      display_legend = true
      queries {
        target {
          query     = "oldest_task_age_seconds / 60"
          text_mode = true
        }
        downsampling { grid_aggregation = "GRID_AGGREGATION_MAX"; max_points = 1000 }
      }
      visualization_settings {
        type        = "VISUALIZATION_TYPE_LINE"
        interpolate = "INTERPOLATE_LINEAR"
        yaxis_settings { left { min = "0"; unit_format = "UNIT_NONE" } }
      }
    }
    position { x = 0; y = 27; w = 12; h = 6 }
  }

  # Worker Health (NEW)
  widgets {
    chart {
      title          = "Worker Health (1=OK, 0=Down)"
      chart_id       = "worker_health"
      display_legend = true
      queries {
        target {
          query     = "worker_healthy"
          text_mode = true
        }
        downsampling { grid_aggregation = "GRID_AGGREGATION_MIN"; max_points = 1000 }
      }
      visualization_settings {
        type        = "VISUALIZATION_TYPE_LINE"
        interpolate = "INTERPOLATE_LINEAR"
        yaxis_settings { left { min = "0"; max = "1"; unit_format = "UNIT_NONE" } }
      }
    }
    position { x = 12; y = 27; w = 12; h = 6 }
  }

  # Task Timeout Rate
  widgets {
    chart {
      title          = "Task Timeouts (per hour)"
      chart_id       = "timeouts"
      display_legend = true
      queries {
        target {
          query     = "rate(task_timeouts_total[5m]) * 3600"
          text_mode = true
        }
        downsampling { grid_aggregation = "GRID_AGGREGATION_AVG"; max_points = 1000 }
      }
      visualization_settings {
        type        = "VISUALIZATION_TYPE_LINE"
        interpolate = "INTERPOLATE_LINEAR"
        yaxis_settings { left { min = "0"; unit_format = "UNIT_NONE" } }
      }
    }
    position { x = 0; y = 33; w = 12; h = 6 }
  }

  # Last Heartbeat (seconds ago)
  widgets {
    chart {
      title          = "Last Heartbeat (seconds ago)"
      chart_id       = "heartbeat"
      display_legend = true
      queries {
        target {
          query     = "time() - last_heartbeat_timestamp"
          text_mode = true
        }
        downsampling { grid_aggregation = "GRID_AGGREGATION_MAX"; max_points = 1000 }
      }
      visualization_settings {
        type        = "VISUALIZATION_TYPE_LINE"
        interpolate = "INTERPOLATE_LINEAR"
        yaxis_settings { left { min = "0"; unit_format = "UNIT_SECONDS" } }
      }
    }
    position { x = 12; y = 33; w = 12; h = 6 }
  }
}

output "comprehensive_dashboard_id" {
  value       = yandex_monitoring_dashboard.comprehensive.id
  description = "Comprehensive Dashboard ID"
}

output "comprehensive_dashboard_url" {
  value       = "https://monitoring.yandex.cloud/folders/${var.folder_id}/monitoring/dashboards/${yandex_monitoring_dashboard.comprehensive.id}"
  description = "Comprehensive Dashboard URL"
}
