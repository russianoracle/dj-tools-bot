terraform {
  required_providers {
    yandex = {
      source  = "yandex-cloud/yandex"
      version = "~> 0.120"
    }
  }
}

provider "yandex" {
  folder_id = var.folder_id
}

variable "folder_id" {
  description = "Yandex Cloud folder ID"
  type        = string
  default     = "b1ge0vpe8dp87vc3n73l"
}

variable "workspace_id" {
  description = "Prometheus workspace ID"
  type        = string
  default     = "monhr2m0jl118pvk9mtu"
}

resource "yandex_monitoring_dashboard" "prometheus_metrics" {
  name  = "prometheus-metrics-dashboard"
  title = "DJ Tools Bot - Prometheus Metrics"

  # Widget 1: ARQ Queue Depth
  widgets {
    chart {
      title          = "ARQ Queue Depth"
      chart_id       = "arq_queue_depth"
      display_legend = true

      queries {
        target {
          query     = "arq_queue_depth{queue_name=\"default\"}"
          text_mode = false
        }
        downsampling {
          grid_aggregation = "GRID_AGGREGATION_AVG"
          max_points       = 1000
        }
      }

      visualization_settings {
        type        = "VISUALIZATION_TYPE_LINE"
        interpolate = "INTERPOLATE_LINEAR"

        yaxis_settings {
          left {
            min         = "0"
            unit_format = "UNIT_NONE"
          }
        }
      }
    }

    position {
      x = 0
      y = 0
      w = 12
      h = 8
    }
  }

  # Widget 2: Tasks In Progress
  widgets {
    chart {
      title          = "ARQ Tasks In Progress"
      chart_id       = "arq_in_progress"
      display_legend = true

      queries {
        target {
          query     = "arq_in_progress{queue_name=\"default\"}"
          text_mode = false
        }
        downsampling {
          grid_aggregation = "GRID_AGGREGATION_AVG"
          max_points       = 1000
        }
      }

      visualization_settings {
        type        = "VISUALIZATION_TYPE_LINE"
        interpolate = "INTERPOLATE_LINEAR"

        yaxis_settings {
          left {
            min         = "0"
            unit_format = "UNIT_NONE"
          }
        }
      }
    }

    position {
      x = 12
      y = 0
      w = 12
      h = 8
    }
  }

  # Widget 3: Task Processing Rate
  widgets {
    chart {
      title          = "Task Processing Rate"
      chart_id       = "task_rate"
      display_legend = true

      queries {
        target {
          query     = "rate(arq_tasks_total[5m])"
          text_mode = false
        }
        downsampling {
          grid_aggregation = "GRID_AGGREGATION_AVG"
          max_points       = 1000
        }
      }

      visualization_settings {
        type        = "VISUALIZATION_TYPE_LINE"
        interpolate = "INTERPOLATE_LINEAR"

        yaxis_settings {
          left {
            min         = "0"
            unit_format = "UNIT_NONE"
            title       = "tasks/sec"
          }
        }
      }
    }

    position {
      x = 0
      y = 8
      w = 12
      h = 8
    }
  }

  # Widget 4: Bot Memory Usage
  widgets {
    chart {
      title          = "Bot Memory Usage"
      chart_id       = "bot_memory"
      display_legend = true

      queries {
        target {
          query     = "memory_usage_bytes{job=\"bot\"}"
          text_mode = false
        }
        downsampling {
          grid_aggregation = "GRID_AGGREGATION_AVG"
          max_points       = 1000
        }
      }

      visualization_settings {
        type        = "VISUALIZATION_TYPE_LINE"
        interpolate = "INTERPOLATE_LINEAR"

        yaxis_settings {
          left {
            min         = "0"
            unit_format = "UNIT_BYTES_SI"
          }
        }
      }
    }

    position {
      x = 12
      y = 8
      w = 12
      h = 8
    }
  }

  # Widget 5: Worker Memory Usage
  widgets {
    chart {
      title          = "Worker Memory Usage"
      chart_id       = "worker_memory"
      display_legend = true

      queries {
        target {
          query     = "memory_usage_bytes{job=\"worker\"}"
          text_mode = false
        }
        downsampling {
          grid_aggregation = "GRID_AGGREGATION_AVG"
          max_points       = 1000
        }
      }

      visualization_settings {
        type        = "VISUALIZATION_TYPE_LINE"
        interpolate = "INTERPOLATE_LINEAR"

        yaxis_settings {
          left {
            min         = "0"
            unit_format = "UNIT_BYTES_SI"
          }
        }
      }
    }

    position {
      x = 0
      y = 16
      w = 12
      h = 8
    }
  }

  # Widget 6: CPU Usage
  widgets {
    chart {
      title          = "CPU Usage"
      chart_id       = "cpu_usage"
      display_legend = true

      queries {
        target {
          query     = "rate(process_cpu_seconds_total[5m])"
          text_mode = false
        }
        downsampling {
          grid_aggregation = "GRID_AGGREGATION_AVG"
          max_points       = 1000
        }
      }

      visualization_settings {
        type        = "VISUALIZATION_TYPE_LINE"
        interpolate = "INTERPOLATE_LINEAR"

        yaxis_settings {
          left {
            min         = "0"
            max         = "1"
            unit_format = "UNIT_PERCENT_UNIT"
          }
        }
      }
    }

    position {
      x = 12
      y = 16
      w = 12
      h = 8
    }
  }

  # Widget 7: Python GC Collections
  widgets {
    chart {
      title          = "Python GC Collections Rate"
      chart_id       = "gc_rate"
      display_legend = true

      queries {
        target {
          query     = "rate(python_gc_collections_total[5m])"
          text_mode = false
        }
        downsampling {
          grid_aggregation = "GRID_AGGREGATION_AVG"
          max_points       = 1000
        }
      }

      visualization_settings {
        type        = "VISUALIZATION_TYPE_LINE"
        interpolate = "INTERPOLATE_LINEAR"

        yaxis_settings {
          left {
            min         = "0"
            unit_format = "UNIT_NONE"
            title       = "collections/sec"
          }
        }
      }
    }

    position {
      x = 0
      y = 24
      w = 24
      h = 8
    }
  }
}

output "dashboard_id" {
  value       = yandex_monitoring_dashboard.prometheus_metrics.id
  description = "Dashboard ID"
}

output "dashboard_url" {
  value       = "https://monitoring.yandex.cloud/folders/${var.folder_id}/monitoring/dashboards/${yandex_monitoring_dashboard.prometheus_metrics.id}"
  description = "Dashboard URL"
}
