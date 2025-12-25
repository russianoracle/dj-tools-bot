resource "yandex_monitoring_dashboard" "simple_test" {
  name  = "simple-test"
  title = "Simple Test - Correct Format"

  # ARQ Queue Depth - with correct format
  widgets {
    chart {
      title          = "ARQ Queue Depth"
      chart_id       = "test_queue"
      display_legend = true
      queries {
        target {
          query     = "\"arq_queue_depth\"{folderId=\"b1ge0vpe8dp87vc3n73l\", service=\"custom\"}"
          text_mode = true
        }
        downsampling {
          grid_aggregation = "GRID_AGGREGATION_MAX"
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
      h = 6
    }
  }

  # Process CPU seconds (from bot)
  widgets {
    chart {
      title          = "Process CPU Seconds (Bot)"
      chart_id       = "test_cpu"
      display_legend = true
      queries {
        target {
          query     = "\"process_cpu_seconds_total\"{folderId=\"b1ge0vpe8dp87vc3n73l\", service=\"custom\"}"
          text_mode = true
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
            unit_format = "UNIT_SECONDS"
          }
        }
      }
    }
    position {
      x = 12
      y = 0
      w = 12
      h = 6
    }
  }

  # Python GC collections (from bot)
  widgets {
    chart {
      title          = "Python GC Collections (Bot)"
      chart_id       = "test_gc"
      display_legend = true
      queries {
        target {
          query     = "\"python_gc_collections_total\"{folderId=\"b1ge0vpe8dp87vc3n73l\", service=\"custom\"}"
          text_mode = true
        }
        downsampling {
          grid_aggregation = "GRID_AGGREGATION_MAX"
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
      y = 6
      w = 12
      h = 6
    }
  }

  # Memory RSS (from bot)
  widgets {
    chart {
      title          = "Process Memory RSS (Bot)"
      chart_id       = "test_memory"
      display_legend = true
      queries {
        target {
          query     = "\"process_resident_memory_bytes\"{folderId=\"b1ge0vpe8dp87vc3n73l\", service=\"custom\"}"
          text_mode = true
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
      y = 6
      w = 12
      h = 6
    }
  }
}

output "simple_test_dashboard_url" {
  value       = "https://monitoring.yandex.cloud/folders/${var.folder_id}/monitoring/dashboards/${yandex_monitoring_dashboard.simple_test.id}"
  description = "Simple Test Dashboard URL"
}
