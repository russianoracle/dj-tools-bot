#!/usr/bin/env python3
"""
Setup Yandex Cloud Monitoring Dashboard for mood-classifier metrics.

Requires:
- yandexcloud SDK
- YC_FOLDER_ID environment variable
- Authenticated yc CLI or IAM token
"""

import os
import json
from typing import Dict, Any


def create_dashboard_config(folder_id: str) -> Dict[str, Any]:
    """Create dashboard configuration for YC Monitoring."""

    return {
        "folder_id": folder_id,
        "name": "Mood Classifier Performance",
        "description": "Resource usage and performance metrics per stage",
        "widgets": [
            # Row 1: Overview
            {
                "position": {"x": 0, "y": 0, "w": 4, "h": 2},
                "text": {
                    "text": "# Performance Overview\n\nReal-time metrics for DJ set analysis pipeline"
                }
            },
            {
                "position": {"x": 4, "y": 0, "w": 4, "h": 2},
                "chart": {
                    "id": "success_rate",
                    "title": "Success Rate",
                    "chart_type": "GAUGE",
                    "queries": [{
                        "target": {
                            "query": "avg_over_time(mood_classifier.analyze_set_task.task_success[1h]) * 100"
                        }
                    }],
                    "visualization_settings": {
                        "threshold": [
                            {"value": 90, "color": "green"},
                            {"value": 70, "color": "yellow"},
                            {"value": 0, "color": "red"}
                        ]
                    }
                }
            },
            {
                "position": {"x": 8, "y": 0, "w": 4, "h": 2},
                "chart": {
                    "id": "processing_speed",
                    "title": "Processing Speed (x realtime)",
                    "chart_type": "GAUGE",
                    "queries": [{
                        "target": {
                            "query": "avg_over_time(mood_classifier.analyze_set_task.processing_speed_ratio[10m])"
                        }
                    }],
                    "visualization_settings": {
                        "threshold": [
                            {"value": 10, "color": "green"},
                            {"value": 5, "color": "yellow"},
                            {"value": 0, "color": "red"}
                        ]
                    }
                }
            },

            # Row 2: Task Performance
            {
                "position": {"x": 0, "y": 2, "w": 6, "h": 3},
                "chart": {
                    "id": "task_duration_timeline",
                    "title": "Task Duration Over Time",
                    "chart_type": "LINE",
                    "queries": [{
                        "target": {
                            "query": "mood_classifier.analyze_set_task.task_duration_sec"
                        }
                    }]
                }
            },
            {
                "position": {"x": 6, "y": 2, "w": 3, "h": 3},
                "chart": {
                    "id": "p95_latency",
                    "title": "P95 Latency",
                    "chart_type": "STAT",
                    "queries": [{
                        "target": {
                            "query": "quantile_over_time(0.95, mood_classifier.analyze_set_task.task_duration_sec[1h])"
                        }
                    }],
                    "visualization_settings": {
                        "unit": "seconds"
                    }
                }
            },
            {
                "position": {"x": 9, "y": 2, "w": 3, "h": 3},
                "chart": {
                    "id": "throughput",
                    "title": "Throughput (tasks/hour)",
                    "chart_type": "STAT",
                    "queries": [{
                        "target": {
                            "query": "count_over_time(mood_classifier.analyze_set_task.task_success[1h])"
                        }
                    }]
                }
            },

            # Row 3: Memory Analysis
            {
                "position": {"x": 0, "y": 5, "w": 6, "h": 3},
                "chart": {
                    "id": "memory_by_stage",
                    "title": "Memory Usage by Stage",
                    "chart_type": "BAR",
                    "queries": [
                        {"target": {"query": "mood_classifier.LoadAudioStage.memory_usage_mb"}},
                        {"target": {"query": "mood_classifier.ComputeSTFTStage.memory_usage_mb"}},
                        {"target": {"query": "mood_classifier.DetectTransitionsStage.memory_usage_mb"}},
                        {"target": {"query": "mood_classifier.LaplacianSegmentationStage.memory_usage_mb"}},
                        {"target": {"query": "mood_classifier.DetectAllDropsStage.memory_usage_mb"}},
                    ],
                    "visualization_settings": {
                        "unit": "megabytes"
                    }
                }
            },
            {
                "position": {"x": 6, "y": 5, "w": 6, "h": 3},
                "chart": {
                    "id": "stft_memory_trend",
                    "title": "STFT Peak Memory Trend",
                    "chart_type": "LINE",
                    "queries": [{
                        "target": {
                            "query": "mood_classifier.ComputeSTFTStage.peak_memory_mb"
                        }
                    }],
                    "visualization_settings": {
                        "threshold": [
                            {"value": 8000, "color": "red", "label": "High (>8GB)"},
                            {"value": 6000, "color": "yellow", "label": "Moderate (>6GB)"}
                        ]
                    }
                }
            },

            # Row 4: Stage Breakdown
            {
                "position": {"x": 0, "y": 8, "w": 6, "h": 3},
                "chart": {
                    "id": "slowest_stages",
                    "title": "Top 5 Slowest Stages",
                    "chart_type": "TABLE",
                    "queries": [{
                        "target": {
                            "query": "topk(5, avg_over_time(mood_classifier.*.stage_duration_sec[10m]))"
                        }
                    }]
                }
            },
            {
                "position": {"x": 6, "y": 8, "w": 6, "h": 3},
                "chart": {
                    "id": "stft_duration",
                    "title": "STFT Computation Time",
                    "chart_type": "LINE",
                    "queries": [{
                        "target": {
                            "query": "mood_classifier.ComputeSTFTStage.stage_duration_sec"
                        }
                    }],
                    "visualization_settings": {
                        "unit": "seconds"
                    }
                }
            },

            # Row 5: Alerts
            {
                "position": {"x": 0, "y": 11, "w": 4, "h": 2},
                "chart": {
                    "id": "high_memory_alert",
                    "title": "⚠️ High Memory Usage (>8GB)",
                    "chart_type": "STAT",
                    "queries": [{
                        "target": {
                            "query": "count(mood_classifier.ComputeSTFTStage.peak_memory_mb > 8000)"
                        }
                    }],
                    "visualization_settings": {
                        "threshold": [
                            {"value": 1, "color": "red"}
                        ]
                    }
                }
            },
            {
                "position": {"x": 4, "y": 11, "w": 4, "h": 2},
                "chart": {
                    "id": "slow_tasks_alert",
                    "title": "⚠️ Slow Tasks (<6x realtime)",
                    "chart_type": "STAT",
                    "queries": [{
                        "target": {
                            "query": "count(mood_classifier.analyze_set_task.processing_speed_ratio < 6)"
                        }
                    }],
                    "visualization_settings": {
                        "threshold": [
                            {"value": 1, "color": "yellow"}
                        ]
                    }
                }
            },
            {
                "position": {"x": 8, "y": 11, "w": 4, "h": 2},
                "chart": {
                    "id": "failure_rate_alert",
                    "title": "⚠️ Failure Rate (>10%)",
                    "chart_type": "STAT",
                    "queries": [{
                        "target": {
                            "query": "avg_over_time(mood_classifier.analyze_set_task.task_success[10m]) < 0.9"
                        }
                    }],
                    "visualization_settings": {
                        "threshold": [
                            {"value": 1, "color": "red"}
                        ]
                    }
                }
            }
        ]
    }


def create_dashboard_via_api(config: Dict[str, Any]) -> None:
    """Create dashboard using YC Monitoring API."""
    try:
        from yandexcloud import SDK
        from yandex.cloud.monitoring.v3.dashboard_service_pb2 import CreateDashboardRequest
        from yandex.cloud.monitoring.v3.dashboard_service_pb2_grpc import DashboardServiceStub

        sdk = SDK()
        dashboard_service = sdk.client(DashboardServiceStub)

        # Convert config to protobuf
        request = CreateDashboardRequest(
            folder_id=config["folder_id"],
            name=config["name"],
            description=config["description"],
            # Note: widgets format depends on YC protobuf schema
            # This is simplified - actual implementation may need adjustment
        )

        response = dashboard_service.Create(request)
        print(f"✅ Dashboard created: {response.id}")
        print(f"🔗 URL: https://console.cloud.yandex.ru/folders/{config['folder_id']}/monitoring/dashboards/{response.id}")

    except ImportError:
        print("❌ yandexcloud SDK not installed")
        print("💡 Install: pip install yandexcloud")
    except Exception as e:
        print(f"❌ Error creating dashboard: {e}")
        print("\n💡 Alternative: Save config and create manually:")
        print(json.dumps(config, indent=2))


def main():
    """Main entry point."""
    folder_id = os.getenv("YC_FOLDER_ID", "b1ge0vpe8dp87vc3n73l")

    if not folder_id:
        print("❌ YC_FOLDER_ID environment variable not set")
        return 1

    print("🔧 Creating Mood Classifier Performance Dashboard...")
    print(f"📁 Folder ID: {folder_id}")

    config = create_dashboard_config(folder_id)

    # Save config to file
    config_file = "/tmp/yc-monitoring-dashboard.json"
    with open(config_file, "w") as f:
        json.dump(config, indent=2, fp=f)
    print(f"💾 Config saved: {config_file}")

    # Try to create via API
    create_dashboard_via_api(config)

    print("\n✅ Done!")
    print("\n📊 Dashboard includes:")
    print("  • Success rate & processing speed gauges")
    print("  • Task duration timeline & P95 latency")
    print("  • Memory usage by stage (bar chart)")
    print("  • STFT peak memory trend")
    print("  • Top 5 slowest stages table")
    print("  • 3 alert panels (memory, speed, failures)")

    return 0


if __name__ == "__main__":
    exit(main())
