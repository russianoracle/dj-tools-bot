#!/usr/bin/env python3
"""
Export Redis ARQ Queue Metrics to Yandex Cloud Monitoring

Collects queue metrics and sends them to YC Monitoring for dashboards and alerts.

Usage:
    # One-time export
    python scripts/export_queue_metrics.py

    # Continuous export (every 60 seconds)
    python scripts/export_queue_metrics.py --loop 60

Environment variables:
    YC_FOLDER_ID - Yandex Cloud folder ID
    REDIS_HOST - Redis host (default: localhost)
    REDIS_PORT - Redis port (default: 6379)
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List

try:
    import requests
except ImportError:
    print("❌ Error: requests not installed")
    print("   Run: pip install requests")
    sys.exit(1)

# Import our metrics collector
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from scripts.queue_metrics import QueueMetrics


class YandexMonitoringExporter:
    """Export metrics to Yandex Cloud Monitoring."""

    def __init__(self, folder_id: str):
        self.folder_id = folder_id
        # Official YC Monitoring API endpoint with service=custom for custom metrics
        self.monitoring_url = f"https://monitoring.api.cloud.yandex.net/monitoring/v2/data/write?folderId={folder_id}&service=custom"

    def get_iam_token(self) -> str:
        """Get IAM token from instance metadata service."""
        try:
            response = requests.get(
                "http://169.254.169.254/computeMetadata/v1/instance/service-accounts/default/token",
                headers={"Metadata-Flavor": "Google"},
                timeout=2,
            )
            response.raise_for_status()
            return response.json()["access_token"]
        except Exception as e:
            print(f"⚠️  Warning: Could not get IAM token from metadata: {e}")
            print("   Using environment variable YC_IAM_TOKEN if available")
            return os.getenv("YC_IAM_TOKEN", "")

    def format_metrics(self, stats: Dict) -> List[Dict]:
        """
        Format queue stats for YC Monitoring API.

        YC Monitoring format:
        {
            "ts": "2025-12-25T10:00:00Z",
            "labels": {"service": "dj-tools-bot"},
            "metrics": [
                {"name": "queue.pending", "value": 5}
            ]
        }
        """
        timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

        metrics = [
            {
                "name": "arq.queue.pending",
                "labels": {
                    "service": "dj-tools-bot",
                    "component": "arq-worker",
                },
                "value": stats["pending"],
                "ts": timestamp,
            },
            {
                "name": "arq.queue.in_progress",
                "labels": {
                    "service": "dj-tools-bot",
                    "component": "arq-worker",
                },
                "value": stats["in_progress"],
                "ts": timestamp,
            },
            {
                "name": "arq.queue.completed",
                "labels": {
                    "service": "dj-tools-bot",
                    "component": "arq-worker",
                },
                "value": stats["completed"],
                "ts": timestamp,
            },
            {
                "name": "arq.queue.failed",
                "labels": {
                    "service": "dj-tools-bot",
                    "component": "arq-worker",
                },
                "value": stats["failed"],
                "ts": timestamp,
            },
            {
                "name": "arq.queue.stuck",
                "labels": {
                    "service": "dj-tools-bot",
                    "component": "arq-worker",
                },
                "value": stats["stuck"],
                "ts": timestamp,
            },
        ]

        # Add wait time stats if available
        if "avg_wait_seconds" in stats:
            metrics.append({
                "name": "arq.queue.avg_wait_seconds",
                "labels": {
                    "service": "dj-tools-bot",
                    "component": "arq-worker",
                },
                "value": stats["avg_wait_seconds"],
                "ts": timestamp,
            })

        if "max_wait_seconds" in stats:
            metrics.append({
                "name": "arq.queue.max_wait_seconds",
                "labels": {
                    "service": "dj-tools-bot",
                    "component": "arq-worker",
                },
                "value": stats["max_wait_seconds"],
                "ts": timestamp,
            })

        return metrics

    def send_metrics(self, metrics: List[Dict]) -> bool:
        """Send metrics to YC Monitoring."""
        iam_token = self.get_iam_token()
        if not iam_token:
            print("❌ No IAM token available, skipping metrics export")
            return False

        payload = {
            "folderId": self.folder_id,
            "metrics": metrics,
        }

        headers = {
            "Authorization": f"Bearer {iam_token}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(
                self.monitoring_url,
                json=payload,
                headers=headers,
                timeout=10,
            )
            response.raise_for_status()
            print(f"✅ Sent {len(metrics)} metrics to YC Monitoring")
            return True
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to send metrics: {e}")
            if hasattr(e, "response") and e.response is not None:
                print(f"   Response: {e.response.text}")
            return False


async def collect_and_export_metrics(
    redis_host: str,
    redis_port: int,
    folder_id: str,
) -> bool:
    """Collect metrics from Redis and export to YC Monitoring."""

    print(f"📊 Collecting queue metrics from Redis at {redis_host}:{redis_port}...")

    metrics_collector = QueueMetrics(redis_host=redis_host, redis_port=redis_port)
    exporter = YandexMonitoringExporter(folder_id=folder_id)

    try:
        await metrics_collector.connect()

        # Get summary stats
        stats = await metrics_collector.get_summary_stats()

        # Get wait time stats
        _, pending_details = await metrics_collector.analyze_pending_jobs()
        if pending_details:
            wait_times = [j["wait_time_seconds"] for j in pending_details]
            stats["avg_wait_seconds"] = sum(wait_times) / len(wait_times)
            stats["max_wait_seconds"] = max(wait_times)
        else:
            stats["avg_wait_seconds"] = 0
            stats["max_wait_seconds"] = 0

        print(f"   Pending: {stats['pending']}")
        print(f"   In Progress: {stats['in_progress']}")
        print(f"   Completed: {stats['completed']}")
        print(f"   Failed: {stats['failed']}")
        print(f"   Stuck: {stats['stuck']}")
        if pending_details:
            print(f"   Avg Wait: {stats['avg_wait_seconds'] / 60:.1f}m")
            print(f"   Max Wait: {stats['max_wait_seconds'] / 60:.1f}m")

        # Format and send to YC Monitoring
        yc_metrics = exporter.format_metrics(stats)
        success = exporter.send_metrics(yc_metrics)

        return success

    except ConnectionRefusedError:
        print("❌ Error: Cannot connect to Redis")
        print(f"   Make sure Redis is running at {redis_host}:{redis_port}")
        return False
    except Exception as e:
        print(f"❌ Error collecting metrics: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        await metrics_collector.close()


async def continuous_export(
    redis_host: str,
    redis_port: int,
    folder_id: str,
    interval_seconds: int,
):
    """Continuously export metrics at regular intervals."""
    print(f"🔄 Starting continuous export (interval: {interval_seconds}s)")
    print("   Press Ctrl+C to stop\n")

    try:
        while True:
            await collect_and_export_metrics(redis_host, redis_port, folder_id)
            print(f"\n⏱  Waiting {interval_seconds}s until next export...\n")
            await asyncio.sleep(interval_seconds)
    except KeyboardInterrupt:
        print("\n\n✋ Stopped by user")


def main():
    """Main entry point."""
    # Get configuration from environment
    folder_id = os.getenv("YC_FOLDER_ID")
    if not folder_id:
        print("❌ Error: YC_FOLDER_ID environment variable not set")
        print("   Export: export YC_FOLDER_ID=b1ge0vpe8dp87vc3n73l")
        sys.exit(1)

    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = int(os.getenv("REDIS_PORT", "6379"))

    # Parse command line arguments
    loop_interval = None
    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--loop" and i + 1 < len(sys.argv) - 1:
            loop_interval = int(sys.argv[i + 2])

    print("=" * 60)
    print("  ARQ Queue Metrics → Yandex Cloud Monitoring")
    print("=" * 60)
    print(f"Folder ID: {folder_id}")
    print(f"Redis:     {redis_host}:{redis_port}")
    print("=" * 60)
    print()

    # Run export
    if loop_interval:
        asyncio.run(
            continuous_export(redis_host, redis_port, folder_id, loop_interval)
        )
    else:
        success = asyncio.run(
            collect_and_export_metrics(redis_host, redis_port, folder_id)
        )
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()