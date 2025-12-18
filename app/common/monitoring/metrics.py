"""Yandex Cloud Monitoring metrics collector."""

import os
import time
import psutil
from typing import Optional, Dict, Any
from datetime import datetime

from app.common.logging import get_logger

logger = get_logger(__name__)


class MetricsCollector:
    """
    Collect and send performance metrics to Yandex Cloud Monitoring.

    Tracks:
    - Memory usage (current, peak)
    - Processing time per stage
    - File duration vs processing time
    - Task success/failure rates
    """

    # Class-level flag to track if SDK warning was already logged
    _sdk_warning_logged = False

    def __init__(self, enabled: bool = True, folder_id: Optional[str] = None):
        """
        Initialize metrics collector.

        Args:
            enabled: Enable metrics collection
            folder_id: YC Folder ID (from env if not provided)
        """
        self.enabled = enabled and bool(os.getenv("YC_FOLDER_ID") or folder_id)
        self.folder_id = folder_id or os.getenv("YC_FOLDER_ID")
        self.process = psutil.Process()

        # YC Monitoring client (lazy init)
        self._monitoring_client = None
        self._sdk_import_failed = False  # Track if SDK import already failed

        if self.enabled:
            logger.info("Metrics collection enabled", data={
                "folder_id": self.folder_id,
            })
        else:
            logger.info("Metrics collection disabled")

    @property
    def monitoring_client(self):
        """Lazy initialization of YC Monitoring client."""
        # If SDK import already failed, skip trying again
        if self._sdk_import_failed:
            return None

        if self._monitoring_client is None and self.enabled:
            try:
                from yandex.cloud.monitoring.v3.metric_service_pb2_grpc import MetricServiceStub
                from yandex.cloud.monitoring.v3.metric_service_pb2 import WriteRequest
                import yandexcloud

                sdk = yandexcloud.SDK()
                self._monitoring_client = sdk.client(MetricServiceStub)
                logger.info("YC Monitoring client initialized")
            except ImportError:
                # Only log warning once across all instances
                if not MetricsCollector._sdk_warning_logged:
                    logger.info("yandexcloud SDK not installed, metrics disabled")
                    MetricsCollector._sdk_warning_logged = True
                self.enabled = False
                self._sdk_import_failed = True
            except Exception as e:
                logger.warning(f"Failed to init YC Monitoring: {e}")
                self.enabled = False
                self._sdk_import_failed = True

        return self._monitoring_client

    def get_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        try:
            return self.process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0

    def record_stage_metrics(
        self,
        stage_name: str,
        duration_sec: float,
        memory_mb: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Record metrics for a pipeline stage.

        Args:
            stage_name: Stage name (e.g., "ComputeSTFTStage")
            duration_sec: Stage duration in seconds
            memory_mb: Memory usage in MB (current if not provided)
            context: Additional context (file_duration, etc.)
        """
        if not self.enabled:
            return

        memory_mb = memory_mb or self.get_memory_mb()

        metrics = {
            "stage_duration_sec": duration_sec,
            "memory_usage_mb": memory_mb,
        }

        # Add context metrics
        if context:
            if "file_duration_sec" in context:
                metrics["file_duration_sec"] = context["file_duration_sec"]
            if "peak_memory_mb" in context:
                metrics["peak_memory_mb"] = context["peak_memory_mb"]

        logger.info(f"Stage metrics: {stage_name}", data={
            "stage_name": stage_name,
            **metrics,
        })

        # Send to YC Monitoring
        self._send_metrics(stage_name, metrics)

    def record_task_metrics(
        self,
        task_name: str,
        duration_sec: float,
        success: bool,
        file_duration_sec: Optional[float] = None,
        peak_memory_mb: Optional[float] = None,
        error: Optional[str] = None
    ):
        """
        Record metrics for a completed task.

        Args:
            task_name: Task name (e.g., "analyze_set_task")
            duration_sec: Task duration in seconds
            success: Task succeeded
            file_duration_sec: Audio file duration
            peak_memory_mb: Peak memory usage
            error: Error message if failed
        """
        if not self.enabled:
            return

        metrics = {
            "task_duration_sec": duration_sec,
            "task_success": 1 if success else 0,
            "memory_usage_mb": self.get_memory_mb(),
        }

        if file_duration_sec:
            metrics["file_duration_sec"] = file_duration_sec
            metrics["processing_speed_ratio"] = file_duration_sec / duration_sec if duration_sec > 0 else 0

        if peak_memory_mb:
            metrics["peak_memory_mb"] = peak_memory_mb

        logger.info(f"Task metrics: {task_name}", data={
            "task_name": task_name,
            "success": success,
            "error": error,
            **metrics,
        })

        self._send_metrics(task_name, metrics)

    def _send_metrics(self, metric_prefix: str, metrics: Dict[str, float]):
        """
        Send metrics to YC Monitoring.

        Args:
            metric_prefix: Metric name prefix
            metrics: Dict of metric_name -> value
        """
        if not self.enabled or not self.monitoring_client:
            return

        try:
            from yandex.cloud.monitoring.v3.metric_service_pb2 import WriteRequest
            from google.protobuf.timestamp_pb2 import Timestamp

            now = Timestamp()
            now.GetCurrentTime()

            metric_data = []
            for name, value in metrics.items():
                metric_data.append({
                    "name": f"mood_classifier.{metric_prefix}.{name}",
                    "labels": {
                        "service": "dj-tools-bot",
                        "component": "arq-worker",
                    },
                    "ts": now,
                    "value": float(value),
                })

            request = WriteRequest(
                folder_id=self.folder_id,
                metrics=metric_data,
            )

            self.monitoring_client.Write(request)
            logger.debug(f"Sent {len(metrics)} metrics to YC Monitoring")

        except Exception as e:
            logger.warning(f"Failed to send metrics: {e}")


# Global instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get or create global metrics collector."""
    global _metrics_collector
    if _metrics_collector is None:
        enabled = os.getenv("ENABLE_METRICS", "true").lower() in ("true", "1", "yes")
        _metrics_collector = MetricsCollector(enabled=enabled)
    return _metrics_collector
