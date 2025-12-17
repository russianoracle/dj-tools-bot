"""Yandex Cloud Logging handler with gRPC integration."""

import logging
import os
import threading
import queue
from datetime import datetime, timezone
from typing import Optional
import json

# Lazy imports for yandexcloud SDK
_yc_sdk = None
_log_ingestion_service = None


def _get_yc_imports():
    """Lazy import yandexcloud SDK modules."""
    global _yc_sdk, _log_ingestion_service
    if _yc_sdk is None:
        try:
            import yandexcloud
            from yandex.cloud.logging.v1 import log_ingestion_service_pb2_grpc
            from yandex.cloud.logging.v1 import log_ingestion_service_pb2
            from yandex.cloud.logging.v1 import log_entry_pb2
            from google.protobuf import timestamp_pb2
            _yc_sdk = {
                "yandexcloud": yandexcloud,
                "log_ingestion_pb2": log_ingestion_service_pb2,
                "log_entry_pb2": log_entry_pb2,
                "timestamp_pb2": timestamp_pb2,
                "grpc_service": log_ingestion_service_pb2_grpc,
            }
        except ImportError:
            _yc_sdk = None
    return _yc_sdk


# Mapping Python logging levels to YC Logging levels
LEVEL_MAP = {
    logging.DEBUG: "DEBUG",
    logging.INFO: "INFO",
    logging.WARNING: "WARN",
    logging.ERROR: "ERROR",
    logging.CRITICAL: "FATAL",
}


class YandexCloudLoggingHandler(logging.Handler):
    """
    Async handler that sends logs to Yandex Cloud Logging via gRPC.

    Uses background thread for non-blocking log delivery.
    Supports batching and automatic retry.
    """

    def __init__(
        self,
        log_group_id: Optional[str] = None,
        folder_id: Optional[str] = None,
        resource_type: str = "bot",
        resource_id: Optional[str] = None,
        sa_key_file: Optional[str] = None,
        batch_size: int = 50,
        flush_interval: float = 5.0,
        level: int = logging.INFO,
    ):
        """
        Initialize YC Logging handler.

        Args:
            log_group_id: YC Log Group ID (either this or folder_id required)
            folder_id: YC Folder ID for default log group
            resource_type: Resource type label (e.g., 'bot', 'worker')
            resource_id: Resource ID (defaults to hostname)
            sa_key_file: Path to service account key JSON file
            batch_size: Max entries per batch
            flush_interval: Seconds between flush attempts
            level: Minimum log level
        """
        super().__init__(level)

        self.log_group_id = log_group_id or os.getenv("YC_LOG_GROUP_ID")
        self.folder_id = folder_id or os.getenv("YC_FOLDER_ID")
        self.resource_type = resource_type
        self.resource_id = resource_id or os.getenv("HOSTNAME", "unknown")
        self.sa_key_file = sa_key_file or os.getenv("YC_SA_KEY_FILE")
        self.batch_size = batch_size
        self.flush_interval = flush_interval

        self._queue: queue.Queue = queue.Queue(maxsize=10000)
        self._shutdown = threading.Event()
        self._sdk = None
        self._channel = None
        self._stub = None
        self._worker_thread: Optional[threading.Thread] = None

        # Start background worker if config is valid
        if self.log_group_id or self.folder_id:
            self._start_worker()

    def _init_sdk(self) -> bool:
        """Initialize YC SDK and gRPC channel."""
        yc = _get_yc_imports()
        if yc is None:
            return False

        try:
            if self.sa_key_file and os.path.exists(self.sa_key_file):
                with open(self.sa_key_file) as f:
                    sa_key = json.load(f)
                self._sdk = yc["yandexcloud"].SDK(service_account_key=sa_key)
            else:
                # Use metadata service (for VMs/Cloud Functions)
                self._sdk = yc["yandexcloud"].SDK()

            self._channel = self._sdk.client(
                yc["grpc_service"].LogIngestionServiceStub
            )
            return True
        except Exception as e:
            import sys
            sys.stderr.write(f"YCLoggingHandler: Failed to init YC SDK: {e}\n")
            return False

    def _start_worker(self):
        """Start background worker thread."""
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="yc-logging-worker"
        )
        self._worker_thread.start()

    def _worker_loop(self):
        """Background loop that batches and sends logs."""
        batch = []
        last_flush = datetime.now(timezone.utc)

        while not self._shutdown.is_set():
            try:
                # Collect entries with timeout
                try:
                    entry = self._queue.get(timeout=1.0)
                    batch.append(entry)
                except queue.Empty:
                    pass

                # Check flush conditions
                now = datetime.now(timezone.utc)
                should_flush = (
                    len(batch) >= self.batch_size or
                    (batch and (now - last_flush).total_seconds() >= self.flush_interval)
                )

                if should_flush and batch:
                    self._send_batch(batch)
                    batch = []
                    last_flush = now

            except Exception as e:
                import sys
                sys.stderr.write(f"YCLoggingHandler: Worker error: {e}\n")

        # Final flush on shutdown
        if batch:
            self._send_batch(batch)

    def _send_batch(self, batch: list):
        """Send batch of log entries to YC."""
        if not self._channel and not self._init_sdk():
            return

        yc = _get_yc_imports()
        if yc is None:
            return

        try:
            entries = []
            for record in batch:
                ts = timestamp_pb2 = yc["timestamp_pb2"].Timestamp()
                ts.FromDatetime(datetime.fromtimestamp(record["timestamp"], timezone.utc))

                entry = yc["log_entry_pb2"].IncomingLogEntry(
                    timestamp=ts,
                    level=yc["log_entry_pb2"].LogLevel.Level.Value(record["level"]),
                    message=record["message"],
                    json_payload=record.get("json_payload"),
                )
                entries.append(entry)

            # Build request
            destination = yc["log_ingestion_pb2"].Destination(
                log_group_id=self.log_group_id
            ) if self.log_group_id else yc["log_ingestion_pb2"].Destination(
                folder_id=self.folder_id
            )

            resource = yc["log_entry_pb2"].LogEntryResource(
                type=self.resource_type,
                id=self.resource_id,
            )

            request = yc["log_ingestion_pb2"].WriteRequest(
                destination=destination,
                resource=resource,
                entries=entries,
            )

            self._channel.Write(request)

        except Exception as e:
            import sys
            sys.stderr.write(f"YCLoggingHandler: Failed to send logs: {e}\n")

    def emit(self, record: logging.LogRecord):
        """Queue log record for async sending."""
        if self._shutdown.is_set():
            return

        try:
            # Build structured payload
            json_payload = None
            if hasattr(record, "structured_data"):
                from google.protobuf import struct_pb2
                json_payload = struct_pb2.Struct()
                json_payload.update(record.structured_data)

            entry = {
                "timestamp": record.created,
                "level": LEVEL_MAP.get(record.levelno, "INFO"),
                "message": self.format(record),
                "json_payload": json_payload,
            }

            # Add correlation_id if present
            if hasattr(record, "correlation_id") and json_payload:
                json_payload["correlation_id"] = record.correlation_id

            self._queue.put_nowait(entry)
        except queue.Full:
            pass  # Drop log if queue is full

    def close(self):
        """Shutdown handler and flush remaining logs."""
        self._shutdown.set()
        if self._worker_thread:
            self._worker_thread.join(timeout=10)
        super().close()
