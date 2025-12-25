"""
Prometheus metrics HTTP server.

Exposes /metrics endpoint for Unified Agent to scrape.
Runs in background thread to not block main application.
"""

import logging
from prometheus_client import start_http_server, REGISTRY, ProcessCollector, GCCollector, PlatformCollector
from threading import Thread
import time
import os

# Register default Prometheus collectors for process metrics
ProcessCollector(registry=REGISTRY)
GCCollector(registry=REGISTRY)
PlatformCollector(registry=REGISTRY)

logger = logging.getLogger(__name__)

# Metrics server configuration
METRICS_PORT = int(os.getenv('METRICS_PORT', '8000'))
METRICS_UPDATE_INTERVAL = int(os.getenv('METRICS_UPDATE_INTERVAL', '10'))  # seconds


class MetricsServer:
    """Background metrics server with periodic updates."""

    def __init__(self, port: int = METRICS_PORT, update_interval: int = METRICS_UPDATE_INTERVAL):
        self.port = port
        self.update_interval = update_interval
        self._server_thread = None
        self._update_thread = None
        self._running = False

    def start(self):
        """Start the metrics server in a background thread."""
        if self._running:
            logger.warning("Metrics server already running")
            return

        try:
            # Start Prometheus HTTP server
            start_http_server(self.port, registry=REGISTRY)
            self._running = True
            logger.info(f"Metrics server started on port {self.port}")

            # Start periodic updates thread
            self._update_thread = Thread(target=self._update_loop, daemon=True)
            self._update_thread.start()
            logger.info(f"Metrics update loop started (interval: {self.update_interval}s)")

        except OSError as e:
            logger.error(f"Failed to start metrics server on port {self.port}: {e}")
            raise

    def _update_loop(self):
        """Periodically update metrics that require polling."""
        from .metrics import update_queue_metrics, memory_usage_bytes
        import psutil
        import redis

        # Initialize Redis connection for queue metrics
        redis_host = os.getenv('REDIS_HOST', 'localhost')
        redis_port = int(os.getenv('REDIS_PORT', '6379'))

        try:
            redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2
            )
        except Exception as e:
            logger.warning(f"Failed to connect to Redis for metrics: {e}")
            redis_client = None

        process = psutil.Process()
        process_name = os.getenv('PROCESS_NAME', 'unknown')

        while self._running:
            try:
                # Update queue metrics
                if redis_client:
                    update_queue_metrics(redis_client)

                # Update memory metrics
                mem_info = process.memory_info()
                memory_usage_bytes.labels(process=process_name).set(mem_info.rss)

            except Exception as e:
                logger.debug(f"Error updating metrics: {e}")

            time.sleep(self.update_interval)

    def stop(self):
        """Stop the metrics server."""
        self._running = False
        logger.info("Metrics server stopped")


# Global metrics server instance
_metrics_server = None


def start_metrics_server(port: int = METRICS_PORT, update_interval: int = METRICS_UPDATE_INTERVAL):
    """
    Start the global metrics server.

    Args:
        port: Port to expose metrics on (default: 8000)
        update_interval: Interval in seconds for periodic metric updates (default: 10)
    """
    global _metrics_server

    if _metrics_server is not None:
        logger.warning("Metrics server already initialized")
        return _metrics_server

    _metrics_server = MetricsServer(port=port, update_interval=update_interval)
    _metrics_server.start()

    return _metrics_server


def get_metrics_server() -> MetricsServer:
    """Get the global metrics server instance."""
    return _metrics_server
