"""
Business metrics collection using Prometheus.

Tracks key performance indicators:
- Track analysis duration
- Queue depth and processing rate
- Cache hit/miss rate
- Drop/transition detection success rate
- Error rates
- User requests per hour
"""

from prometheus_client import Counter, Histogram, Gauge, Info
import time
from functools import wraps
from typing import Callable, Any

# =============================================================================
# System Metrics
# =============================================================================

# Request counters
telegram_requests_total = Counter(
    'telegram_requests_total',
    'Total Telegram bot requests',
    ['command', 'user_id']
)

telegram_errors_total = Counter(
    'telegram_errors_total',
    'Total Telegram bot errors',
    ['error_type', 'command']
)

# =============================================================================
# Queue Metrics
# =============================================================================

arq_queue_depth = Gauge(
    'arq_queue_depth',
    'Current ARQ queue depth',
    ['queue_name']
)

arq_in_progress = Gauge(
    'arq_in_progress',
    'Number of tasks currently processing',
    ['queue_name']
)

arq_tasks_total = Counter(
    'arq_tasks_total',
    'Total ARQ tasks processed',
    ['task_name', 'status']  # status: success, failure, retry
)

# =============================================================================
# Analysis Performance Metrics
# =============================================================================

analysis_duration_seconds = Histogram(
    'analysis_duration_seconds',
    'Track analysis duration in seconds',
    ['analysis_type'],  # full_analysis, drop_detection, transition_detection
    buckets=[1, 5, 10, 30, 60, 120, 300, 600]  # 1s to 10min
)

stft_computation_seconds = Histogram(
    'stft_computation_seconds',
    'STFT computation duration in seconds',
    ['mode'],  # standard, streaming
    buckets=[0.5, 1, 2, 5, 10, 20, 30, 60, 120]
)

track_duration_seconds = Histogram(
    'track_duration_seconds',
    'Analyzed track duration in seconds',
    buckets=[60, 180, 300, 600, 1200, 1800, 3600, 5400, 7200]  # 1min to 2hr
)

# =============================================================================
# Cache Metrics
# =============================================================================

cache_operations_total = Counter(
    'cache_operations_total',
    'Total cache operations',
    ['operation', 'result']  # operation: get, set, delete; result: hit, miss, error
)

cache_size_bytes = Gauge(
    'cache_size_bytes',
    'Cache size in bytes',
    ['cache_type']  # predictions_db, stft, features
)

cache_entries_total = Gauge(
    'cache_entries_total',
    'Total cache entries',
    ['cache_type']
)

# =============================================================================
# Feature Detection Metrics
# =============================================================================

drops_detected_total = Counter(
    'drops_detected_total',
    'Total drops detected',
    ['zone']  # yellow, green, purple
)

transitions_detected_total = Counter(
    'transitions_detected_total',
    'Total transitions detected',
    ['from_zone', 'to_zone']
)

feature_extraction_duration_seconds = Histogram(
    'feature_extraction_duration_seconds',
    'Feature extraction duration',
    buckets=[0.1, 0.5, 1, 2, 5, 10]
)

# =============================================================================
# Memory Metrics (Application-Level)
# =============================================================================

memory_usage_bytes = Gauge(
    'memory_usage_bytes',
    'Application memory usage in bytes',
    ['process']  # bot, worker
)

peak_memory_bytes = Gauge(
    'peak_memory_bytes',
    'Peak memory usage during processing',
    ['task_type']
)

# =============================================================================
# Error Tracking
# =============================================================================

processing_errors_total = Counter(
    'processing_errors_total',
    'Total processing errors',
    ['error_type', 'stage']  # stage: load_audio, stft, feature_extraction, etc.
)

oom_kills_total = Counter(
    'oom_kills_total',
    'Total OOM kill events detected'
)

# =============================================================================
# Business Metrics
# =============================================================================

sets_analyzed_total = Counter(
    'sets_analyzed_total',
    'Total DJ sets analyzed',
    ['source']  # telegram, local, api
)

unique_users_total = Gauge(
    'unique_users_total',
    'Total unique users (last 24h)'
)

analysis_requests_per_hour = Gauge(
    'analysis_requests_per_hour',
    'Analysis requests per hour (rolling average)'
)

# =============================================================================
# Info Metrics
# =============================================================================

app_info = Info(
    'app',
    'Application version and environment info'
)

# =============================================================================
# Decorators for Automatic Metrics
# =============================================================================

def track_duration(metric: Histogram, labels: dict = None):
    """
    Decorator to track function execution duration.

    Usage:
        @track_duration(analysis_duration_seconds, {'analysis_type': 'full_analysis'})
        def analyze_track(path):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time

                # Record metric
                if labels:
                    metric.labels(**labels).observe(duration)
                else:
                    metric.observe(duration)

                return result
            except Exception as e:
                duration = time.time() - start_time
                # Still record duration even on error
                if labels:
                    metric.labels(**labels).observe(duration)
                raise
        return wrapper
    return decorator


def count_calls(metric: Counter, labels: dict = None, success_label: str = 'status'):
    """
    Decorator to count function calls and track success/failure.

    Usage:
        @count_calls(arq_tasks_total, {'task_name': 'analyze_track'})
        def process_task(ctx):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                result = func(*args, **kwargs)
                # Success
                if labels:
                    metric.labels(**{**labels, success_label: 'success'}).inc()
                else:
                    metric.labels(**{success_label: 'success'}).inc()
                return result
            except Exception as e:
                # Failure
                if labels:
                    metric.labels(**{**labels, success_label: 'failure'}).inc()
                else:
                    metric.labels(**{success_label: 'failure'}).inc()
                raise
        return wrapper
    return decorator


# =============================================================================
# Helper Functions
# =============================================================================

def record_cache_hit():
    """Record a cache hit."""
    cache_operations_total.labels(operation='get', result='hit').inc()


def record_cache_miss():
    """Record a cache miss."""
    cache_operations_total.labels(operation='get', result='miss').inc()


def record_telegram_request(command: str, user_id: int):
    """Record a Telegram bot request."""
    telegram_requests_total.labels(command=command, user_id=str(user_id)).inc()


def record_error(error_type: str, stage: str = 'unknown'):
    """Record a processing error."""
    processing_errors_total.labels(error_type=error_type, stage=stage).inc()


def update_queue_metrics(redis_client):
    """
    Update ARQ queue metrics from Redis.

    Call this periodically (e.g., every 10 seconds).
    """
    try:
        # Get queue depth
        default_queue = redis_client.llen('arq:queue:default')
        arq_queue_depth.labels(queue_name='default').set(default_queue)

        # Get in-progress count
        in_progress = redis_client.llen('arq:queue:in-progress')
        arq_in_progress.labels(queue_name='default').set(in_progress)
    except Exception:
        pass  # Don't fail on metrics collection


def set_app_info(version: str, environment: str, python_version: str):
    """Set application info metric."""
    app_info.info({
        'version': version,
        'environment': environment,
        'python_version': python_version
    })
