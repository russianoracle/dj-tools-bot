#!/bin/sh
set -e

# Create API key file from environment variable
if [ -n "$PROMETHEUS_API_KEY" ]; then
    echo "$PROMETHEUS_API_KEY" > /etc/prometheus/api_key.txt
    chmod 600 /etc/prometheus/api_key.txt
    echo "API key file created successfully"
else
    echo "WARNING: PROMETHEUS_API_KEY not set - remote_write will fail"
fi

# Start Prometheus with original arguments
exec /bin/prometheus "$@"
