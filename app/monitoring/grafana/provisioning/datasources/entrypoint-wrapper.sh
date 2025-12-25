#!/bin/sh
# Replace PROMETHEUS_API_KEY placeholder with actual value
if [ -n "$PROMETHEUS_API_KEY" ]; then
  sed -i "s/\${PROMETHEUS_API_KEY}/$PROMETHEUS_API_KEY/g" /etc/grafana/provisioning/datasources/prometheus.yml
fi

# Run original Grafana entrypoint
exec /run.sh "$@"
