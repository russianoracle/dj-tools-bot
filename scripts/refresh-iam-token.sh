#!/bin/bash
set -euo pipefail

# Refresh IAM token for Yandex Cloud Logging
# Run this script periodically (every 6 hours) via cron
# Add to crontab: 0 */6 * * * /usr/local/bin/refresh-iam-token.sh >> /var/log/iam-token-refresh.log 2>&1

# Detect environment
if [ -d "/home/ubuntu/app" ]; then
    # Production server
    APP_DIR="/home/ubuntu/app"
elif [ -d "/home/app" ]; then
    # Alternative production path
    APP_DIR="/home/app"
else
    echo "❌ App directory not found"
    exit 1
fi

ENV_FILE="${APP_DIR}/.env"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🔐 Refreshing IAM token..."

# Create IAM token using yc CLI (uses VM's attached service account)
IAM_TOKEN=$(yc iam create-token 2>/dev/null)

if [ -z "$IAM_TOKEN" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ❌ Failed to create IAM token"
    exit 1
fi

# Update .env file
if grep -q "^YC_TOKEN=" "$ENV_FILE" 2>/dev/null; then
    # Replace existing token
    sed -i.bak "s|^YC_TOKEN=.*|YC_TOKEN=$IAM_TOKEN|" "$ENV_FILE"
else
    # Add new token
    echo "YC_TOKEN=$IAM_TOKEN" >> "$ENV_FILE"
fi

# Restart fluent-bit to pick up new token
cd "$APP_DIR" || exit 1
docker restart mood-fluent-bit >/dev/null 2>&1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ IAM token refreshed and fluent-bit restarted"