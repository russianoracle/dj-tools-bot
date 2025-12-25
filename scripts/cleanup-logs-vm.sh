#!/bin/bash
# Immediate log cleanup script for VM
# Run on VM: bash /home/ubuntu/app/scripts/cleanup-logs-vm.sh

set -e

echo "════════════════════════════════════════════════════════════"
echo "  🧹 EMERGENCY LOG CLEANUP - $(date)"
echo "════════════════════════════════════════════════════════════"

# Show disk usage before
echo ""
echo "📊 Disk usage BEFORE cleanup:"
df -h / | tail -1
echo ""

# 1. Truncate Docker container logs (don't delete, just empty)
echo "🔹 Truncating Docker container logs..."
find /var/lib/docker/containers -name "*.log" -type f -exec sh -c '
    size=$(du -h "$1" | cut -f1)
    if [ "$size" != "0" ]; then
        echo "  Truncating: $1 ($size)"
        : > "$1"
    fi
' _ {} \; 2>/dev/null || echo "  ⚠️ Permission denied (run with sudo)"

# 2. Clean Fluent Bit storage buffer
echo ""
echo "🔹 Cleaning Fluent Bit storage buffer..."
if [ -d /var/log/flb-storage ]; then
    size=$(du -sh /var/log/flb-storage 2>/dev/null | cut -f1 || echo "unknown")
    rm -rf /var/log/flb-storage/*
    echo "  ✅ Cleaned: $size"
else
    echo "  ℹ️ Not found: /var/log/flb-storage"
fi

# 3. Clean BuildKit cache
echo ""
echo "🔹 Cleaning BuildKit cache..."
if [ -d /tmp/.buildx-cache ]; then
    size=$(du -sh /tmp/.buildx-cache 2>/dev/null | cut -f1 || echo "unknown")
    rm -rf /tmp/.buildx-cache
    mkdir -p /tmp/.buildx-cache
    echo "  ✅ Cleaned: $size"
else
    echo "  ℹ️ Not found: /tmp/.buildx-cache"
fi

# 4. Clean Docker system (images, containers, volumes)
echo ""
echo "🔹 Docker system prune..."
docker system prune -af --volumes 2>&1 | grep -E "Total reclaimed|deleted" || echo "  ✅ Cleanup complete"

# 5. Clean APT cache
echo ""
echo "🔹 Cleaning APT cache..."
apt-get clean 2>/dev/null || echo "  ⚠️ Permission denied (run with sudo)"

# 6. Clean old log files
echo ""
echo "🔹 Cleaning old rotated logs..."
find /var/log -name "*.log.*.gz" -mtime +7 -delete 2>/dev/null || true
find /var/log -name "*.log.*" -mtime +14 -delete 2>/dev/null || true
echo "  ✅ Deleted logs older than 7-14 days"

# 7. Clean downloads directory
echo ""
echo "🔹 Cleaning downloads directory..."
if [ -d /tmp/downloads ]; then
    size=$(du -sh /tmp/downloads 2>/dev/null | cut -f1 || echo "unknown")
    rm -rf /tmp/downloads/*
    echo "  ✅ Cleaned: $size"
else
    echo "  ℹ️ Not found: /tmp/downloads"
fi

# Show disk usage after
echo ""
echo "════════════════════════════════════════════════════════════"
echo "📊 Disk usage AFTER cleanup:"
df -h / | tail -1
echo ""

# Show largest directories
echo "📁 Largest directories in /var:"
du -sh /var/* 2>/dev/null | sort -rh | head -10
echo ""

echo "✅ Cleanup complete at $(date)"
echo "════════════════════════════════════════════════════════════"
