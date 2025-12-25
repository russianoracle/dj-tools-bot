#!/bin/bash
# VM Cleanup Script - Find and remove trash files

echo "🔍 Analyzing disk usage on VM..."
echo ""

echo "=== Top 20 largest files ==="
find /home /var -type f -size +100M -exec ls -lh {} \; 2>/dev/null | sort -k5 -hr | head -20

echo ""
echo "=== Docker disk usage ==="
docker system df

echo ""
echo "=== Old Docker logs (>7 days) ==="
find /var/lib/docker/containers -name "*.log" -mtime +7 -exec ls -lh {} \;

echo ""
echo "=== Large directories ==="
du -h --max-depth=2 /home /var/log /tmp 2>/dev/null | sort -hr | head -20

echo ""
echo "=== Cleanup commands ==="
echo "# Remove old Docker logs:"
echo "find /var/lib/docker/containers -name '*.log' -mtime +7 -delete"
echo ""
echo "# Clean Docker system:"
echo "docker system prune -af --volumes"
echo ""
echo "# Clean package cache:"
echo "sudo apt-get clean"
