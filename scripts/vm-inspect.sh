#!/bin/bash
# VM Structure Inspector
# Run this on the VM to see application layout

echo "=== VM Structure Inspection ==="
echo ""

echo "1. Application directories:"
echo "---"
find /home /opt -maxdepth 3 -name "mood-classifier" -o -name "dj-tools-bot" -o -name "app" 2>/dev/null | head -20
echo ""

echo "2. Docker Compose files:"
echo "---"
find /home /opt -name "docker-compose.yml" 2>/dev/null
echo ""

echo "3. Environment files:"
echo "---"
find /home /opt -name ".env" 2>/dev/null
echo ""

echo "4. Configuration files:"
echo "---"
find /home /opt -name "*.conf" -o -name "*.yaml" -o -name "*.yml" 2>/dev/null | grep -E "(fluent|logging|docker)" | head -20
echo ""

echo "5. Current directory structure:"
echo "---"
pwd
ls -lah
echo ""

echo "6. Home directory:"
echo "---"
ls -lah ~/
echo ""

echo "7. Docker volumes:"
echo "---"
docker volume ls
echo ""

echo "8. Running containers and their mounts:"
echo "---"
docker ps --format "table {{.Names}}\t{{.Mounts}}"
echo ""

echo "9. Check common app locations:"
for dir in /home/app /opt/mood-classifier /home/ubuntu/mood-classifier /root/mood-classifier; do
    if [ -d "$dir" ]; then
        echo "✅ Found: $dir"
        ls -la "$dir" | head -10
        echo ""
    fi
done
