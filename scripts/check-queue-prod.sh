#!/bin/bash
# Quick Redis Queue Health Check (Production - Docker)
# Usage: ./scripts/check-queue-prod.sh

set -e

echo "========================================="
echo "   Redis ARQ Queue Health Check (PROD)"
echo "========================================="
echo ""

# Find Redis container
REDIS_CONTAINER=$(docker ps -q -f name=redis)

if [ -z "$REDIS_CONTAINER" ]; then
    echo "❌ Redis container not found"
    echo "   Run: docker ps | grep redis"
    exit 1
fi

echo "🐳 Redis Container: $REDIS_CONTAINER"
echo ""

# Helper function for redis-cli in Docker
redis_exec() {
    docker exec $REDIS_CONTAINER redis-cli "$@"
}

# Check Redis connectivity
echo "🔍 Redis Connection:"
if redis_exec ping > /dev/null 2>&1; then
    echo "  ✅ Connected"
else
    echo "  ❌ Failed to connect to Redis"
    exit 1
fi
echo ""

# Queue metrics
echo "📊 Queue Metrics:"
QUEUE_LEN=$(redis_exec LLEN arq:queue 2>/dev/null || echo "0")
IN_PROGRESS=$(redis_exec --scan --pattern "arq:in-progress:*" 2>/dev/null | wc -l | tr -d ' ')
COMPLETED=$(redis_exec --scan --pattern "arq:result:*" 2>/dev/null | wc -l | tr -d ' ')
WORKERS=$(redis_exec --scan --pattern "arq:worker:*" 2>/dev/null | wc -l | tr -d ' ')

echo "  Pending Jobs:       $QUEUE_LEN"
echo "  In Progress:        $IN_PROGRESS"
echo "  Completed (cached): $COMPLETED"
echo "  Active Workers:     $WORKERS"
echo ""

# Memory usage
echo "💾 Memory Usage:"
MEMORY=$(redis_exec INFO memory 2>/dev/null | grep "used_memory_human:" | cut -d: -f2 | tr -d '\r')
MAX_MEMORY=$(redis_exec INFO memory 2>/dev/null | grep "maxmemory_human:" | cut -d: -f2 | tr -d '\r')
echo "  Used:      $MEMORY"
echo "  Max:       $MAX_MEMORY"
echo ""

# Performance stats
echo "⚡ Performance:"
OPS=$(redis_exec INFO stats 2>/dev/null | grep "instantaneous_ops_per_sec:" | cut -d: -f2 | tr -d '\r')
HITS=$(redis_exec INFO stats 2>/dev/null | grep "keyspace_hits:" | cut -d: -f2 | tr -d '\r')
MISSES=$(redis_exec INFO stats 2>/dev/null | grep "keyspace_misses:" | cut -d: -f2 | tr -d '\r')
echo "  Ops/sec:   $OPS"
echo "  Hit rate:  $HITS hits / $MISSES misses"
echo ""

# Show recent jobs (last 5)
echo "📋 Recent Jobs (last 5):"
redis_exec LRANGE arq:queue 0 4 2>/dev/null | head -5 | while IFS= read -r job; do
    echo "  • $job"
done
echo ""

# Health assessment
echo "🏥 Health Status:"
if [ "$QUEUE_LEN" -gt 100 ]; then
    echo "  ⚠️  High queue length ($QUEUE_LEN) - consider scaling workers"
elif [ "$QUEUE_LEN" -gt 50 ]; then
    echo "  ⚠️  Moderate queue length ($QUEUE_LEN)"
else
    echo "  ✅ Queue length normal ($QUEUE_LEN)"
fi

if [ "$WORKERS" -eq 0 ]; then
    echo "  ❌ No active workers detected!"
    echo "     Check: docker ps | grep worker"
else
    echo "  ✅ Workers active ($WORKERS)"
fi

if [ "$IN_PROGRESS" -gt 0 ]; then
    echo "  🔄 Jobs processing ($IN_PROGRESS)"
fi

echo ""
echo "========================================="
echo "Interactive mode: docker exec -it $REDIS_CONTAINER redis-cli"
echo "Documentation:    docs/REDIS_DIAGNOSTICS.md"
echo "========================================="