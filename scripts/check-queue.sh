#!/bin/bash
# Quick Redis Queue Health Check
# Usage: ./scripts/check-queue.sh

set -e

echo "========================================="
echo "   Redis ARQ Queue Health Check"
echo "========================================="
echo ""

# Check Redis connectivity
echo "🔍 Redis Connection:"
if redis-cli ping > /dev/null 2>&1; then
    echo "  ✅ Connected"
else
    echo "  ❌ Failed to connect to Redis"
    exit 1
fi
echo ""

# Queue metrics
echo "📊 Queue Metrics:"
QUEUE_LEN=$(redis-cli LLEN arq:queue 2>/dev/null || echo "0")
IN_PROGRESS=$(redis-cli --scan --pattern "arq:in-progress:*" 2>/dev/null | wc -l | tr -d ' ')
COMPLETED=$(redis-cli --scan --pattern "arq:result:*" 2>/dev/null | wc -l | tr -d ' ')
WORKERS=$(redis-cli --scan --pattern "arq:worker:*" 2>/dev/null | wc -l | tr -d ' ')

echo "  Pending Jobs:     $QUEUE_LEN"
echo "  In Progress:      $IN_PROGRESS"
echo "  Completed (cached): $COMPLETED"
echo "  Active Workers:   $WORKERS"
echo ""

# Memory usage
echo "💾 Memory Usage:"
MEMORY=$(redis-cli INFO memory 2>/dev/null | grep "used_memory_human:" | cut -d: -f2 | tr -d '\r')
MAX_MEMORY=$(redis-cli INFO memory 2>/dev/null | grep "maxmemory_human:" | cut -d: -f2 | tr -d '\r')
echo "  Used:      $MEMORY"
echo "  Max:       $MAX_MEMORY"
echo ""

# Performance stats
echo "⚡ Performance:"
OPS=$(redis-cli INFO stats 2>/dev/null | grep "instantaneous_ops_per_sec:" | cut -d: -f2 | tr -d '\r')
HITS=$(redis-cli INFO stats 2>/dev/null | grep "keyspace_hits:" | cut -d: -f2 | tr -d '\r')
MISSES=$(redis-cli INFO stats 2>/dev/null | grep "keyspace_misses:" | cut -d: -f2 | tr -d '\r')
echo "  Ops/sec:   $OPS"
echo "  Hit rate:  $HITS hits / $MISSES misses"
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
else
    echo "  ✅ Workers active ($WORKERS)"
fi

if [ "$IN_PROGRESS" -gt 0 ]; then
    echo "  🔄 Jobs processing ($IN_PROGRESS)"
fi

echo ""
echo "========================================="
echo "For detailed diagnostics: docs/REDIS_DIAGNOSTICS.md"
echo "========================================="
