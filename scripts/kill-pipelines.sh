#!/bin/bash
# Force kill all running/queued GitHub Actions pipelines

set -e

REPO="russianoracle/dj-tools-bot"
GH_BIN="${GH_BIN:-/opt/homebrew/bin/gh}"

if ! command -v "$GH_BIN" &> /dev/null; then
    echo "❌ gh CLI not found at $GH_BIN"
    exit 1
fi

echo "🛑 Force canceling all GitHub Actions jobs..."

# Force cancel in_progress jobs
RUNNING_IDS=$($GH_BIN run list --repo "$REPO" --status in_progress --json databaseId -q '.[].databaseId')
if [ -n "$RUNNING_IDS" ]; then
    echo "Killing $(echo "$RUNNING_IDS" | wc -l | tr -d ' ') in_progress jobs..."
    echo "$RUNNING_IDS" | while read RUN_ID; do
        $GH_BIN api --method POST "/repos/$REPO/actions/runs/$RUN_ID/force-cancel" 2>&1 || true
        echo "  ✓ Killed run $RUN_ID"
    done
fi

# Force cancel queued jobs
QUEUED_IDS=$($GH_BIN run list --repo "$REPO" --status queued --json databaseId -q '.[].databaseId')
if [ -n "$QUEUED_IDS" ]; then
    echo "Killing $(echo "$QUEUED_IDS" | wc -l | tr -d ' ') queued jobs..."
    echo "$QUEUED_IDS" | while read RUN_ID; do
        $GH_BIN api --method POST "/repos/$REPO/actions/runs/$RUN_ID/force-cancel" 2>&1 || true
        echo "  ✓ Killed run $RUN_ID"
    done
fi

if [ -z "$RUNNING_IDS" ] && [ -z "$QUEUED_IDS" ]; then
    echo "ℹ️  No jobs to kill"
else
    echo "✅ All jobs killed"
fi
