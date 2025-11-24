#!/bin/bash

# Monitor training progress every 60 seconds
LOG_FILE="training_dj_features.log"

echo "=========================================="
echo "DJ Features Training Monitor"
echo "=========================================="
echo "Monitoring: $LOG_FILE"
echo "Update interval: 60 seconds"
echo "Press Ctrl+C to stop"
echo "=========================================="
echo ""

while true; do
    clear
    echo "=========================================="
    echo "Training Progress - $(date '+%H:%M:%S')"
    echo "=========================================="

    # Feature extraction progress
    echo ""
    echo "📊 FEATURE EXTRACTION:"
    if [ -f "$LOG_FILE" ]; then
        LAST_EXTRACTED=$(grep -o '\[.*\] (.*%) Extracted features' "$LOG_FILE" | tail -1)
        if [ -n "$LAST_EXTRACTED" ]; then
            echo "  $LAST_EXTRACTED"
        else
            echo "  Starting..."
        fi
    else
        echo "  Log file not found yet..."
    fi

    # Training progress
    echo ""
    echo "🎯 TRAINING STATUS:"
    if [ -f "$LOG_FILE" ]; then
        TRAINING_START=$(grep -o "Training XGBoost" "$LOG_FILE" | tail -1)
        if [ -n "$TRAINING_START" ]; then
            echo "  ✓ Training started"

            # Grid search progress
            GRID_PROGRESS=$(grep -o "Testing.*parameters" "$LOG_FILE" | tail -1)
            if [ -n "$GRID_PROGRESS" ]; then
                echo "  $GRID_PROGRESS"
            fi
        else
            echo "  Waiting for feature extraction to complete..."
        fi
    fi

    # Results
    echo ""
    echo "📈 RESULTS:"
    if [ -f "$LOG_FILE" ]; then
        ACCURACY=$(grep -o "Overall Accuracy: .*%" "$LOG_FILE" | tail -1)
        if [ -n "$ACCURACY" ]; then
            echo "  🎉 $ACCURACY"

            # Per-zone accuracy
            echo ""
            echo "  Per-zone breakdown:"
            grep -E "Yellow Zone Accuracy:|Green Zone Accuracy:|Purple Zone Accuracy:" "$LOG_FILE" | tail -3 | while read line; do
                echo "    $line"
            done

            echo ""
            echo "✅ Training completed!"
            echo "Full results in: $LOG_FILE"
            exit 0
        else
            echo "  Not yet available"
        fi
    fi

    echo ""
    echo "=========================================="
    echo "Next update in 60 seconds..."
    echo "=========================================="

    sleep 60
done
