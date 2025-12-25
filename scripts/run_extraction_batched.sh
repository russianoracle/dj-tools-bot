#!/bin/bash
# Batched GPU extraction with auto-restart every N tracks
# This prevents MPS memory fragmentation on Apple Silicon

INPUT="results/user_tracks.csv"
OUTPUT_DIR="results/ultimate_features"
MODE="frames"
BATCH_SIZE=20  # Tracks per batch before restart
PYTHON="/Applications/miniforge3/bin/python3"

# Count total tracks
TOTAL=$(tail -n +2 "$INPUT" | wc -l | tr -d ' ')
echo "=============================================="
echo "BATCHED GPU EXTRACTION"
echo "=============================================="
echo "Total tracks: $TOTAL"
echo "Batch size: $BATCH_SIZE"
echo "Output: $OUTPUT_DIR"
echo ""

# Loop until all tracks are processed
ITERATION=0
while true; do
    ITERATION=$((ITERATION + 1))

    echo "----------------------------------------------"
    echo "Batch $ITERATION starting..."
    echo "----------------------------------------------"

    # Run extraction with max-tracks limit
    PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 \
    PYTHONUNBUFFERED=1 \
    $PYTHON scripts/extract_ultimate_gpu.py \
        --input "$INPUT" \
        --output-dir "$OUTPUT_DIR" \
        --mode "$MODE" \
        --max-tracks "$BATCH_SIZE"

    EXIT_CODE=$?

    # Check if finished (script exits with 0 when all done and merged)
    if [ $EXIT_CODE -eq 0 ]; then
        # Check if frames.pkl exists (means merging completed)
        if [ -f "$OUTPUT_DIR/frames.pkl" ]; then
            echo ""
            echo "=============================================="
            echo "EXTRACTION COMPLETE!"
            echo "=============================================="
            echo "Output: $OUTPUT_DIR/frames.pkl"
            ls -lh "$OUTPUT_DIR/frames.pkl"
            break
        fi
    fi

    # Check checkpoint to see progress
    if [ -f "$OUTPUT_DIR/checkpoint.pkl" ]; then
        PROCESSED=$($PYTHON -c "import pickle; print(len(pickle.load(open('$OUTPUT_DIR/checkpoint.pkl', 'rb'))['processed']))" 2>/dev/null || echo "?")
        REMAINING=$((TOTAL - PROCESSED))
        echo "Progress: $PROCESSED/$TOTAL processed, $REMAINING remaining"

        if [ "$REMAINING" -le 0 ]; then
            echo "All tracks processed, running final merge..."
            # One more run to trigger merge
            PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 $PYTHON scripts/extract_ultimate_gpu.py \
                --input "$INPUT" \
                --output-dir "$OUTPUT_DIR" \
                --mode "$MODE" \
                --max-tracks 1
            break
        fi
    fi

    echo "Batch $ITERATION complete. Restarting for next batch..."
    sleep 2  # Brief pause for memory cleanup
done

echo ""
echo "Done! Output files:"
ls -lh "$OUTPUT_DIR"/*.pkl 2>/dev/null || echo "No .pkl files yet"
