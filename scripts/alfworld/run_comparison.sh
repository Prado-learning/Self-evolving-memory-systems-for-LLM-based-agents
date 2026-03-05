#!/bin/bash
# Run all 4 methods for comparison
set -e
export ALFWORLD_DATA=/home/user/alfworld_data

EPOCHS=3
MAX_TASKS=20
OUTPUT_DIR=outputs/alfworld

cd /home/user/lvhuanzhu/AutoEvolve

echo "=== Starting 4-method comparison: $EPOCHS epochs x $MAX_TASKS tasks ==="

for METHOD in no_memory rag memrl task_memrl; do
    echo ""
    echo "============================="
    echo "Running method: $METHOD"
    echo "============================="
    python3 scripts/alfworld/run_experiment.py \
        --method $METHOD \
        --epochs $EPOCHS \
        --max-tasks $MAX_TASKS \
        --output-dir $OUTPUT_DIR \
        --alpha 0.3 \
        --lam 0.5 \
        --delta 0.3 \
        --k1 5 \
        --k2 3
    echo "Done: $METHOD"
done

echo ""
echo "=== All methods complete ==="
echo "Results in $OUTPUT_DIR/"
