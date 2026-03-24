#!/bin/bash
# Run multi-seed comparison experiment
# Usage: bash scripts/alfworld/run_comparison.sh [epochs] [max_tasks] [seeds]
set -e

EPOCHS=${1:-5}
MAX_TASKS=${2:-20}
SEEDS="${3:-42 123 456}"
OUTPUT_DIR=outputs/alfworld/multiseed

# Get project root (parent of scripts/alfworld)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "=== Multi-seed comparison: $EPOCHS epochs x $MAX_TASKS tasks x seeds=[$SEEDS] ==="
echo "Project root: $PROJECT_ROOT"

python3 scripts/alfworld/run_multiseed.py \
    --methods no_memory rag memrl task_memrl \
    --seeds $SEEDS \
    --epochs $EPOCHS \
    --max-tasks $MAX_TASKS \
    --output-dir $OUTPUT_DIR \
    --alpha 0.3 \
    --lam 0.5 \
    --delta 0.3 \
    --k1 5 \
    --k2 3

echo ""
echo "=== Done. Results in $OUTPUT_DIR/ ==="

# Generate plots
echo "Generating plots..."
python3 scripts/alfworld/plot_results.py --input $OUTPUT_DIR/aggregated_results.json --output $OUTPUT_DIR/plots
echo "Plots saved to $OUTPUT_DIR/plots/"
