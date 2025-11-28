#!/bin/bash

# Configuration for zero-shot evaluation
MODEL_CFG="yoloe-v8l.yaml"
MODEL_PATH="yoloe-v8l.yaml"
MODEL_WEIGHT="yoloe-v8l-seg.pt"
# OUTPUT_DIR can be specified via command line argument
# Usage: bash eval-zero-shot.sh [OUTPUT_DIR]
# Example: bash eval-zero-shot.sh runs/yoloev8l_4-domain_zero-shot
OUTPUT_DIR="${1:-runs/yoloev8l_4-domain_zero-shot}"  # Default value if not specified
EVAL_OUTPUT_DIR="${OUTPUT_DIR}/evaluation_results"
DEVICE=0

# Specify dataset path for each task
TASK_DATASETS=(
    "data/4-domain/voc/dataset.yaml"
    "data/4-domain/clipart/dataset.yaml"
    "data/4-domain/watercolor/dataset.yaml"
    "data/4-domain/comic/dataset.yaml"
)

NUM_TASKS=${#TASK_DATASETS[@]}

# Create evaluation output directory
mkdir -p "$EVAL_OUTPUT_DIR"

echo "=========================================="
echo "Zero-Shot Model Evaluation"
echo "=========================================="
echo "Output directory: $OUTPUT_DIR"
echo "Evaluation results directory: $EVAL_OUTPUT_DIR"
echo "Model: $MODEL_PATH"
echo "Number of tasks: $NUM_TASKS"
echo ""

# Evaluate pre-trained model on all datasets
# In zero-shot, we use the same pre-trained model for all tasks
# Each dataset is evaluated only once
echo "Evaluating pre-trained model on all datasets..."
for task_id in $(seq 1 $NUM_TASKS); do
    DATASET_PATH="${TASK_DATASETS[$((task_id - 1))]}"
    
    if [ ! -f "$DATASET_PATH" ]; then
        echo "Warning: Dataset not found: $DATASET_PATH"
        continue
    fi
    
    echo "=========================================="
    echo "Evaluating on Task $task_id dataset: $DATASET_PATH"
    echo "=========================================="
    
    # Use original dataset directory directly (no class ID conversion needed)
    VAL_DATASET_DIR="$(dirname "$DATASET_PATH")"
    
    # Use consistent naming: model_1_eval_task_X for compatibility
    CSV_OUTPUT="$EVAL_OUTPUT_DIR/model_1_eval_task_${task_id}.csv"
    
    python tools/eval.py \
        --model "$MODEL_PATH" \
        --weight "$MODEL_WEIGHT" \
        --data "$VAL_DATASET_DIR/dataset.yaml" \
        --device "$DEVICE" \
        --save_path "$CSV_OUTPUT" \
        --confusion_matrix_path "$EVAL_OUTPUT_DIR/model_1_eval_task_${task_id}_confusion_matrix.csv" \
        --project "$EVAL_OUTPUT_DIR/model_1_eval_task_${task_id}"
    
    if [ $? -eq 0 ]; then
        echo "    ✓ Results saved to $CSV_OUTPUT"
    else
        echo "    ✗ Evaluation failed for task $task_id"
    fi
    echo ""
done

echo "=========================================="
echo "Generating Evaluation Tables"
echo "=========================================="

# Generate evaluation tables
python tools/generate_eval_tables.py \
    --eval_dir "$EVAL_OUTPUT_DIR" \
    --num_tasks "$NUM_TASKS" \
    --output_dir "$EVAL_OUTPUT_DIR" \
    --zero_shot

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Evaluation Complete!"
    echo "=========================================="
    echo "Individual datasets evaluation table: $EVAL_OUTPUT_DIR/individual_datasets_eval.csv"
    echo ""
else
    echo "Error: Failed to generate evaluation tables"
    exit 1
fi

