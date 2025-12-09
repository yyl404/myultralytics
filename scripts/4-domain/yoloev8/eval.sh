#!/bin/bash

# Configuration - should match training script configuration
MODEL_CFG="yolov8l.yaml"
# OUTPUT_DIR can be specified via command line argument or environment variable
# Usage: bash eval.sh [OUTPUT_DIR]
# Example: bash eval.sh runs/yoloe-v8l_4-domain_pretrained_lp
OUTPUT_DIR="${1:-runs/yolov8l_4-domain_pretrained-yoloe_zero-shot}"  # Default value if not specified
EVAL_OUTPUT_DIR="${OUTPUT_DIR}/evaluation_results"
DEVICE=0

# Specify dataset path for each task - should match training script
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
echo "Incremental Learning Model Evaluation"
echo "=========================================="
echo "Output directory: $OUTPUT_DIR"
echo "Evaluation results directory: $EVAL_OUTPUT_DIR"
echo "Number of tasks: $NUM_TASKS"
echo ""

# Evaluate each model on all datasets it has seen
for model_task in $(seq 1 $NUM_TASKS); do
    MODEL_PATH="$OUTPUT_DIR/task-$model_task/best.pt"
    
    echo "=========================================="
    echo "Evaluating Model from Task $model_task"
    echo "Model: $MODEL_PATH"
    echo "=========================================="
    
    # Evaluate on each individual dataset seen so far
    echo "Evaluating on individual datasets..."
    for dataset_task in $(seq 1 $model_task); do
        DATASET_PATH="${TASK_DATASETS[$((dataset_task - 1))]}"
        
        echo "  - Evaluating on Task $dataset_task dataset: $DATASET_PATH"
        
        # Convert dataset class IDs to match model's class mapping
        CONVERTED_DATASET_DIR="$EVAL_OUTPUT_DIR/task_${model_task}_task_${dataset_task}_converted"
        echo "    Converting dataset class IDs..."
        python tools/convert_dataset_class_ids.py \
            --model "$MODEL_PATH" \
            --dataset "$DATASET_PATH" \
            --output_dir "$CONVERTED_DATASET_DIR" \
            --splits train val test
        
        CSV_OUTPUT="$EVAL_OUTPUT_DIR/model_${model_task}_eval_task_${dataset_task}.csv"
        
        python tools/eval.py \
            --model "$MODEL_PATH" \
            --data "$CONVERTED_DATASET_DIR/dataset.yaml" \
            --device "$DEVICE" \
            --save_path "$CSV_OUTPUT" \
            --confusion_matrix_path "$EVAL_OUTPUT_DIR/model_${model_task}_eval_task_${dataset_task}_confusion_matrix.csv" \
            --project "$EVAL_OUTPUT_DIR/model_${model_task}_eval_task_${dataset_task}"
    done
    echo ""
done

echo "=========================================="
echo "Generating Evaluation Tables"
echo "=========================================="

# Generate evaluation tables
python tools/generate_eval_tables.py \
    --eval_dir "$EVAL_OUTPUT_DIR" \
    --num_tasks "$NUM_TASKS" \
    --output_dir "$EVAL_OUTPUT_DIR"
    
echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "=========================================="
echo "Individual datasets evaluation table: $EVAL_OUTPUT_DIR/individual_datasets_eval.csv"
echo ""