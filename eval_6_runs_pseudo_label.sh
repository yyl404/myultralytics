#!/bin/bash
set -e

DEVICE="${1:-0}"
ROOT_DIR="runs"
SUMMARY_SCRIPT="tools/summarize_6_runs_incremental_map50.py"

RUNS=(
  "yolov8x_voc_15_5_fromcls_abr_pseudo_label"
  "yolov8x_voc_15_5_fromcls_osr_pseudo_label"
  "yolov8x_voc_19_1_fromcls_abr_pseudo_label"
  "yolov8x_voc_19_1_fromcls_osr_pseudo_label"
  "yolov8x_voc_10_10_fromcls_abr_pseudo_label"
  "yolov8x_voc_10_10_fromcls_osr_pseudo_label"
)

echo "=========================================="
echo "Evaluate 6 pseudo-label experiments"
echo "Device: $DEVICE"
echo "=========================================="

for RUN_NAME in "${RUNS[@]}"; do
  OUTPUT_DIR="${ROOT_DIR}/${RUN_NAME}"
  MODEL_PATH="${OUTPUT_DIR}/task-2/best.pt"
  EVAL_OUTPUT_DIR="${OUTPUT_DIR}/evaluation_results"

  if [[ ! -f "$MODEL_PATH" ]]; then
    echo "Warning: model not found, skip: $MODEL_PATH"
    continue
  fi

  mkdir -p "$EVAL_OUTPUT_DIR"

  if [[ "$RUN_NAME" == *"voc_15_5"* ]]; then
    SPLIT_NAME="15_5"
    CUMULATIVE_DATASET="data/VOC_15_5/task_1-2_cls_20/dataset.yaml"
  elif [[ "$RUN_NAME" == *"voc_19_1"* ]]; then
    SPLIT_NAME="19_1"
    CUMULATIVE_DATASET="data/VOC_19_1/task_1-2_cls_20/dataset.yaml"
  elif [[ "$RUN_NAME" == *"voc_10_10"* ]]; then
    SPLIT_NAME="10_10"
    CUMULATIVE_DATASET="data/VOC_10_10/task_1-2_cls_20/dataset.yaml"
  else
    echo "Warning: unknown split for $RUN_NAME, skip."
    continue
  fi

  if [[ ! -f "$CUMULATIVE_DATASET" ]]; then
    echo "Warning: cumulative dataset not found: $CUMULATIVE_DATASET"
    continue
  fi

  echo ""
  echo "=========================================="
  echo "Evaluating: $RUN_NAME"
  echo "Model: $MODEL_PATH"
  echo "Dataset: $CUMULATIVE_DATASET"
  echo "=========================================="

  CONVERTED_DATASET_DIR="${EVAL_OUTPUT_DIR}/cumulative_converted"

  python tools/convert_dataset_class_ids.py \
    --model "$MODEL_PATH" \
    --dataset "$CUMULATIVE_DATASET" \
    --output_dir "$CONVERTED_DATASET_DIR" \
    --splits train val test

  python tools/eval.py \
    --model "$MODEL_PATH" \
    --data "$CONVERTED_DATASET_DIR/dataset.yaml" \
    --device "$DEVICE" \
    --batch 1 \
    --save_path "$EVAL_OUTPUT_DIR/final_cumulative_eval.csv" \
    --confusion_matrix_path "$EVAL_OUTPUT_DIR/final_cumulative_confusion_matrix.csv" \
    --project "$EVAL_OUTPUT_DIR/final_cumulative_eval"
done

echo ""
echo "=========================================="
echo "Summarizing old/new/all mAP50"
echo "=========================================="

python "$SUMMARY_SCRIPT" \
  --runs_root "$ROOT_DIR" \
  --output_csv "$ROOT_DIR/final_incremental_map50_pseudo_label.csv" \
  --output_md "$ROOT_DIR/final_incremental_map50_pseudo_label.md"

echo ""
echo "Done."
echo "CSV : $ROOT_DIR/final_incremental_map50_pseudo_label.csv"
echo "MD  : $ROOT_DIR/final_incremental_map50_pseudo_label.md"