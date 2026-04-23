#!/bin/bash
set -euo pipefail

# Evaluate final task-2 models for six VOC incremental experiments and summarize old/new/all mAP50.
# Run from the myultralytics repo root.
# Usage:
#   bash eval_6_runs_final.sh [DEVICE]
# Example:
#   bash eval_6_runs_final.sh 0

DEVICE="${1:-0}"
BATCH=1

# Format:
# name|output_dir|split_tag|old_count|cumulative_dataset
EXPERIMENTS=(
  "yolov8x_voc_15_5_fromcls_abr|runs/yolov8x_voc_15_5_fromcls_abr|15_5|15|data/VOC_15_5/task_1-2_cls_20/dataset.yaml"
  "yolov8x_voc_15_5_fromcls_osr|runs/yolov8x_voc_15_5_fromcls_osr|15_5|15|data/VOC_15_5/task_1-2_cls_20/dataset.yaml"
  "yolov8x_voc_19_1_fromcls_abr|runs/yolov8x_voc_19_1_fromcls_abr|19_1|19|data/VOC_19_1/task_1-2_cls_20/dataset.yaml"
  "yolov8x_voc_19_1_fromcls_osr|runs/yolov8x_voc_19_1_fromcls_osr|19_1|19|data/VOC_19_1/task_1-2_cls_20/dataset.yaml"
  "yolov8x_voc_10_10_fromcls_abr|runs/yolov8x_voc_10_10_fromcls_abr|10_10|10|data/VOC_10_10/task_1-2_cls_20/dataset.yaml"
  "yolov8x_voc_10_10_fromcls_osr|runs/yolov8x_voc_10_10_fromcls_osr|10_10|10|data/VOC_10_10/task_1-2_cls_20/dataset.yaml"
)

SUMMARY_INPUT_CSV="runs/final_eval_summary_inputs.csv"
mkdir -p runs

echo "name,output_dir,split_tag,old_count,cumulative_dataset,model_path,converted_dir,result_csv,cm_csv" > "$SUMMARY_INPUT_CSV"

echo "=========================================="
echo "Final Task-2 Evaluation for 6 Experiments"
echo "=========================================="
echo "Device: $DEVICE"
echo

for item in "${EXPERIMENTS[@]}"; do
  IFS='|' read -r NAME OUTPUT_DIR SPLIT_TAG OLD_COUNT CUM_DATASET <<< "$item"

  MODEL_PATH="$OUTPUT_DIR/task-2/best.pt"
  EVAL_DIR="$OUTPUT_DIR/final_eval"
  CONVERTED_DIR="$EVAL_DIR/cumulative_converted"
  RESULT_CSV="$EVAL_DIR/final_cumulative_eval.csv"
  CM_CSV="$EVAL_DIR/final_cumulative_confusion_matrix.csv"

  echo "------------------------------------------"
  echo "Experiment: $NAME"
  echo "Model:      $MODEL_PATH"
  echo "Dataset:    $CUM_DATASET"
  echo "------------------------------------------"

  if [ ! -f "$MODEL_PATH" ]; then
    echo "Warning: model not found, skipping: $MODEL_PATH"
    echo
    continue
  fi

  if [ ! -f "$CUM_DATASET" ]; then
    echo "Warning: cumulative dataset not found, skipping: $CUM_DATASET"
    echo
    continue
  fi

  mkdir -p "$EVAL_DIR"

  python tools/convert_dataset_class_ids.py \
    --model "$MODEL_PATH" \
    --dataset "$CUM_DATASET" \
    --output_dir "$CONVERTED_DIR" \
    --splits train val test

  python tools/eval.py \
    --model "$MODEL_PATH" \
    --data "$CONVERTED_DIR/dataset.yaml" \
    --device "$DEVICE" \
    --batch "$BATCH" \
    --save_path "$RESULT_CSV" \
    --confusion_matrix_path "$CM_CSV" \
    --project "$EVAL_DIR/final_eval_project"

  echo "$NAME,$OUTPUT_DIR,$SPLIT_TAG,$OLD_COUNT,$CUM_DATASET,$MODEL_PATH,$CONVERTED_DIR,$RESULT_CSV,$CM_CSV" >> "$SUMMARY_INPUT_CSV"
  echo

done

python tools/summarize_final_incremental_map50.py \
  --input_csv "$SUMMARY_INPUT_CSV" \
  --output_csv runs/final_incremental_map50_table.csv \
  --output_md runs/final_incremental_map50_table.md

echo "Done."
echo "CSV : runs/final_incremental_map50_table.csv"
echo "MD  : runs/final_incremental_map50_table.md"
