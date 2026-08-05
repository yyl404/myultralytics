#!/bin/bash

set -euo pipefail

: "${MODEL_ADAPTER:?Set MODEL_ADAPTER to a shell adapter}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR before launching}"
: "${METHOD:?Set METHOD before launching}"

if ! declare -p TASK_DATASETS >/dev/null 2>&1; then
    echo "Define TASK_DATASETS as an array of dataset configs" >&2
    exit 1
fi
if (( ${#TASK_DATASETS[@]} == 0 )); then
    echo "TASK_DATASETS must contain at least one dataset config" >&2
    exit 1
fi
if [[ ! -f "$MODEL_ADAPTER" ]]; then
    echo "Model adapter not found: $MODEL_ADAPTER" >&2
    exit 1
fi

# Model adapters own model-specific preparation, training, and artifact generation.
source "$MODEL_ADAPTER"
required_adapter_functions=(
    model_adapter_validate
    model_adapter_initialize
    model_adapter_prepare_task
    model_adapter_train_task
    model_adapter_finalize_task
)
for function_name in "${required_adapter_functions[@]}"; do
    if ! declare -F "$function_name" >/dev/null; then
        echo "Model adapter '$MODEL_ADAPTER' does not define $function_name" >&2
        exit 1
    fi
done

START_TASK="${START_TASK:-1}"
END_TASK="${END_TASK:-${#TASK_DATASETS[@]}}"
if (( START_TASK < 1 || START_TASK > ${#TASK_DATASETS[@]} )); then
    echo "START_TASK must be between 1 and ${#TASK_DATASETS[@]}" >&2
    exit 1
fi
if (( END_TASK < START_TASK || END_TASK > ${#TASK_DATASETS[@]} )); then
    echo "END_TASK must be between START_TASK (${START_TASK}) and ${#TASK_DATASETS[@]}" >&2
    exit 1
fi

model_adapter_validate
model_adapter_initialize

for task_index in "${!TASK_DATASETS[@]}"; do
    TASK_ID=$((task_index + 1))
    if (( TASK_ID < START_TASK || TASK_ID > END_TASK )); then
        continue
    fi

    DATASET_PATH="${TASK_DATASETS[$task_index]}"
    TASK_DIR="${OUTPUT_DIR}/task-${TASK_ID}"
    PREVIOUS_TASK_DIR=""
    if (( TASK_ID > 1 )); then
        PREVIOUS_TASK_DIR="${OUTPUT_DIR}/task-$((TASK_ID - 1))"
    fi

    if [[ ! -f "$DATASET_PATH" ]]; then
        echo "Task ${TASK_ID} dataset config not found: $DATASET_PATH" >&2
        exit 1
    fi
    mkdir -p "$TASK_DIR"

    model_adapter_prepare_task
    model_adapter_train_task
    model_adapter_finalize_task
done
