#!/bin/bash

# Ultralytics model-family adapter for scripts/run_incremental.sh.

model_adapter_validate() {
    : "${MODEL_CONFIG:?Set MODEL_CONFIG for the Ultralytics adapter}"
    : "${MODEL_ID:?Set MODEL_ID for output naming and diagnostics}"

    case "$METHOD" in
        naive|pseudo_label|pseudo_label+ewc|pseudo_label+espreg|pseudo_label+dist+espreg|pseudo_label+nsgp|pseudo_label+nsgp+repre)
            ;;
        *)
            echo "Unsupported Ultralytics incremental method: $METHOD" >&2
            exit 1
            ;;
    esac
}

model_adapter_initialize() {
    EPOCHS="${EPOCHS:-100}"
    BATCH_SIZE="${BATCH_SIZE:-16}"
    IMGSZ="${IMGSZ:-640}"
    WORKERS="${WORKERS:-8}"
    DEVICE="${DEVICE:-0}"
    CONF_THRESHOLD="${CONF_THRESHOLD:-0.25}"
    FILTER_IOU_THRESHOLD="${FILTER_IOU_THRESHOLD:-0.5}"
    EWC_LOSS_WEIGHT="${EWC_LOSS_WEIGHT:-100.0}"
    ESPREG_LOSS_WEIGHT="${ESPREG_LOSS_WEIGHT:-100.0}"
    DIST_LOSS_WEIGHT="${DIST_LOSS_WEIGHT:-100.0}"
    if ! declare -p EXTRA_TRAIN_ARGS >/dev/null 2>&1; then
        EXTRA_TRAIN_ARGS=()
    fi

    PREVIOUS_MODEL=""
    PREVIOUS_PCA=""
    PREVIOUS_PROTOTYPES=""
    PREVIOUS_IMPORTANCE=""
    if (( START_TASK > 1 )); then
        previous_task_dir="${OUTPUT_DIR}/task-$((START_TASK - 1))"
        PREVIOUS_MODEL="${previous_task_dir}/best.pt"
        PREVIOUS_PCA="${previous_task_dir}/pca_cache.pkl"
        PREVIOUS_PROTOTYPES="${previous_task_dir}/repre_prototypes.pt"
        PREVIOUS_IMPORTANCE="${previous_task_dir}/importance.pth"
        if [[ ! -f "$PREVIOUS_MODEL" ]]; then
            echo "Previous task model not found: $PREVIOUS_MODEL" >&2
            exit 1
        fi
        if [[ "$METHOD" == "pseudo_label+ewc" || "$METHOD" == "pseudo_label+nsgp" \
            || "$METHOD" == "pseudo_label+nsgp+repre" ]] \
            && [[ ! -f "$PREVIOUS_IMPORTANCE" ]]; then
            echo "Previous task importance artifact not found: $PREVIOUS_IMPORTANCE" >&2
            exit 1
        fi
        if [[ "$METHOD" == *"espreg"* || "$METHOD" == "pseudo_label+nsgp" \
            || "$METHOD" == "pseudo_label+nsgp+repre" ]] \
            && [[ ! -f "$PREVIOUS_PCA" ]]; then
            echo "Previous task PCA artifact not found: $PREVIOUS_PCA" >&2
            exit 1
        fi
        if [[ "$METHOD" == "pseudo_label+nsgp+repre" && ! -f "$PREVIOUS_PROTOTYPES" ]]; then
            echo "Previous task RePRE artifact not found: $PREVIOUS_PROTOTYPES" >&2
            exit 1
        fi
    fi
}

model_adapter_prepare_task() {
    TRAINER_ARGS=()
    WEIGHT_ARGS=()
    FREEZE_ARGS=()

    if (( TASK_ID == 1 )); then
        TRAIN_MODEL="$MODEL_CONFIG"
        TRAIN_DATA="$DATASET_PATH"
        if [[ -n "${MODEL_WEIGHTS:-}" ]]; then
            WEIGHT_ARGS=(--weight "$MODEL_WEIGHTS")
        fi
    else
        if [[ ! -f "$PREVIOUS_MODEL" ]]; then
            echo "Previous task model not found: $PREVIOUS_MODEL" >&2
            exit 1
        fi

        expanded_model="${TASK_DIR}/task-$((TASK_ID - 1))-best-expanded.pt"
        python tools/expand_model_head.py \
            --model "$PREVIOUS_MODEL" \
            --model_cfg "$MODEL_CONFIG" \
            --dataset "$DATASET_PATH" \
            --save_path "$expanded_model"

        converted_data="${TASK_DIR}/task_${TASK_ID}_id_converted"
        python tools/convert_dataset_class_ids.py \
            --model "$expanded_model" \
            --dataset "$DATASET_PATH" \
            --output_dir "$converted_data" \
            --workers "$WORKERS"

        TRAIN_MODEL="$expanded_model"
        TRAIN_DATA="${converted_data}/dataset.yaml"
        TRAINER_ARGS=(--trainer antiforget)

        if [[ "$METHOD" == "pseudo_label+ewc" ]]; then
            expanded_importance="${TASK_DIR}/task-$((TASK_ID - 1))-importance-expanded.pth"
            python tools/expand_importance.py \
                --old_importance "$PREVIOUS_IMPORTANCE" \
                --old_model "$PREVIOUS_MODEL" \
                --new_model "$expanded_model" \
                --save_path "$expanded_importance" \
                --copy_importance_init
            PREVIOUS_IMPORTANCE="$expanded_importance"
        fi

        case "$METHOD" in
            naive)
                TRAINER_ARGS=()
                ;;
            pseudo_label)
                TRAINER_ARGS+=(
                    --pseudo_label True
                    --conf_threshold "$CONF_THRESHOLD"
                    --filter_iou_threshold "$FILTER_IOU_THRESHOLD"
                )
                ;;
            pseudo_label+ewc)
                TRAINER_ARGS+=(
                    --pseudo_label True
                    --conf_threshold "$CONF_THRESHOLD"
                    --filter_iou_threshold "$FILTER_IOU_THRESHOLD"
                    --ewc True
                    --importance_path "$PREVIOUS_IMPORTANCE"
                    --ewc_loss_weight "$EWC_LOSS_WEIGHT"
                )
                ;;
            pseudo_label+espreg)
                TRAINER_ARGS+=(
                    --pseudo_label True
                    --conf_threshold "$CONF_THRESHOLD"
                    --filter_iou_threshold "$FILTER_IOU_THRESHOLD"
                    --espreg True
                    --pca_cache_path "$PREVIOUS_PCA"
                    --espreg_loss_weight "$ESPREG_LOSS_WEIGHT"
                )
                ;;
            pseudo_label+dist+espreg)
                TRAINER_ARGS+=(
                    --pseudo_label True
                    --conf_threshold "$CONF_THRESHOLD"
                    --filter_iou_threshold "$FILTER_IOU_THRESHOLD"
                    --distillation True
                    --dist_loss_weight "$DIST_LOSS_WEIGHT"
                    --espreg True
                    --pca_cache_path "$PREVIOUS_PCA"
                    --espreg_loss_weight "$ESPREG_LOSS_WEIGHT"
                )
                ;;
            pseudo_label+nsgp|pseudo_label+nsgp+repre)
                TRAINER_ARGS+=(
                    --pseudo_label True
                    --conf_threshold "$CONF_THRESHOLD"
                    --filter_iou_threshold "$FILTER_IOU_THRESHOLD"
                    --nsgp True
                    --nsgp_flexibility 1.0
                    --pca_cache_path "$PREVIOUS_PCA"
                    --ewc True
                    --importance_path "$PREVIOUS_IMPORTANCE"
                    --ewc_loss_weight 1000.0
                    --ewc_internal_scale 2.0
                    --ewc_average_parameters False
                )
                if [[ "$METHOD" == "pseudo_label+nsgp+repre" ]]; then
                    TRAINER_ARGS+=(
                        --repre True
                        --repre_prototypes "$PREVIOUS_PROTOTYPES"
                        --repre_loss_weight 1.0
                    )
                fi
                ;;
        esac
    fi

    if declare -p TASK_FREEZE_LAYERS >/dev/null 2>&1; then
        freeze_index=$((TASK_ID - 1))
        if (( freeze_index < ${#TASK_FREEZE_LAYERS[@]} )) && [[ -n "${TASK_FREEZE_LAYERS[$freeze_index]}" ]]; then
            FREEZE_ARGS=(--freeze "${TASK_FREEZE_LAYERS[$freeze_index]}")
        fi
    fi
}

model_adapter_train_task() {
    optimizer_args=()
    if [[ "$METHOD" == "pseudo_label+nsgp" || "$METHOD" == "pseudo_label+nsgp+repre" ]]; then
        if [[ "${DATASET_FAMILY:-}" == "coco" ]]; then
            optimizer_args=(--optimizer AdamW --lr0 0.00005 --weight_decay 0.01)
        else
            optimizer_args=(--optimizer SGD --lr0 0.02 --momentum 0.9 --weight_decay 0.001)
        fi
    fi

    python tools/train.py \
        --model "$TRAIN_MODEL" \
        --data "$TRAIN_DATA" \
        --save_path "${TASK_DIR}/best.pt" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --imgsz "$IMGSZ" \
        --workers "$WORKERS" \
        --device "$DEVICE" \
        --project "$TASK_DIR" \
        "${optimizer_args[@]}" \
        "${WEIGHT_ARGS[@]}" \
        "${FREEZE_ARGS[@]}" \
        "${TRAINER_ARGS[@]}" \
        "${EXTRA_TRAIN_ARGS[@]}"

    PREVIOUS_MODEL="${TASK_DIR}/best.pt"
}

model_adapter_finalize_task() {
    if [[ "$METHOD" == "pseudo_label+ewc" || "$METHOD" == "pseudo_label+nsgp" \
        || "$METHOD" == "pseudo_label+nsgp+repre" ]]; then
        importance_args=()
        if [[ "$METHOD" == "pseudo_label+nsgp" || "$METHOD" == "pseudo_label+nsgp+repre" ]]; then
            importance_args=(--module_pattern "*bn*" --raw)
        fi
        python tools/cal_importance.py \
            --model "$PREVIOUS_MODEL" \
            --dataset "$TRAIN_DATA" \
            --save_path "${TASK_DIR}/importance.pth" \
            --batch_size "$BATCH_SIZE" \
            --workers "$WORKERS" \
            --device "$DEVICE" \
            "${importance_args[@]}"
        PREVIOUS_IMPORTANCE="${TASK_DIR}/importance.pth"
    fi

    if [[ "$METHOD" == *"espreg"* || "$METHOD" == "pseudo_label+nsgp" \
        || "$METHOD" == "pseudo_label+nsgp+repre" ]]; then
        history_args=()
        covariance_args=()
        if [[ -n "$PREVIOUS_PCA" ]]; then
            history_args=(--load_hist "$PREVIOUS_PCA")
        fi
        if [[ "$METHOD" == "pseudo_label+nsgp" || "$METHOD" == "pseudo_label+nsgp+repre" ]]; then
            covariance_args=(--uncentered --sample_num 0 --batch_size "$BATCH_SIZE")
        fi
        python tools/pca.py \
            --model "$PREVIOUS_MODEL" \
            --dataset "$DATASET_PATH" \
            --save_path "${TASK_DIR}/pca_cache.pkl" \
            --device "$DEVICE" \
            --exclude_head \
            "${covariance_args[@]}" \
            "${history_args[@]}"
        PREVIOUS_PCA="${TASK_DIR}/pca_cache.pkl"
    fi

    if [[ "$METHOD" == "pseudo_label+nsgp+repre" ]]; then
        history_args=()
        if [[ -n "$PREVIOUS_PROTOTYPES" ]]; then
            history_args=(--load_hist "$PREVIOUS_PROTOTYPES")
        fi
        python tools/generate_prototypes.py \
            --model "$PREVIOUS_MODEL" \
            --data "$DATASET_PATH" \
            --output "${TASK_DIR}/repre_prototypes.pt" \
            --device "$DEVICE" \
            --imgsz "$IMGSZ" \
            --num_protos 10 \
            --selection density \
            --radius 0.6 \
            "${history_args[@]}"
        PREVIOUS_PROTOTYPES="${TASK_DIR}/repre_prototypes.pt"
    fi
}
