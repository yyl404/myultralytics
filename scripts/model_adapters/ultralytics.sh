#!/bin/bash

# Ultralytics model-family adapter for scripts/run_incremental.sh.
#
# METHOD is a '+'-joined list of method components (e.g. "pseudo_label+ewc").
# Components are matched as whole tokens, so any combination is allowed. There
# is no up-front legality check: an unimplemented component fails when the
# method is resolved in model_adapter_prepare_task, and an incompatible one
# fails at the point where its inputs/artifacts are used.

method_has() {
    # Check whether METHOD contains an exact '+'-separated component.
    [[ "+${METHOD}+" == *"+$1+"* ]]
}

method_needs_importance() {
    method_has ewc || method_has nsgp
}

method_needs_pca() {
    method_has espreg || method_has nsgp
}

method_needs_prototypes() {
    method_has repre
}

method_needs_replay() {
    method_has replay
}

model_adapter_validate() {
    : "${MODEL_CONFIG:?Set MODEL_CONFIG for the Ultralytics adapter}"
    : "${MODEL_ID:?Set MODEL_ID for output naming and diagnostics}"
}

model_adapter_check_method_components() {
    # Fail on components with no implementation; combinations are not restricted.
    local component
    for component in ${METHOD//+/ }; do
        case "$component" in
            naive|bpf|pseudo_label|ewc|l2|dist|espreg|nsgp|repre|replay)
                ;;
            *)
                echo "Unimplemented Ultralytics method component: '$component' (METHOD='$METHOD')" >&2
                exit 1
                ;;
        esac
    done
}

model_adapter_regenerate_missing_artifacts() {
    local previous_dataset="$1"

    if method_needs_importance && [[ ! -f "$PREVIOUS_IMPORTANCE" ]]; then
        local -a importance_args=()
        if method_has nsgp; then
            importance_args=(--scope normalization)
        fi
        echo "Regenerating missing importance artifact: $PREVIOUS_IMPORTANCE"
        python tools/compute_importance.py \
            --model "$PREVIOUS_MODEL" \
            --dataset "$previous_dataset" \
            --save_path "$PREVIOUS_IMPORTANCE" \
            --batch_size "$BATCH_SIZE" \
            --workers "$WORKERS" \
            --device "$TOOL_DEVICE" \
            "${importance_args[@]}"
    fi

    if method_needs_pca && [[ ! -f "$PREVIOUS_PCA" ]]; then
        local -a covariance_args=()
        if method_has nsgp; then
            covariance_args=(--uncentered --sample_num 0 --batch_size "$BATCH_SIZE")
        fi
        echo "Regenerating missing PCA artifact: $PREVIOUS_PCA"
        python tools/pca.py \
            --model "$PREVIOUS_MODEL" \
            --dataset "$previous_dataset" \
            --save_path "$PREVIOUS_PCA" \
            --device "$TOOL_DEVICE" \
            --exclude_head \
            "${covariance_args[@]}"
    fi

    if method_needs_prototypes && [[ ! -f "$PREVIOUS_PROTOTYPES" ]]; then
        echo "Regenerating missing RePRE prototypes: $PREVIOUS_PROTOTYPES"
        python tools/generate_prototypes.py \
            --model "$PREVIOUS_MODEL" \
            --data "$previous_dataset" \
            --output "$PREVIOUS_PROTOTYPES" \
            --device "$TOOL_DEVICE" \
            --imgsz "$IMGSZ" \
            --num_protos 10 \
            --selection density \
            --radius 0.6
    fi
}

model_adapter_initialize() {
    EPOCHS="${EPOCHS:-100}"
    BATCH_SIZE="${BATCH_SIZE:-16}"
    IMGSZ="${IMGSZ:-640}"
    WORKERS="${WORKERS:-8}"
    DEVICE="${DEVICE:-0}"
    # Artifact tools (importance / PCA / prototypes) run single-device; DEVICE may be a
    # multi-GPU list for training (e.g. "0,1"), so tools use its first entry by default.
    TOOL_DEVICE="${TOOL_DEVICE:-${DEVICE%%,*}}"
    CONF_THRESHOLD="${CONF_THRESHOLD:-0.25}"
    FILTER_IOU_THRESHOLD="${FILTER_IOU_THRESHOLD:-0.5}"
    EWC_LOSS_WEIGHT="${EWC_LOSS_WEIGHT:-100.0}"
    L2_LOSS_WEIGHT="${L2_LOSS_WEIGHT:-100.0}"
    ESPREG_LOSS_WEIGHT="${ESPREG_LOSS_WEIGHT:-100.0}"
    DIST_LOSS_WEIGHT="${DIST_LOSS_WEIGHT:-100.0}"
    DIST_TOPK="${DIST_TOPK:-1}"
    RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
    BPF_INTERIM_EPOCHS="${BPF_INTERIM_EPOCHS:-$EPOCHS}"
    BPF_SCORE_THRESHOLD="${BPF_SCORE_THRESHOLD:-0.75}"
    BPF_NMS_THRESHOLD="${BPF_NMS_THRESHOLD:-0.3}"
    BPF_IOU_LOW="${BPF_IOU_LOW:-0.4}"
    BPF_IOU_HIGH="${BPF_IOU_HIGH:-0.7}"
    BPF_LOW_WEIGHT="${BPF_LOW_WEIGHT:-1.0}"
    BPF_HIGH_WEIGHT="${BPF_HIGH_WEIGHT:-0.3}"
    BPF_OBJECT_TOPK="${BPF_OBJECT_TOPK:-0.1}"
    BPF_ATTENTION_TOPK="${BPF_ATTENTION_TOPK:-0.1}"
    BPF_FUTURE_IOU="${BPF_FUTURE_IOU:-0.1}"
    BPF_DWF_WEIGHT="${BPF_DWF_WEIGHT:-0.15}"
    REPLAY_SAMPLE_NUM="${REPLAY_SAMPLE_NUM:-100}"
    REPLAY_STRATEGY="${REPLAY_STRATEGY:-random}"
    REPLAY_LOSS_WEIGHT="${REPLAY_LOSS_WEIGHT:-1.0}"
    REPLAY_SEED="${REPLAY_SEED:-0}"
    if ! declare -p EXTRA_TRAIN_ARGS >/dev/null 2>&1; then
        EXTRA_TRAIN_ARGS=()
    fi

    PREVIOUS_MODEL=""
    PREVIOUS_PCA=""
    PREVIOUS_PROTOTYPES=""
    PREVIOUS_IMPORTANCE=""
    PREVIOUS_REPLAY=""
    if (( START_TASK > 1 )); then
        previous_task_dir="${OUTPUT_DIR}/task-$((START_TASK - 1))"
        PREVIOUS_MODEL="${previous_task_dir}/best.pt"
        PREVIOUS_PCA="${previous_task_dir}/pca_cache.pkl"
        PREVIOUS_PROTOTYPES="${previous_task_dir}/repre_prototypes.pt"
        PREVIOUS_IMPORTANCE="${previous_task_dir}/importance.pth"
        PREVIOUS_REPLAY="${previous_task_dir}/replay_dataset"
        if [[ ! -f "$PREVIOUS_MODEL" ]]; then
            echo "Previous task model not found: $PREVIOUS_MODEL" >&2
            exit 1
        fi
        previous_dataset="${TASK_DATASETS[$((START_TASK - 2))]}"
        if [[ ! -f "$previous_dataset" ]]; then
            echo "Previous task dataset config not found: $previous_dataset" >&2
            exit 1
        fi
        model_adapter_regenerate_missing_artifacts "$previous_dataset"
        if method_needs_importance && [[ ! -f "$PREVIOUS_IMPORTANCE" ]]; then
            echo "Previous task importance artifact not found: $PREVIOUS_IMPORTANCE" >&2
            exit 1
        fi
        if method_needs_pca && [[ ! -f "$PREVIOUS_PCA" ]]; then
            echo "Previous task PCA artifact not found: $PREVIOUS_PCA" >&2
            exit 1
        fi
        if method_needs_prototypes && [[ ! -f "$PREVIOUS_PROTOTYPES" ]]; then
            echo "Previous task RePRE artifact not found: $PREVIOUS_PROTOTYPES" >&2
            exit 1
        fi
        # Replay data is cumulative, so it cannot be regenerated for a single task; require the
        # previous task's replay dataset (which embeds its own history) to exist.
        if method_needs_replay && [[ ! -f "$PREVIOUS_REPLAY/dataset.yaml" ]]; then
            echo "Previous task replay dataset not found: $PREVIOUS_REPLAY" >&2
            exit 1
        fi
    fi
}

model_adapter_prepare_task() {
    model_adapter_check_method_components

    TRAINER_ARGS=()
    WEIGHT_ARGS=()
    FREEZE_ARGS=()
    REFERENCE_MODEL=""

    if (( TASK_ID == 1 )); then
        TRAIN_MODEL="$MODEL_CONFIG"
        TRAIN_DATA="$DATASET_PATH"
        if [[ -n "${MODEL_WEIGHTS:-}" ]]; then
            WEIGHT_ARGS=(--weight "$MODEL_WEIGHTS")
        fi
        if method_has bpf; then
            TRAINER_ARGS=(
                --trainer bpf
                --bpf True
                --bpf_future True
                --bpf_object_topk "$BPF_OBJECT_TOPK"
                --bpf_attention_topk "$BPF_ATTENTION_TOPK"
                --bpf_future_iou "$BPF_FUTURE_IOU"
            )
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
        REFERENCE_MODEL="$expanded_model"
        TRAIN_DATA="${converted_data}/dataset.yaml"
        TRAINER_ARGS=(--trainer antiforget)

        if method_has ewc && ! method_has nsgp; then
            expanded_importance="${TASK_DIR}/task-$((TASK_ID - 1))-importance-expanded.pth"
            python tools/expand_importance.py \
                --old_importance "$PREVIOUS_IMPORTANCE" \
                --old_model "$PREVIOUS_MODEL" \
                --new_model "$expanded_model" \
                --save_path "$expanded_importance"
            PREVIOUS_IMPORTANCE="$expanded_importance"
        fi

        if method_has bpf; then
            bpf_source_model="$PREVIOUS_MODEL"
            bpf_interim_model="${TASK_DIR}/interim.pt"
            interim_weight_args=()
            if [[ -n "${MODEL_WEIGHTS:-}" ]]; then
                interim_weight_args=(--weight "$MODEL_WEIGHTS")
            fi
            python tools/train.py \
                --model "$MODEL_CONFIG" \
                --data "$DATASET_PATH" \
                --save_path "$bpf_interim_model" \
                --epochs "$BPF_INTERIM_EPOCHS" \
                --batch_size "$BATCH_SIZE" \
                --imgsz "$IMGSZ" \
                --workers "$WORKERS" \
                --device "$DEVICE" \
                --project "${TASK_DIR}/interim" \
                "${interim_weight_args[@]}" \
                "${EXTRA_TRAIN_ARGS[@]}"
            if [[ ! -f "$bpf_interim_model" ]]; then
                echo "BPF interim model not found after training: $bpf_interim_model" >&2
                exit 1
            fi
            TRAINER_ARGS=(
                --trainer bpf
                --bpf True
                --bpf_past True
                --bpf_dwf True
                --bpf_source_model "$bpf_source_model"
                --bpf_interim_model "$bpf_interim_model"
                --bpf_score_threshold "$BPF_SCORE_THRESHOLD"
                --bpf_nms_threshold "$BPF_NMS_THRESHOLD"
                --bpf_iou_low "$BPF_IOU_LOW"
                --bpf_iou_high "$BPF_IOU_HIGH"
                --bpf_low_weight "$BPF_LOW_WEIGHT"
                --bpf_high_weight "$BPF_HIGH_WEIGHT"
                --bpf_dwf_weight "$BPF_DWF_WEIGHT"
            )
        fi

        if method_has pseudo_label; then
            TRAINER_ARGS+=(
                --pseudo_label True
                --reference_model "$REFERENCE_MODEL"
                --conf_threshold "$CONF_THRESHOLD"
                --filter_iou_threshold "$FILTER_IOU_THRESHOLD"
            )
        fi

        if method_has ewc && ! method_has nsgp; then
            TRAINER_ARGS+=(
                --ewc True
                --importance_path "$PREVIOUS_IMPORTANCE"
                --ewc_loss_weight "$EWC_LOSS_WEIGHT"
            )
        fi

        if method_has l2; then
            TRAINER_ARGS+=(
                --l2 True
                --l2_loss_weight "$L2_LOSS_WEIGHT"
            )
        fi

        if method_has dist; then
            TRAINER_ARGS+=(
                --distillation True
                --dist_loss_weight "$DIST_LOSS_WEIGHT"
                --dist_topk "$DIST_TOPK"
            )
        fi

        if method_has espreg; then
            TRAINER_ARGS+=(
                --espreg True
                --pca_cache_path "$PREVIOUS_PCA"
                --espreg_loss_weight "$ESPREG_LOSS_WEIGHT"
            )
        fi

        if method_has nsgp; then
            TRAINER_ARGS+=(
                --nsgp True
                --nsgp_flexibility 1.0
                --pca_cache_path "$PREVIOUS_PCA"
                --ewc True
                --importance_path "$PREVIOUS_IMPORTANCE"
                --ewc_loss_weight 2000.0
            )
        fi

        if method_has repre; then
            TRAINER_ARGS+=(
                --repre True
                --repre_prototypes "$PREVIOUS_PROTOTYPES"
                --repre_loss_weight 1.0
            )
        fi

        if method_has replay; then
            if [[ ! -f "$PREVIOUS_REPLAY/dataset.yaml" ]]; then
                echo "Previous task replay dataset not found: $PREVIOUS_REPLAY" >&2
                exit 1
            fi
            # Replay labels live in the previous task's class-id space. Head expansion keeps
            # existing class ids and appends unseen classes after them; remap the replay
            # dataset into the expanded model's class-id space, same as the task dataset above.
            converted_replay="${TASK_DIR}/replay_id_converted"
            python tools/convert_dataset_class_ids.py \
                --model "$expanded_model" \
                --dataset "$PREVIOUS_REPLAY/dataset.yaml" \
                --output_dir "$converted_replay" \
                --workers "$WORKERS"
            TRAINER_ARGS+=(
                --replay True
                --replay_data "$converted_replay/dataset.yaml"
                --replay_loss_weight "$REPLAY_LOSS_WEIGHT"
            )
        fi
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
    if method_has nsgp; then
        if [[ "${DATASET_FAMILY:-}" == "coco" ]]; then
            optimizer_args=(--optimizer AdamW --lr0 0.00005 --weight_decay 0.01)
        else
            optimizer_args=(--optimizer SGD --lr0 0.02 --momentum 0.9 --weight_decay 0.001)
        fi
    fi

    # Resume from an in-training checkpoint (weights/last.pt or weights/epoch*.pt) when provided.
    # The checkpoint must match this dataset/task/method; this is the user's responsibility.
    resume_args=()
    if [[ -n "$RESUME_CHECKPOINT" ]]; then
        resume_args=(--resume_checkpoint "$RESUME_CHECKPOINT")
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
        "${resume_args[@]}" \
        "${WEIGHT_ARGS[@]}" \
        "${FREEZE_ARGS[@]}" \
        "${TRAINER_ARGS[@]}" \
        "${EXTRA_TRAIN_ARGS[@]}"

    PREVIOUS_MODEL="${TASK_DIR}/best.pt"
}

model_adapter_finalize_task() {
    if method_needs_importance; then
        importance_args=()
        history_args=()
        reference_args=()
        if method_has nsgp; then
            importance_args=(--scope normalization)
        fi
        if [[ -n "$PREVIOUS_IMPORTANCE" ]]; then
            history_args=(--load_hist "$PREVIOUS_IMPORTANCE")
        fi
        if [[ -n "$REFERENCE_MODEL" ]]; then
            reference_args=(
                --reference_model "$REFERENCE_MODEL"
                --conf_threshold "$CONF_THRESHOLD"
                --filter_iou_threshold "$FILTER_IOU_THRESHOLD"
            )
        fi
        python tools/compute_importance.py \
            --model "$PREVIOUS_MODEL" \
            --dataset "$TRAIN_DATA" \
            --save_path "${TASK_DIR}/importance.pth" \
            --batch_size "$BATCH_SIZE" \
            --workers "$WORKERS" \
            --device "$TOOL_DEVICE" \
            "${importance_args[@]}" \
            "${history_args[@]}" \
            "${reference_args[@]}"
        PREVIOUS_IMPORTANCE="${TASK_DIR}/importance.pth"
    fi

    if method_needs_pca; then
        history_args=()
        covariance_args=()
        if [[ -n "$PREVIOUS_PCA" ]]; then
            history_args=(--load_hist "$PREVIOUS_PCA")
        fi
        if method_has nsgp; then
            covariance_args=(--uncentered --sample_num 0 --batch_size "$BATCH_SIZE")
        fi
        python tools/pca.py \
            --model "$PREVIOUS_MODEL" \
            --dataset "$DATASET_PATH" \
            --save_path "${TASK_DIR}/pca_cache.pkl" \
            --device "$TOOL_DEVICE" \
            --exclude_head \
            "${covariance_args[@]}" \
            "${history_args[@]}"
        PREVIOUS_PCA="${TASK_DIR}/pca_cache.pkl"
    fi

    if method_needs_prototypes; then
        history_args=()
        if [[ -n "$PREVIOUS_PROTOTYPES" ]]; then
            history_args=(--load_hist "$PREVIOUS_PROTOTYPES")
        fi
        python tools/generate_prototypes.py \
            --model "$PREVIOUS_MODEL" \
            --data "$DATASET_PATH" \
            --output "${TASK_DIR}/repre_prototypes.pt" \
            --device "$TOOL_DEVICE" \
            --imgsz "$IMGSZ" \
            --num_protos 10 \
            --selection density \
            --radius 0.6 \
            "${history_args[@]}"
        PREVIOUS_PROTOTYPES="${TASK_DIR}/repre_prototypes.pt"
    fi

    if method_needs_replay; then
        history_args=()
        if [[ -n "$PREVIOUS_REPLAY" ]]; then
            history_args=(--load_hist "$PREVIOUS_REPLAY")
        fi
        # TRAIN_DATA labels are in this task's class-id space (task-1 local IDs; later tasks use
        # the id-converted dataset), and select_replay_samples remaps carried-over history labels
        # into the same space by class name, so each task's replay_dataset stays self-consistent.
        # The next task's prepare step remaps the replay dataset into the expanded model's
        # class-id space before training (see model_adapter_prepare_task).
        python tools/select_replay_samples.py \
            --dataset "$TRAIN_DATA" \
            --output_dir "${TASK_DIR}/replay_dataset" \
            --num "$REPLAY_SAMPLE_NUM" \
            --strategy "$REPLAY_STRATEGY" \
            --seed "$REPLAY_SEED" \
            "${history_args[@]}"
        PREVIOUS_REPLAY="${TASK_DIR}/replay_dataset"
    fi
}
