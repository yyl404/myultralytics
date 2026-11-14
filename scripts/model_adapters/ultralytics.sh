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
    method_has ewc
}

method_needs_pca() {
    method_has espreg || method_has nsgp
}

method_needs_prototypes() {
    method_has repre
}

method_needs_patch_prototypes() {
    method_has prad || method_has prd
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
            naive|bpf|pseudo_label|ewc|l2|dist|lwf|espreg|nsgp|repre|replay|prad|prd)
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

    if method_needs_patch_prototypes && [[ ! -f "$PREVIOUS_PATCH_PROTOTYPES" ]]; then
        echo "Regenerating missing patch prototype artifact: $PREVIOUS_PATCH_PROTOTYPES"
        python tools/generate_patch_prototypes.py \
            --model "$PREVIOUS_MODEL" \
            --data "$previous_dataset" \
            --output "$PREVIOUS_PATCH_PROTOTYPES" \
            --device "$TOOL_DEVICE" \
            --imgsz "$IMGSZ" \
            --batch_size "$BATCH_SIZE" \
            --workers "$WORKERS" \
            --pca_dims "$PATCH_PCA_DIMS" \
            --max_centers "$PATCH_PROTO_MAX_CENTERS" \
            --split_ratio "$PATCH_PROTO_SPLIT_RATIO" \
            --max_samples_per_class "$PATCH_PROTO_MAX_SAMPLES" \
            --seed "$PATCH_PROTO_SEED"
    fi

    if method_needs_pca && [[ ! -f "$PREVIOUS_PCA" ]]; then
        local -a bbox_args=()
        if method_has espreg; then
            bbox_args=(--use_bbox)
        fi
        echo "Regenerating missing PCA artifact: $PREVIOUS_PCA"
        python tools/pca.py \
            --model "$PREVIOUS_MODEL" \
            --dataset "$previous_dataset" \
            --save_path "$PREVIOUS_PCA" \
            --device "$TOOL_DEVICE" \
            --pca_mode "$PCA_MODE" \
            --sample_num "$PCA_SAMPLE_NUM" \
            --batch_size "$BATCH_SIZE" \
            --imgsz "$IMGSZ" \
            --workers "$WORKERS" \
            --seed "$PCA_SEED" \
            --split "$PCA_SPLIT" \
            --exclude_head \
            "${bbox_args[@]}"
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
    LWF_MODE="${LWF_MODE:-yolo_lwf}"
    LWF_CLS_LOSS_WEIGHT="${LWF_CLS_LOSS_WEIGHT:-4000.0}"
    LWF_REG_LOSS_WEIGHT="${LWF_REG_LOSS_WEIGHT:-500.0}"
    LWF_FEATURE_LOSS_WEIGHT="${LWF_FEATURE_LOSS_WEIGHT:-1.0}"
    LWF_TEMPERATURE="${LWF_TEMPERATURE:-2.0}"
    LWF_WCE="${LWF_WCE:-True}"
    LWF_CLS_IOU="${LWF_CLS_IOU:-True}"
    LWF_MASK_OLD="${LWF_MASK_OLD:-True}"
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
    REPLAY_MODE="${REPLAY_MODE:-supervised}"
    OCDM_STATS="${OCDM_STATS:-model}"
    OCDM_CONF="${OCDM_CONF:-0.5}"
    OCDM_BATCH_SIZE="${OCDM_BATCH_SIZE:-1000}"
    REPLAY_LOSS_WEIGHT="${REPLAY_LOSS_WEIGHT:-1.0}"
    REPLAY_SEED="${REPLAY_SEED:-0}"
    ESPREG_WEIGHT_MODE="${ESPREG_WEIGHT_MODE:-nt}"
    ESPREG_FORM="${ESPREG_FORM:-original}"
    ESPREG_SCOPE="${ESPREG_SCOPE:-all}"
    PRAD_FG_THRESHOLD="${PRAD_FG_THRESHOLD:-0.5}"
    PRAD_BG_THRESHOLD="${PRAD_BG_THRESHOLD:-0.1}"
    PRAD_CONF_POWER="${PRAD_CONF_POWER:-1.0}"
    PRAD_PROTO_THRESHOLD="${PRAD_PROTO_THRESHOLD:-0.5}"
    PRAD_PROTO_TEMPERATURE="${PRAD_PROTO_TEMPERATURE:-0.1}"
    PRAD_BG_WEIGHT="${PRAD_BG_WEIGHT:-1.0}"
    PRAD_SPLIT_NORMALIZE="${PRAD_SPLIT_NORMALIZE:-True}"
    PRAD_MAX_FG="${PRAD_MAX_FG:-512}"
    PRAD_CLS_LOSS_WEIGHT="${PRAD_CLS_LOSS_WEIGHT:-30.0}"
    PRAD_REG_LOSS_WEIGHT="${PRAD_REG_LOSS_WEIGHT:-1.5}"
    PRD_SAMPLES="${PRD_SAMPLES:-64}"
    PRD_STD_SCALE="${PRD_STD_SCALE:-1.0}"
    PRD_LOSS_WEIGHT="${PRD_LOSS_WEIGHT:-30.0}"
    PRD_REG_LOSS_WEIGHT="${PRD_REG_LOSS_WEIGHT:-5.0}"
    PATCH_PCA_DIMS="${PATCH_PCA_DIMS:-128}"
    PATCH_PROTO_MAX_CENTERS="${PATCH_PROTO_MAX_CENTERS:-8}"
    PATCH_PROTO_SPLIT_RATIO="${PATCH_PROTO_SPLIT_RATIO:-0.2}"
    PATCH_PROTO_MAX_SAMPLES="${PATCH_PROTO_MAX_SAMPLES:-256}"
    PATCH_PROTO_SEED="${PATCH_PROTO_SEED:-0}"
    PCA_SEED="${PCA_SEED:-42}"
    PCA_SPLIT="${PCA_SPLIT:-val}"
    PCA_MODE="${PCA_MODE:-incremental}"
    PCA_SAMPLE_NUM="${PCA_SAMPLE_NUM:-500}"
    if ! declare -p EXTRA_TRAIN_ARGS >/dev/null 2>&1; then
        EXTRA_TRAIN_ARGS=()
    fi

    PREVIOUS_MODEL=""
    PREVIOUS_PCA=""
    PREVIOUS_PROTOTYPES=""
    PREVIOUS_IMPORTANCE=""
    PREVIOUS_REPLAY=""
    PREVIOUS_PATCH_PROTOTYPES=""
    if (( START_TASK > 1 )); then
        previous_task_dir="${OUTPUT_DIR}/task-$((START_TASK - 1))"
        PREVIOUS_MODEL="${previous_task_dir}/best.pt"
        PREVIOUS_PCA="${previous_task_dir}/pca_cache.pkl"
        PREVIOUS_PROTOTYPES="${previous_task_dir}/repre_prototypes.pt"
        PREVIOUS_IMPORTANCE="${previous_task_dir}/importance.pth"
        PREVIOUS_REPLAY="${previous_task_dir}/replay_dataset"
        PREVIOUS_PATCH_PROTOTYPES="${previous_task_dir}/patch_prototypes.pt"
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
        # The fixed shared PCA inside the patch artifact is fitted at the base stage: the adapter can
        # rebuild it only when the previous task is that stage (START_TASK=2), never from a later one.
        if method_needs_patch_prototypes && [[ ! -f "$PREVIOUS_PATCH_PROTOTYPES" ]] && (( START_TASK > 2 )); then
            echo "Previous task patch prototype artifact not found: $PREVIOUS_PATCH_PROTOTYPES" >&2
            echo "The fixed shared PCA is fitted at the base stage; rerun from START_TASK=2" >&2
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
        # The fixed shared PCA inside the patch artifact is fitted at the base stage: it can be rebuilt
        # only when the previous task is that stage (START_TASK=2), never from a later stage.
        if method_needs_patch_prototypes && [[ ! -f "$PREVIOUS_PATCH_PROTOTYPES" ]] && (( START_TASK > 2 )); then
            echo "Previous task patch prototype artifact not found: $PREVIOUS_PATCH_PROTOTYPES" >&2
            echo "The fixed shared PCA is fitted at the base stage; rerun from START_TASK=2" >&2
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

    # Trainer intermediates land in $TASK_DIR (absolute --project). Drop leftovers from a
    # previous attempt at this task so a rerun overwrites instead of accumulating
    # train2/train3/... . Skipped when resuming: the in-training checkpoint may live there.
    if [[ -z "$RESUME_CHECKPOINT" ]]; then
        rm -rf "${TASK_DIR}/train" "${TASK_DIR}/interim/train"
    fi

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

        if method_has ewc; then
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

        if method_has ewc; then
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

        if method_has lwf; then
            TRAINER_ARGS+=(
                --lwf True
                --lwf_mode "$LWF_MODE"
                --lwf_teacher_model "$PREVIOUS_MODEL"
                --lwf_cls_loss_weight "$LWF_CLS_LOSS_WEIGHT"
                --lwf_reg_loss_weight "$LWF_REG_LOSS_WEIGHT"
                --lwf_feature_loss_weight "$LWF_FEATURE_LOSS_WEIGHT"
                --lwf_temperature "$LWF_TEMPERATURE"
                --lwf_wce "$LWF_WCE"
                --lwf_cls_iou "$LWF_CLS_IOU"
                --lwf_mask_old "$LWF_MASK_OLD"
            )
        fi

        if method_has espreg; then
            TRAINER_ARGS+=(
                --espreg True
                --pca_cache_path "$PREVIOUS_PCA"
                --espreg_loss_weight "$ESPREG_LOSS_WEIGHT"
                --espreg_weight_mode "$ESPREG_WEIGHT_MODE"
                --espreg_form "$ESPREG_FORM"
                --espreg_scope "$ESPREG_SCOPE"
            )
        fi

        if method_has prad || method_has prd; then
            TRAINER_ARGS+=(
                --lwf_teacher_model "$PREVIOUS_MODEL"
                --patch_prototypes "$PREVIOUS_PATCH_PROTOTYPES"
            )
        fi

        if method_has prad; then
            TRAINER_ARGS+=(
                --lwf True
                --lwf_mode prad
                --lwf_mask_old "$LWF_MASK_OLD"
                --prad_fg_threshold "$PRAD_FG_THRESHOLD"
                --prad_bg_threshold "$PRAD_BG_THRESHOLD"
                --prad_conf_power "$PRAD_CONF_POWER"
                --prad_proto_threshold "$PRAD_PROTO_THRESHOLD"
                --prad_proto_temperature "$PRAD_PROTO_TEMPERATURE"
                --prad_bg_weight "$PRAD_BG_WEIGHT"
                --prad_split_normalize "$PRAD_SPLIT_NORMALIZE"
                --prad_max_fg "$PRAD_MAX_FG"
                --prad_cls_loss_weight "$PRAD_CLS_LOSS_WEIGHT"
                --prad_reg_loss_weight "$PRAD_REG_LOSS_WEIGHT"
            )
        fi

        if method_has prd; then
            TRAINER_ARGS+=(
                --prd True
                --prd_samples "$PRD_SAMPLES"
                --prd_std_scale "$PRD_STD_SCALE"
                --prd_loss_weight "$PRD_LOSS_WEIGHT"
                --prd_reg_loss_weight "$PRD_REG_LOSS_WEIGHT"
            )
        fi

        if method_has nsgp; then
            TRAINER_ARGS+=(
                --nsgp True
                --nsgp_flexibility 1.0
                --pca_cache_path "$PREVIOUS_PCA"
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
                --replay_mode "$REPLAY_MODE"
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
        local -a bbox_args=()
        if [[ -n "$PREVIOUS_PCA" ]]; then
            history_args=(--load_hist "$PREVIOUS_PCA")
        fi
        if method_has espreg; then
            bbox_args=(--use_bbox)
        fi
        python tools/pca.py \
            --model "$PREVIOUS_MODEL" \
            --dataset "$DATASET_PATH" \
            --save_path "${TASK_DIR}/pca_cache.pkl" \
            --device "$TOOL_DEVICE" \
            --pca_mode "$PCA_MODE" \
            --sample_num "$PCA_SAMPLE_NUM" \
            --batch_size "$BATCH_SIZE" \
            --imgsz "$IMGSZ" \
            --workers "$WORKERS" \
            --seed "$PCA_SEED" \
            --split "$PCA_SPLIT" \
            --exclude_head \
            "${bbox_args[@]}" \
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

    if method_needs_patch_prototypes; then
        history_args=()
        if [[ -n "$PREVIOUS_PATCH_PROTOTYPES" ]]; then
            history_args=(--load_hist "$PREVIOUS_PATCH_PROTOTYPES")
        fi
        python tools/generate_patch_prototypes.py \
            --model "$PREVIOUS_MODEL" \
            --data "$TRAIN_DATA" \
            --output "${TASK_DIR}/patch_prototypes.pt" \
            --device "$TOOL_DEVICE" \
            --imgsz "$IMGSZ" \
            --batch_size "$BATCH_SIZE" \
            --workers "$WORKERS" \
            --pca_dims "$PATCH_PCA_DIMS" \
            --max_centers "$PATCH_PROTO_MAX_CENTERS" \
            --split_ratio "$PATCH_PROTO_SPLIT_RATIO" \
            --max_samples_per_class "$PATCH_PROTO_MAX_SAMPLES" \
            --seed "$PATCH_PROTO_SEED" \
            "${history_args[@]}"
        PREVIOUS_PATCH_PROTOTYPES="${TASK_DIR}/patch_prototypes.pt"
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
        # Pool strategies (ocdm) keep the memory at --num samples and annotate the
        # pool with the model of the task that just finished.
        ocdm_args=()
        if [[ "$REPLAY_STRATEGY" == "ocdm" ]]; then
            ocdm_args=(
                --ocdm_stats "$OCDM_STATS"
                --ocdm_model "$PREVIOUS_MODEL"
                --ocdm_conf "$OCDM_CONF"
                --ocdm_imgsz "$IMGSZ"
                --ocdm_device "$TOOL_DEVICE"
                --ocdm_batch_size "$OCDM_BATCH_SIZE"
            )
        fi
        python tools/select_replay_samples.py \
            --dataset "$TRAIN_DATA" \
            --output_dir "${TASK_DIR}/replay_dataset" \
            --num "$REPLAY_SAMPLE_NUM" \
            --strategy "$REPLAY_STRATEGY" \
            --seed "$REPLAY_SEED" \
            "${history_args[@]}" \
            "${ocdm_args[@]}"
        PREVIOUS_REPLAY="${TASK_DIR}/replay_dataset"
    fi
}
