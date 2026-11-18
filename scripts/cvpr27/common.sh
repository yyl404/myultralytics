#!/bin/bash
# Shared driver for the CVPR-27 revision experiments.
#
# Protocol (paper's Implementation Details):
#   detector      YOLOv8m, backbone initialized from yolov8m-cls.pt
#   training      100 epochs per stage, batch 16, SGD lr0=0.01,
#                 momentum=0.937, weight decay=5e-4
#   stage-1       one naive checkpoint per split, shared by every YOLO row
#   rows          ESP-YOLO = pseudo_label+espreg+dist (CAD top-1)
#                 19+1 baselines = pseudo_label+ewc / +nsgp / +nsgp+repre / +bpf
#   CAD ablation  15+5, pseudo_label+espreg+dist, dist_topk in {1,3,5,-1}
#   metric        final model on the 20-class union test set:
#                 Old = Task_1, New = Task_2, All = all 20 classes
#
# The machine scripts only set MACHINE_ID, TRAIN_STAGE1, STAGE1_SPLITS and RUNS.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

: "${MACHINE_ID:?the machine script must set MACHINE_ID}"
TRAIN_STAGE1="${TRAIN_STAGE1:-0}"
DEVICE="${DEVICE:-0}"
WORKERS="${WORKERS:-8}"
BATCH_SIZE="${BATCH_SIZE:-16}"
IMGSZ="${IMGSZ:-640}"
EPOCHS="${EPOCHS:-100}"
LR0="${LR0:-0.01}"
MOMENTUM="${MOMENTUM:-0.937}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0005}"
SEED="${SEED:-0}"
YOLOV8M_CLS="${YOLOV8M_CLS:-$REPO_ROOT/yolov8m-cls.pt}"
FORCE="${FORCE:-0}"
PLAN_ONLY="${PLAN_ONLY:-0}"

STAGE1_ROOT="$REPO_ROOT/runs/_cvpr27_stage1"
RESULTS_DIR="$REPO_ROOT/results/cvpr27"
RESULT_CSV="$RESULTS_DIR/machine${MACHINE_ID}.csv"
LOG_DIR="$REPO_ROOT/runs/_cvpr27_logs"
mkdir -p "$STAGE1_ROOT" "$RESULTS_DIR" "$LOG_DIR"

log() {
    printf '[%s][machine %s] %s\n' "$(date '+%H:%M:%S')" "$MACHINE_ID" "$*"
}

die() {
    printf '[machine %s] ERROR: %s\n' "$MACHINE_ID" "$*" >&2
    exit 1
}

method_has() {
    [[ "+${METHOD}+" == *"+$1+"* ]]
}

require_weight() {
    [[ -f "$YOLOV8M_CLS" ]] || die "yolov8m-cls.pt not found at '$YOLOV8M_CLS'. Put the same file at this path on all three machines, or set YOLOV8M_CLS=/path/to/yolov8m-cls.pt."
}

split_config() {
    case "$1" in
        "15+5")
            DATA_TAG="VOC_15+5"
            TASK1="$REPO_ROOT/data/VOC_15+5/task_1_cls_15/dataset.yaml"
            TASK2="$REPO_ROOT/data/VOC_15+5/task_2_cls_5/dataset.yaml"
            CUM="$REPO_ROOT/data/VOC_15+5/task_1-2_cls_20/dataset.yaml"
            ;;
        "19+1")
            DATA_TAG="VOC_19+1"
            TASK1="$REPO_ROOT/data/VOC_19+1/task_1_cls_19/dataset.yaml"
            TASK2="$REPO_ROOT/data/VOC_19+1/task_2_cls_1/dataset.yaml"
            CUM="$REPO_ROOT/data/VOC_19+1/task_1-2_cls_20/dataset.yaml"
            ;;
        *)
            die "unknown split '$1' (expected 15+5 or 19+1)"
            ;;
    esac
}

ensure_data() {
    local split="$1"
    split_config "$split"
    local missing=0
    local path
    for path in "$TASK1" "$TASK2" "$CUM"; do
        [[ -f "$path" ]] || missing=1
    done
    if (( missing == 0 )); then
        return
    fi
    if [[ "$split" == "19+1" ]]; then
        log "VOC 19+1 split missing; creating it with scripts/create.sh (one-time, ~minutes)"
        bash scripts/create.sh voc 19_1
        split_config "$split"
        for path in "$TASK1" "$TASK2" "$CUM"; do
            [[ -f "$path" ]] || die "VOC 19+1 was not created at '$path'"
        done
    else
        die "VOC 15+5 yaml missing ($TASK1 / $TASK2 / $CUM); create the 15_5 split first"
    fi
}

stage1_best() {
    printf '%s/%s/task-1-best.pt' "$STAGE1_ROOT" "$DATA_TAG"
}

base_run() {
    printf '%s/runs/yolov8m_%s_pretrained-from-yolov8m-cls_naive' "$REPO_ROOT" "$DATA_TAG"
}

print_copy_instructions() {
    local best="$1"
    cat <<COPY_EOF
============================================================
[machine 1] STAGE-1 READY: ${DATA_TAG}
  local file : ${best}
  copy it to the other machines at the SAME path:
    scp "${best}" <user>@<offline-host>:${best}
  The offline scripts require this file; if missing, they exit immediately.
============================================================
COPY_EOF
}

train_stage1() {
    split_config "$1"
    local best base
    best="$(stage1_best)"
    base="$(base_run)"
    if [[ -f "$best" && "$FORCE" != "1" ]]; then
        log "stage-1 already shared: $best"
        return
    fi
    if [[ -f "$base/task-1/best.pt" && "$FORCE" != "1" ]]; then
        log "stage-1 checkpoint exists, publishing it: $base/task-1/best.pt"
        mkdir -p "$(dirname "$best")"
        cp -f "$base/task-1/best.pt" "$best"
        print_copy_instructions "$best"
        return
    fi

    log "training stage-1 ${DATA_TAG}: yolov8m + yolov8m-cls, ${EPOCHS} epochs, SGD lr0=${LR0}"
    mkdir -p "$base"
    EPOCHS="$EPOCHS" BATCH_SIZE="$BATCH_SIZE" WORKERS="$WORKERS" DEVICE="$DEVICE" \
    STAGE_EVAL_PERIOD=0 \
    bash scripts/train.sh \
        --tasks "$TASK1" \
        --model yolov8m \
        --weights "$YOLOV8M_CLS" \
        --method naive \
        --output "$base" \
        -- --optimizer SGD --lr0 "$LR0" --momentum "$MOMENTUM" \
           --weight_decay "$WEIGHT_DECAY" --seed "$SEED" \
        2>&1 | tee -a "$LOG_DIR/stage1_${DATA_TAG}.log"

    [[ -f "$base/task-1/best.pt" ]] || die "stage-1 training finished without $base/task-1/best.pt"
    mkdir -p "$(dirname "$best")"
    cp -f "$base/task-1/best.pt" "$best"
    print_copy_instructions "$best"
}

wait_for_stage1() {
    split_config "$1"
    local best
    best="$(stage1_best)"
    if [[ -f "$best" ]]; then
        log "stage-1 available: $best"
        return
    fi
    if [[ "$TRAIN_STAGE1" == "1" ]]; then
        train_stage1 "$1"
        return
    fi

    die "stage-1 checkpoint missing: $best. Copy it from machine 1 to this exact path, then rerun this script."
}

pick_pca_mode() {
    local method="$1"
    if [[ "+${method}+" == *"+espreg+"* ]]; then
        printf 'incremental'
    elif [[ "+${method}+" == *"+nsgp+"* ]]; then
        printf 'gram'
    else
        printf ''
    fi
}

append_result() {
    local split="$1" method="$2" tag="$3" topk="$4" seed="$5" run_dir="$6"
    local stage_csv="$run_dir/evaluation_results/stage_mAP_sequence.csv"
    [[ -f "$stage_csv" ]] || die "missing stage result table: $stage_csv"

    if [[ ! -f "$RESULT_CSV" ]]; then
        printf 'machine,split,method,tag,dist_topk,seed,old50,new50,all50,avg50,old95,new95,all95,run_dir\n' > "$RESULT_CSV"
    fi
    # Keep one row per run directory when a run is resumed.
    awk -F, -v run="$run_dir" 'NF == 0 || $NF != run' "$RESULT_CSV" > "$RESULT_CSV.tmp" || true
    mv "$RESULT_CSV.tmp" "$RESULT_CSV"

    awk -F, -v machine="$MACHINE_ID" -v sp="$split" -v method="$method" -v tag="$tag" \
        -v topk="$topk" -v seed="$seed" -v run="$run_dir" '
        $1 == "Model_2" && $2 == "cumulative_2" {
            if ($3 == "Task_1") { old50=$5; old95=$6 }
            else if ($3 == "Task_2") { new50=$5; new95=$6 }
            else if ($3 == "All") { all50=$5; all95=$6 }
        }
        END {
            if (old50 == "" || new50 == "" || all50 == "") exit 2
            printf "%s,%s,%s,%s,%s,%s,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%s\n", \
                machine, sp, method, tag, topk, seed, \
                100*old50, 100*new50, 100*all50, 50*(old50+new50), \
                100*old95, 100*new95, 100*all95, run
        }' "$stage_csv" >> "$RESULT_CSV"

    log "result appended: $RESULT_CSV"
}

run_task() {
    local split="$1" method="$2" tag="$3" topk="$4" seed="$5" dist_mode="${6:-topk}"
    split_config "$split"
    METHOD="$method"
    local best run_dir pca_mode
    best="$(stage1_best)"
    run_dir="$REPO_ROOT/runs/yolov8m_${DATA_TAG}_pretrained-from-yolov8m-cls_${tag}"
    pca_mode="$(pick_pca_mode "$method")"

    [[ -f "$best" ]] || die "stage-1 checkpoint missing: $best"
    mkdir -p "$run_dir/task-1" "$LOG_DIR"
    ln -sfn "$best" "$run_dir/task-1/best.pt"
    log "run ${split}/${tag}: method=${method} dist_topk=${topk:-n/a} seed=${seed} pca_mode=${pca_mode:-n/a}"

    if [[ -f "$run_dir/task-2/best.pt" && "$FORCE" != "1" ]]; then
        log "task-2 checkpoint exists, skipping training: $run_dir/task-2/best.pt"
    else
        env \
            START_TASK=2 \
            EPOCHS="$EPOCHS" BATCH_SIZE="$BATCH_SIZE" WORKERS="$WORKERS" DEVICE="$DEVICE" \
            IMGSZ="$IMGSZ" STAGE_EVAL_PERIOD=0 \
            REFERENCE_MODEL_MODE=previous \
            DIST_TOPK="${topk:-1}" DIST_LOSS_WEIGHT=100 DIST_MODE="$dist_mode" \
            PCA_MODE="$pca_mode" \
            ESPREG_WEIGHT_MODE=raw ESPREG_FORM=frobenius \
            ESPREG_LOSS_WEIGHT=0.01 ESPREG_SCOPE=all \
            EWC_LOSS_WEIGHT=100 NSGP_FLEXIBILITY=1.0 \
            bash scripts/train.sh \
                --tasks "$TASK1" "$TASK2" \
                --model yolov8m \
                --weights "$YOLOV8M_CLS" \
                --method "$method" \
                --output "$run_dir" \
                -- --optimizer SGD --lr0 "$LR0" --momentum "$MOMENTUM" \
                   --weight_decay "$WEIGHT_DECAY" --seed "$seed" \
                2>&1 | tee -a "$LOG_DIR/${split}_${tag}.log"
        [[ -f "$run_dir/task-2/best.pt" ]] || die "training finished without $run_dir/task-2/best.pt"
    fi

    if [[ ! -f "$run_dir/evaluation_results/stage_mAP_sequence.csv" || "$FORCE" == "1" ]]; then
        log "evaluating ${split}/${tag} on the 20-class union test set"
        bash scripts/eval.sh "$run_dir" \
            --tasks "$TASK1" "$TASK2" \
            --cumulative "$TASK1" "$CUM" \
            -- --agnostic_nms False --max_det 300 \
            2>&1 | tee -a "$LOG_DIR/${split}_${tag}_eval.log"
    else
        log "evaluation exists, skipping eval: $run_dir/evaluation_results/stage_mAP_sequence.csv"
    fi

    python tools/summarize_run.py --run "$run_dir" --output "$run_dir/summary.txt" >/dev/null
    append_result "$split" "$method" "$tag" "${topk:-}" "$seed" "$run_dir"
}

main() {
    log "CVPR-27 revision driver: machine ${MACHINE_ID}, TRAIN_STAGE1=${TRAIN_STAGE1}"
    if [[ "$PLAN_ONLY" == "1" ]]; then
        echo "PLAN ONLY: machine ${MACHINE_ID}"
        echo "  stage-1: ${STAGE1_SPLITS[*]}"
        local run
        for run in "${RUNS[@]}"; do
            echo "  run: ${run}"
        done
        echo "  result: ${RESULT_CSV}"
        return
    fi

    require_weight

    local split
    for split in "${STAGE1_SPLITS[@]}"; do
        ensure_data "$split"
    done
    for split in "${STAGE1_SPLITS[@]}"; do
        wait_for_stage1 "$split"
    done

    local entry
    for entry in "${RUNS[@]}"; do
        IFS='|' read -r split method tag topk seed dist_mode <<< "$entry"
        run_task "$split" "$method" "$tag" "$topk" "$seed" "$dist_mode"
    done

    log "done. results: $RESULT_CSV"
    cat "$RESULT_CSV"
}

main "$@"
