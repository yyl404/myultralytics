#!/bin/bash
# One-command offline experiment: one incremental method on VOC 10+10.
#
# Reuses the naive baseline's stage-1 checkpoint (no retraining of task 1), trains only
# task 2 on one GPU, evaluates the resulting two-stage run (per-task and cumulative,
# class-aware) and writes one plain-text summary plus one
# machine-readable result row. Everything is repo-relative and runs offline.
#
# The method is set in the configuration block below; the sibling scripts in this
# directory are byte-identical except for that block.
#
# Usage:
#   bash scripts/voc/all/10+10/<script>.sh                 # full 100-epoch run + eval + summary
#   EPOCHS=1 bash scripts/voc/all/10+10/<script>.sh --skip-eval     # short pipeline check
#   DEVICE=1 WORKERS=4 bash scripts/voc/all/10+10/<script>.sh       # another GPU
#   BASE_MODEL=/path/to/task-1/best.pt bash scripts/voc/all/10+10/<script>.sh
#   bash scripts/voc/all/10+10/<script>.sh --prepare-only  # resolve split/base/plan, then exit
#
# Outputs (copy these back for analysis):
#   results/<tag>.txt       full run summary (protocol, matrices, stage mAP, inventory)
#   results/<tag>.row.csv   one comparison row (Old/New/All, class-aware)
#   runs/<run>/task-2/best.pt  final stage-2 checkpoint
set -euo pipefail

# ============================================================================
# Method configuration — the only block that differs between the sibling scripts.
# ============================================================================
RUN_TAG="${RUN_TAG:-lwf+ewc+nsgp}"
METHOD="${METHOD:-lwf+ewc+nsgp}"
ESPREG_WEIGHT_MODE="${ESPREG_WEIGHT_MODE:-raw}"
ESPREG_FORM="${ESPREG_FORM:-frobenius}"
ESPREG_SCOPE="${ESPREG_SCOPE:-all}"
ESPREG_LOSS_WEIGHT="${ESPREG_LOSS_WEIGHT:-0.01}"
NSGP_FLEXIBILITY="${NSGP_FLEXIBILITY:-1.0}"
PCA_MODE="${PCA_MODE:-gram}"
# ============================================================================

# Protocol / run identity.
DATASET="voc"
SPLIT="10_10"
MODEL_SPEC="yolo26m"
WEIGHTS="yolo26m-cls.pt"
BASE_MODEL="${BASE_MODEL:-runs/yolo26m_VOC_10+10_pretrained-from-yolo26m-cls_naive/task-1/best.pt}"
RUN_DIR="${RUN_DIR:-runs/yolo26m_VOC_10+10_pretrained-from-yolo26m-cls_${RUN_TAG}}"
SUMMARY="${SUMMARY:-results/${RUN_TAG}.txt}"
ROW="${ROW:-results/${RUN_TAG}.row.csv}"
LOG="${LOG:-logs/${RUN_TAG}.log}"

# Training budget and loader (override for a smoke or a different machine).
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-16}"
IMGSZ="${IMGSZ:-640}"
WORKERS="${WORKERS:-8}"
DEVICE="${DEVICE:-0}"

die() { echo "ERROR: $*" >&2; exit 1; }

# Resolve the repository root: the directory that holds scripts/offline_experiment.sh.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "$ROOT/scripts/offline_experiment.sh" ]]; do
    if [[ "$ROOT" == "/" ]]; then
        echo "ERROR: cannot locate the repository root (scripts/offline_experiment.sh not found)" >&2
        exit 1
    fi
    ROOT="$(dirname "$ROOT")"
done
cd "$ROOT"

# The tools under tools/ import the `ultralytics` package; put the repository root on
# the import path so the run works without an editable install of the package.
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

# ---- environment pre-flight -------------------------------------------------
command -v python >/dev/null 2>&1 || die "python not found; activate the project environment first"
python -c "import torch, ultralytics" >/dev/null 2>&1 \
    || die "python cannot import torch/ultralytics; activate the project environment first"
if [[ "$DEVICE" != "cpu" ]]; then
    python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
        || die "CUDA is not available for DEVICE=$DEVICE (use DEVICE=cpu only for a dry run)"
fi

TRAIN_BASE=0
SKIP_EVAL=0
PREPARE_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --train-base) TRAIN_BASE=1 ;;
        --skip-eval) SKIP_EVAL=1 ;;
        --prepare-only) PREPARE_ONLY=1 ;;
    esac
done

if [[ ! -f "data/VOC_10+10/task_1_cls_10/dataset.yaml" && ! -f "data/VOC-YOLO/VOC.yaml" ]]; then
    die "neither data/VOC_10+10 nor the VOC source yaml (data/VOC-YOLO/VOC.yaml) is present"
fi
if (( TRAIN_BASE == 0 )) && [[ ! -f "$BASE_MODEL" ]]; then
    die "base-stage checkpoint not found: $BASE_MODEL
     Copy the naive baseline's task-1/best.pt there, pass BASE_MODEL=/path/to/best.pt,
     or add --train-base to train stage 1 on this machine."
fi

# ---- method knobs read by the adapter / offline driver ----------------------
export METHOD ESPREG_WEIGHT_MODE ESPREG_FORM ESPREG_SCOPE ESPREG_LOSS_WEIGHT PCA_MODE
export EWC_LOSS_WEIGHT="${EWC_LOSS_WEIGHT:-100}"
export LWF_MODE="${LWF_MODE:-yolo_lwf}"
export LWF_CLS_LOSS_WEIGHT="${LWF_CLS_LOSS_WEIGHT:-4000.0}"
export LWF_REG_LOSS_WEIGHT="${LWF_REG_LOSS_WEIGHT:-500.0}"
export LWF_FEATURE_LOSS_WEIGHT="${LWF_FEATURE_LOSS_WEIGHT:-1.0}"
export LWF_TEMPERATURE="${LWF_TEMPERATURE:-2.0}"
export LWF_WCE="${LWF_WCE:-True}"
export LWF_CLS_IOU="${LWF_CLS_IOU:-True}"
export LWF_MASK_OLD="${LWF_MASK_OLD:-True}"
export PCA_SAMPLE_NUM="${PCA_SAMPLE_NUM:-500}"
export PCA_SEED="${PCA_SEED:-42}"
export PCA_SPLIT="${PCA_SPLIT:-val}"
export TOOL_DEVICE="${TOOL_DEVICE:-${DEVICE%%,*}}"
export END2END="False"
export STAGE_EVAL_PERIOD="${STAGE_EVAL_PERIOD:-0}"

# One optimizer / schedule / decode protocol for every method, so the only difference
# between the three runs is the regularizer. Passed after `--` so it overrides the
# adapter's NSGP-native SGD defaults.
EXTRA_ARGS="--optimizer AdamW --lr0 0.001 --lrf 0.01 --momentum 0.937 --weight_decay 0.0005 \
--warmup_bias_lr 0.0 --mosaic 0.5 --close_mosaic 10 \
--end2end False --agnostic_nms True --max_det 300 --nsgp_flexibility $NSGP_FLEXIBILITY"

mkdir -p results logs
mkdir -p "$(dirname "$RUN_DIR")"
: > "$LOG"
rm -f "results/${RUN_TAG}.done"

echo "=============================================================================="
echo "Offline incremental experiment"
echo "  run     : $RUN_TAG"
echo "  method  : $METHOD  (pca=$PCA_MODE)"
if [[ "+${METHOD}+" == *"+espreg+"* ]]; then
    echo "  espreg  : weight_mode=$ESPREG_WEIGHT_MODE form=$ESPREG_FORM scope=$ESPREG_SCOPE lambda=$ESPREG_LOSS_WEIGHT"
fi
if [[ "+${METHOD}+" == *"+nsgp+"* ]]; then
    echo "  nsgp    : flexibility=$NSGP_FLEXIBILITY"
fi
echo "  protocol: $DATASET $SPLIT, $MODEL_SPEC, ${EPOCHS} epochs, batch=$BATCH_SIZE, device=$DEVICE"
echo "  base    : $BASE_MODEL"
echo "  output  : $RUN_DIR"
echo "  summary : $SUMMARY"
echo "=============================================================================="

bash scripts/offline_experiment.sh \
    --dataset "$DATASET" \
    --split "$SPLIT" \
    --model "$MODEL_SPEC" \
    --weights "$WEIGHTS" \
    --base-model "$BASE_MODEL" \
    --method "$METHOD" \
    --output "$RUN_DIR" \
    --summary "$SUMMARY" \
    --epochs "$EPOCHS" \
    --batch "$BATCH_SIZE" \
    --workers "$WORKERS" \
    --device "$DEVICE" \
    --skip-report \
    --extra-args "$EXTRA_ARGS" \
    "$@" 2>&1 | tee "$LOG"

if (( PREPARE_ONLY == 1 )); then
    echo "Prepare-only finished; no training or evaluation was launched."
    exit 0
fi
if (( SKIP_EVAL == 1 )); then
    echo "Training finished with evaluation skipped; no result row was written."
    exit 0
fi

# ---- extract the final comparison row ---------------------------------------
# Every matrix CSV has one row per evaluated checkpoint; Model_2 is the final stage.
read_metrics() {
    awk -F, '$1=="Model_2"{printf "%s %s %s %s\n", $2, $3, $4, $5; found=1; exit}
             END{if (!found) printf "N/A N/A N/A N/A\n"}' "$1"
}

pct() {
    if [[ "$1" == "N/A" || -z "$1" ]]; then
        printf "N/A"
    else
        awk -v x="$1" -v d="${2:-1}" 'BEGIN{printf "%.*f", d, 100*x}'
    fi
}

mean_pct() {
    if [[ "$1" == "N/A" || "$2" == "N/A" || -z "$1" || -z "$2" ]]; then
        printf "N/A"
    else
        awk -v a="$1" -v b="$2" -v d="${3:-1}" 'BEGIN{printf "%.*f", d, 50*(a+b)}'
    fi
}

read -r t1_50 t1_95 t2_50 t2_95 < <(read_metrics "$RUN_DIR/evaluation_results/individual_datasets_eval.csv")
read -r c1_50 c1_95 c2_50 c2_95 < <(read_metrics "$RUN_DIR/evaluation_results/cumulative_datasets_eval.csv")

row_espreg_weight_mode=""
row_espreg_form=""
row_espreg_loss_weight=""
row_nsgp_flexibility=""
if [[ "+${METHOD}+" == *"+espreg+"* ]]; then
    row_espreg_weight_mode="$ESPREG_WEIGHT_MODE"
    row_espreg_form="$ESPREG_FORM"
    row_espreg_loss_weight="$ESPREG_LOSS_WEIGHT"
fi
if [[ "+${METHOD}+" == *"+nsgp+"* ]]; then
    row_nsgp_flexibility="$NSGP_FLEXIBILITY"
fi

{
    echo "tag,method,espreg_weight_mode,espreg_form,espreg_loss_weight,nsgp_flexibility,pca_mode,epochs,map50_t1,map50_t2,map50_all,map5095_t1,map5095_t2,map5095_all,cumulative_map5095_1,cumulative_map5095_2,run_dir"
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$RUN_TAG" "$METHOD" "$row_espreg_weight_mode" "$row_espreg_form" "$row_espreg_loss_weight" \
        "$row_nsgp_flexibility" "$PCA_MODE" "$EPOCHS" \
        "$(pct "$t1_50" 2)" "$(pct "$t2_50" 2)" "$(mean_pct "$t1_50" "$t2_50" 2)" \
        "$(pct "$t1_95" 2)" "$(pct "$t2_95" 2)" "$(mean_pct "$t1_95" "$t2_95" 2)" \
        "$(pct "$c1_95" 2)" "$(pct "$c2_95" 2)" "$RUN_DIR"
} > "$ROW"

echo "=============================================================================="
echo "FINAL RESULT: $RUN_TAG  ($METHOD)"
printf '  %-22s %8s %8s %8s\n' "" "T1 (Old)" "T2 (New)" "All"
printf '  %-22s %8s %8s %8s\n' "mAP50 (class-aware)" "$(pct "$t1_50")" "$(pct "$t2_50")" "$(mean_pct "$t1_50" "$t2_50")"
printf '  %-22s %8s %8s %8s\n' "mAP50-95 (class-aware)" "$(pct "$t1_95")" "$(pct "$t2_95")" "$(mean_pct "$t1_95" "$t2_95")"
printf '  %-22s %8s %8s %8s\n' "cumulative mAP50-95" "$(pct "$c1_95")" "$(pct "$c2_95")" "-"
echo "  summary : $SUMMARY"
echo "  row     : $ROW"
echo "  ckpt    : $RUN_DIR/task-2/best.pt"
echo "=============================================================================="

touch "results/${RUN_TAG}.done"
