#!/bin/bash
# Prune-vs-random comparison: for each keep ratio, train the coreset subset and
# a random subset with the same hyperparameters and epoch budget, then score
# both on the original full eval split (mAP50). Tunables live in common.sh.

set -euo pipefail

DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$DEMO_DIR"
while [[ ! -f "$REPO_ROOT/scripts/train.sh" && "$REPO_ROOT" != "/" ]]; do
    REPO_ROOT="$(dirname "$REPO_ROOT")"
done
if [[ ! -f "$REPO_ROOT/scripts/train.sh" ]]; then
    echo "Cannot locate the repository root above $DEMO_DIR" >&2
    exit 1
fi
cd "$REPO_ROOT"
source "$DEMO_DIR/common.sh"

echo "Comparing ${DATA_TAG}: ratios ${RATIOS[*]}  splits ${SPLITS[*]}"

if [[ ! -f "$SRC_DATA" ]]; then
    echo "Source yaml not found: $SRC_DATA" >&2
    echo "Fix SRC_DATA in common.sh" >&2
    exit 1
fi
if [[ ! -f "$EVAL_YAML" ]]; then
    echo "Eval yaml not found: $EVAL_YAML" >&2
    echo "Fix EVAL_YAML in common.sh" >&2
    exit 1
fi

for cmd in "${PREP_CMDS[@]+"${PREP_CMDS[@]}"}"; do
    eval "$cmd"
done

mkdir -p "$SUBSET_ROOT" "$RUN_ROOT"
echo "ratio,method,mAP50,run_dir,subset_yaml" > "$SUMMARY_CSV"

ratio_tag() {
    python -c "print('{:.2f}'.format(float('$1')).replace('.', 'p'))"
}

run_one() {
    local method="$1" ratio="$2" subset_dir="$3" run_dir="$4" random_splits_csv="$5"
    local yaml="$subset_dir/dataset.yaml"
    local eval_csv="$run_dir/evaluation_results/model_1_eval_task_1.csv"
    if [[ ! -f "$yaml" ]]; then
        local -a prune_args=(
            --src_data "$SRC_DATA"
            --out_dir "$subset_dir"
            --ratio "$ratio"
            --splits "${SPLITS[@]}"
            --seed "$SEED"
            --backbone "$BACKBONE"
            --feature_cache_dir "$FEATURE_CACHE_DIR"
        )
        if [[ -n "$random_splits_csv" ]]; then
            local -a extra_random
            IFS=',' read -ra extra_random <<< "$random_splits_csv"
            prune_args+=(--random_splits "${extra_random[@]}")
        elif (( ${#RANDOM_SPLITS[@]} > 0 )); then
            prune_args+=(--random_splits "${RANDOM_SPLITS[@]}")
        else
            prune_args+=(--random_splits)
        fi
        bash scripts/prune.sh "${prune_args[@]}"
    fi
    if (( ${#ATTACH_SOURCE_SPLITS[@]} > 0 )); then
        python tools/attach_source_splits.py --src_data "$SRC_DATA" --dst_yaml "$yaml" --splits "${ATTACH_SOURCE_SPLITS[@]}"
    fi
    if [[ ! -f "$eval_csv" ]]; then
    EPOCHS="$EPOCHS" DEVICE="$DEVICE" BATCH_SIZE="$BATCH_SIZE" CLASS_AGNOSTIC_EVAL=0 \
        bash scripts/train.sh \
            --tasks "$yaml" \
            --model "$MODEL" \
            --method "$METHOD" \
            --weights "$WEIGHTS" \
            --tag "${DATA_TAG}_${method}_r$(ratio_tag "$ratio")" \
            --output "$run_dir" \
            -- \
            "${EXTRA_TRAIN_ARGS[@]}"
    CLASS_AGNOSTIC_EVAL=0 \
        bash scripts/eval.sh --run "$run_dir" --tasks "$EVAL_YAML" --split "$EVAL_SPLIT" -- "${EXTRA_EVAL_ARGS[@]}"
    fi
    local map50
    map50="$(python - "$run_dir/evaluation_results/model_1_eval_task_1.csv" <<'PY'
import csv, sys
from pathlib import Path
path = Path(sys.argv[1])
with path.open(encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))
values = []
for row in rows:
    if row.get("Class", "").lower() == "all":
        print(row["mAP50"])
        raise SystemExit
    if row.get("mAP50", "") != "":
        values.append(float(row["mAP50"]))
if not values:
    raise SystemExit(f"No mAP50 in {path}")
print(sum(values) / len(values))
PY
)"
    echo "${ratio},${method},${map50},${run_dir},${yaml}" >> "$SUMMARY_CSV"
}

for ratio in "${RATIOS[@]}"; do
    tag="$(ratio_tag "$ratio")"
    echo "========== ratio ${ratio} coreset =========="
    run_one coreset "$ratio" "${SUBSET_ROOT}/coreset_r${tag}" "${RUN_ROOT}/coreset_r${tag}" ""
    echo "========== ratio ${ratio} random =========="
    all_random="$(IFS=','; echo "${SPLITS[*]}")"
    run_one random "$ratio" "${SUBSET_ROOT}/random_r${tag}" "${RUN_ROOT}/random_r${tag}" "$all_random"
done

python tools/summarize_prune_compare.py --csv "$SUMMARY_CSV" --min_interval "$MIN_INTERVAL"
