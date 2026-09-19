#!/bin/bash
# Prune-vs-random comparison: for each keep ratio, train the coreset subset and
# a random subset with the same hyperparameters and epoch budget, then score
# both on the original full eval split (mAP50). Tunables live in common.sh.
# If DOMAINS is non-empty, run the same protocol once per domain under
# SRC_DATA_ROOT (DATA_TAG = DATA_TAG_PREFIX + domain).

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
PREFERRED_EVAL_SPLIT="$EVAL_SPLIT"

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
            --selector "$SELECTOR"
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

resolve_eval_split() {
    python - "$SRC_DATA" "$EVAL_SPLIT" <<'PY'
from pathlib import Path
import sys
from ultralytics.utils import YAML
cfg = YAML().load(Path(sys.argv[1]))
want = sys.argv[2]
for key in (want, "test", "val"):
    if isinstance(cfg.get(key), str):
        print(key)
        raise SystemExit
raise SystemExit(2)
PY
}

run_compare() {
    echo "Comparing ${DATA_TAG}: ratios ${RATIOS[*]}  splits ${SPLITS[*]}"

    if [[ ! -f "$SRC_DATA" ]]; then
        echo "Source yaml not found: $SRC_DATA" >&2
        echo "Fix SRC_DATA in common.sh" >&2
        return 1
    fi
    if [[ ! -f "$EVAL_YAML" ]]; then
        echo "Eval yaml not found: $EVAL_YAML" >&2
        echo "Fix EVAL_YAML in common.sh" >&2
        return 1
    fi
    local resolved_split
    if ! resolved_split="$(resolve_eval_split)"; then
        echo "No val/test split in ${SRC_DATA}; skip eval comparison" >&2
        return 2
    fi
    EVAL_SPLIT="$resolved_split"

    mkdir -p "$SUBSET_ROOT" "$RUN_ROOT"
    echo "ratio,method,mAP50,run_dir,subset_yaml" > "$SUMMARY_CSV"

    local ratio tag all_random
    for ratio in "${RATIOS[@]}"; do
        tag="$(ratio_tag "$ratio")"
        echo "========== ${DATA_TAG} ratio ${ratio} coreset =========="
        run_one coreset "$ratio" "${SUBSET_ROOT}/coreset_r${tag}" "${RUN_ROOT}/coreset_r${tag}" ""
        echo "========== ${DATA_TAG} ratio ${ratio} random =========="
        all_random="$(IFS=','; echo "${SPLITS[*]}")"
        run_one random "$ratio" "${SUBSET_ROOT}/random_r${tag}" "${RUN_ROOT}/random_r${tag}" "$all_random"
    done

    python tools/summarize_prune_compare.py --csv "$SUMMARY_CSV" --min_interval "$MIN_INTERVAL"
}

bind_domain() {
    local domain="$1"
    SRC_DATA="${SRC_DATA_ROOT}/${domain}/dataset.yaml"
    DATA_TAG="${DATA_TAG_PREFIX}${domain}"
    EVAL_YAML="$SRC_DATA"
    FEATURE_CACHE_DIR="data/prune_subsets/${DATA_TAG}/feature_cache"
    SUBSET_ROOT="data/prune_subsets/${DATA_TAG}"
    RUN_ROOT="runs/prune_compare/${DATA_TAG}"
    SUMMARY_CSV="$RUN_ROOT/summary.csv"
    EVAL_SPLIT="$PREFERRED_EVAL_SPLIT"
}

for cmd in "${PREP_CMDS[@]+"${PREP_CMDS[@]}"}"; do
    eval "$cmd"
done

DOMAINS=("${DOMAINS[@]+"${DOMAINS[@]}"}")
if (( ${#DOMAINS[@]} > 0 )); then
    if [[ -z "${SRC_DATA_ROOT:-}" || -z "${DATA_TAG_PREFIX:-}" ]]; then
        echo "DOMAINS is set; also set SRC_DATA_ROOT and DATA_TAG_PREFIX in common.sh" >&2
        exit 1
    fi
    family_status=0
    family_summary="${FAMILY_SUMMARY:-runs/prune_compare/${DATA_TAG_PREFIX}family_summary.csv}"
    mkdir -p "$(dirname "$family_summary")"
    echo "domain,ratio,random,coreset,delta,pass" > "$family_summary"
    for domain in "${DOMAINS[@]}"; do
        bind_domain "$domain"
        if run_compare; then
            domain_pass=1
        else
            cmp_status=$?
            if [[ "$cmp_status" -eq 2 ]]; then
                echo "SKIP ${domain}: no held-out eval split"
                echo "${domain},NA,NA,NA,NA,skip" >> "$family_summary"
                continue
            fi
            domain_pass=0
            family_status=1
        fi
        python - "$SUMMARY_CSV" "$family_summary" "$domain" "$domain_pass" <<'PY'
import csv, sys
from collections import defaultdict
from pathlib import Path
src, dst, domain, passed = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
table = defaultdict(dict)
with src.open(encoding="utf-8", newline="") as handle:
    for row in csv.DictReader(handle):
        table[float(row["ratio"])][row["method"].strip()] = float(row["mAP50"])
with dst.open("a", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle)
    for ratio in sorted(table):
        random_map = table[ratio]["random"]
        coreset_map = table[ratio]["coreset"]
        writer.writerow(
            [domain, f"{ratio:.2f}", f"{random_map:.5f}", f"{coreset_map:.5f}",
             f"{coreset_map - random_map:+.5f}", passed]
        )
PY
    done
    echo "Family summary: $family_summary"
    exit "$family_status"
fi

run_compare
