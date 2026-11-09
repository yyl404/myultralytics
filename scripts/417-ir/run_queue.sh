#!/bin/bash
# Sequential run queue shared by the three 417-ir GPU schedulers.
#
#   bash scripts/417-ir/run_queue.sh <queue> <device-list>
#
# Every experiment of the queue is trained and evaluated one after the other on
# the whole device list: training runs as multi-GPU DDP (DEVICE=0,1,2,3), eval
# scores on the first device of the list. One failing experiment does not stop
# the rest of the queue.
#
# Checkpoint inheritance (resume):
#   * a run whose evaluation finished (.eval_complete) is skipped;
#   * otherwise training continues from the furthest task-*/best.pt already on
#     disk (START_TASK), so finished stages are never retrained;
#   * an interrupted stage resumes from its own train/weights/last.pt instead of
#     being restarted from the previous stage's checkpoint;
#   * a run that has every best.pt but no finished evaluation only re-runs eval;
#   * FORCE=1 deletes the run directory and trains everything from scratch.
# When a resumed stage finds the previous stage's PCA cache / replay dataset
# missing, the model adapter regenerates it before training starts.
#
# Only train + eval run here; the predict stage is not needed for get_results.sh.
#
# Env: RUN_ROOT (default runs/417-ir), FORCE=1 for a clean rerun, DRY_RUN=1 to
#      print the commands without running them, plus every knob understood by
#      the experiment common.sh (EPOCHS, BATCH_SIZE, REPLAY_SAMPLE_NUM, ...).
set -euo pipefail

QUEUE="${1:?usage: run_queue.sh <queue> <device-list>}"
DEVICE="${2:?usage: run_queue.sh <queue> <device-list>}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export DEVICE
export RUN_ROOT="${RUN_ROOT:-runs/417-ir}"
RUN_ROOT="${RUN_ROOT%/}"

PLAN="$(python tools/ir_campaign.py plan --queue "$QUEUE" --root "$RUN_ROOT")"

failed=()
while IFS=$'\t' read -r run_dir method script_dir stage_count; do
    [[ -n "$run_dir" ]] || continue
    [[ -n "$stage_count" ]] || {
        echo "Plan line without a stage count: $run_dir" >&2
        exit 1
    }

    force="${FORCE:-0}"
    if [[ "$force" != 1 && -f "$run_dir/.eval_complete" && -f "$run_dir/task-${stage_count}/best.pt" ]]; then
        echo "[skip] $run_dir (finished; FORCE=1 to rerun)"
        continue
    fi
    if [[ "$force" == 1 ]]; then
        case "$run_dir" in
            "$RUN_ROOT"/*)
                if [[ "${DRY_RUN:-0}" == 1 ]]; then
                    echo "[plan] rm -rf $run_dir"
                else
                    rm -rf -- "$run_dir"
                fi
                ;;
            *)
                echo "Refusing to remove $run_dir: outside $RUN_ROOT" >&2
                exit 1
                ;;
        esac
    fi

    # Furthest stage whose best.pt exists; everything before it is inherited.
    completed=0
    if [[ "$force" != 1 ]]; then
        for ((index = 1; index <= stage_count; index++)); do
            [[ -f "$run_dir/task-${index}/best.pt" ]] || break
            completed=$index
        done
    fi

    START_TASK=$((completed + 1))
    RESUME_CHECKPOINT=""
    if [[ "$force" != 1 && -f "$run_dir/task-${START_TASK}/train/weights/last.pt" ]]; then
        RESUME_CHECKPOINT="$run_dir/task-${START_TASK}/train/weights/last.pt"
    fi
    export START_TASK RESUME_CHECKPOINT

    echo "=================================================================="
    echo "[run ] $script_dir  method=$method  gpus=$DEVICE"
    echo "       -> $run_dir"
    if (( completed == 0 )); then
        echo "       resume: nothing yet (start from task-1)"
    elif (( completed == stage_count )); then
        echo "       resume: all ${stage_count} stages trained, evaluation missing; eval only"
    else
        echo "       resume: ${completed}/${stage_count} stages done, inherit $run_dir/task-${completed}/best.pt"
    fi
    if [[ -n "$RESUME_CHECKPOINT" ]]; then
        echo "       resume: task-${START_TASK} from $RESUME_CHECKPOINT"
    fi
    echo "=================================================================="

    if [[ "${DRY_RUN:-0}" == 1 ]]; then
        if (( completed < stage_count )); then
            echo "[plan] START_TASK=$START_TASK RESUME_CHECKPOINT=$RESUME_CHECKPOINT METHOD=$method RUN_DIR=$run_dir bash $script_dir/train.sh"
        fi
        echo "[plan] METHOD=$method RUN_DIR=$run_dir bash $script_dir/eval.sh"
        continue
    fi

    rm -f "$run_dir/.eval_complete"
    status=0
    if (( completed < stage_count )); then
        METHOD="$method" RUN_DIR="$run_dir" bash "$script_dir/train.sh" || status=$?
    fi
    if (( status == 0 )); then
        METHOD="$method" RUN_DIR="$run_dir" bash "$script_dir/eval.sh" || status=$?
    fi
    if (( status == 0 )); then
        touch "$run_dir/.eval_complete"
        echo "[done] $run_dir"
    else
        echo "[fail] $run_dir" >&2
        failed+=("$run_dir")
    fi
done <<< "$PLAN"

if (( ${#failed[@]} > 0 )); then
    echo "" >&2
    echo "以下实验失败，请检查日志：" >&2
    printf '  %s\n' "${failed[@]}" >&2
    exit 1
fi
echo "队列完成：$QUEUE（GPU $DEVICE）"
