#!/bin/bash
# Unified entry point for incremental-learning analysis tools.
#
# Usage:
#   bash scripts/analyze.sh <analysis> [tool arguments...]
#
# Analyses (run with no extra arguments to see each tool's own --help):
#   pca_projection    Kernel-update / value-shift PCA projection analysis (tools/vis.py)
#   kernel_projection Kernel update projection on principal components (tools/vis_kernel_proj_pc.py)
#   eigen_adjust      ESPReg eigenvalue adjustment visualization (tools/vis_eigen_adjust.py)
#   prototypes        RePRE prototype replay visualization (tools/vis_prototypes_det.py)
#   confusion_matrix  Old/new/background confusion matrix aggregation (tools/parse_confusion_matrix.py)
#
# Example:
#   bash scripts/analyze.sh pca_projection \
#       --pca_cache_path runs/<run>/task-1/pca_cache.joblib \
#       --base_model runs/<run>/task-1/best.pt \
#       --incremental_model runs/<run>/task-2/best.pt \
#       --save_dir runs/<run>/vis_task2_on_task1 \
#       --sample_dir data/<dataset>/task_1/images/val \
#       --label_dir data/<dataset>/task_1/labels/val

set -euo pipefail

if (( $# < 1 )); then
    sed -n '2,20p' "$0"
    exit 1
fi

ANALYSIS="$1"
shift

case "$ANALYSIS" in
    pca_projection)
        exec python tools/vis.py "$@"
        ;;
    kernel_projection)
        exec python tools/vis_kernel_proj_pc.py "$@"
        ;;
    eigen_adjust)
        exec python tools/vis_eigen_adjust.py "$@"
        ;;
    prototypes)
        exec python tools/vis_prototypes_det.py "$@"
        ;;
    confusion_matrix)
        exec python tools/parse_confusion_matrix.py "$@"
        ;;
    -h|--help|help)
        sed -n '2,20p' "$0"
        ;;
    *)
        echo "Unknown analysis: $ANALYSIS" >&2
        echo "Expected one of: pca_projection, kernel_projection, eigen_adjust, prototypes, confusion_matrix" >&2
        exit 1
        ;;
esac
