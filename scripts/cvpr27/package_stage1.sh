#!/bin/bash
# Package the code revision and the two shared stage-1 checkpoints.
# Runs after both stage-1 files exist. Optionally uploads when OSS_DEST is set
# and `oss` is logged in; otherwise the archive stays under runs/_cvpr27_uploads/.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

S15="$REPO_ROOT/runs/_cvpr27_stage1/VOC_15+5/task-1-best.pt"
S19="$REPO_ROOT/runs/_cvpr27_stage1/VOC_19+1/task-1-best.pt"
for f in "$S15" "$S19"; do
    [[ -f "$f" ]] || { echo "missing stage-1 checkpoint: $f" >&2; exit 1; }
done

OUT="$REPO_ROOT/runs/_cvpr27_uploads"
mkdir -p "$OUT"
STAMP="$(date '+%Y%m%d_%H%M%S')"
PKG="$OUT/cvpr27_stage1_${STAMP}.tar.gz"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

mkdir -p "$TMP/code/scripts" "$TMP/code/ultralytics/data" "$TMP/code/tools" \
         "$TMP/weights" "$TMP/configs" "$TMP/data"

# New files (not in git diff) plus the tracked working-tree patch.
cp -a scripts/cvpr27 "$TMP/code/scripts/"
cp -f ultralytics/data/class_mapped.py "$TMP/code/ultralytics/data/"
cp -f tools/make_class_mapped_yaml.py "$TMP/code/tools/"
cp -f README.md "$TMP/code/"
git diff > "$TMP/code/working_tree.patch"
git status --short > "$TMP/code/git_status.txt"

# Stage-1 checkpoints and the pretrained backbone.
cp -f "$S15" "$TMP/weights/VOC_15+5-task-1-best.pt"
cp -f "$S19" "$TMP/weights/VOC_19+1-task-1-best.pt"
if [[ -f "$REPO_ROOT/yolov8m-cls.pt" ]]; then
    cp -Lf "$REPO_ROOT/yolov8m-cls.pt" "$TMP/weights/yolov8m-cls.pt"
fi

# Stage-1 training configs and logs.
for split in VOC_15+5 VOC_19+1; do
    run="$REPO_ROOT/runs/yolov8m_${split}_pretrained-from-yolov8m-cls_naive"
    mkdir -p "$TMP/configs/$split"
    for f in "$run/task-1/train/args.yaml" "$run/task-1/train/results.csv"; do
        [[ -f "$f" ]] && cp -f "$f" "$TMP/configs/$split/"
    done
done

# Dataset yamls only (no images).
while IFS= read -r f; do
    rel="${f#"$REPO_ROOT"/}"
    mkdir -p "$TMP/data/$(dirname "$rel")"
    cp -f "$f" "$TMP/data/$rel"
done < <(find data/VOC_15+5 data/VOC_19+1 -name dataset.yaml 2>/dev/null)

# Local result row if the ESP run already finished.
if [[ -f "$REPO_ROOT/results/cvpr27/machine1.csv" ]]; then
    cp -f "$REPO_ROOT/results/cvpr27/machine1.csv" "$TMP/configs/"
fi

{
    echo "CVPR-27 stage-1 package"
    echo "created: $(date -Is)"
    echo "host: $(hostname)"
    echo "commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo
    echo "== git status =="
    git status --short
    echo
    echo "== stage-1 weights =="
    sha256sum "$S15" "$S19"
} > "$TMP/MANIFEST.txt"

tar -czf "$PKG" -C "$TMP" .
sha256sum "$PKG" > "$PKG.sha256"
echo "package: $PKG"
echo "sha256:  $(cat "$PKG.sha256")"

if [[ -n "${OSS_DEST:-}" ]]; then
    echo "uploading to $OSS_DEST"
    oss cp "$PKG" "$OSS_DEST"
fi
