refactor(scripts,tools): streamline incremental training workflows and clean legacy experiment scripts

Consolidate the incremental-learning experiment layout by pruning outdated scenario scripts
(4-domain, COCO, RSAR, VOC 10_10/19_1, yoloev8/yolo11 variants) and standardizing VOC 15+5
entrypoints under a unified `voc-tiny/15_5` structure.

- remove a large set of deprecated shell pipelines and one-off templates to reduce maintenance burden
- migrate/rename key VOC 15+5 training scripts to `scripts/voc-tiny/15_5/yolov8/` and add a dedicated dataset creation entrypoint
- update remaining VOC 15+5 scripts to a consistent command-building style:
  - auto-built model/dataset/output naming
  - safer execution wrappers (`run_step`) and explicit error logging
  - resumable task flow with task-wise head expansion and ID conversion
  - unified anti-forgetting flags for `pseudo_label`, `espreg`, and `ewc`
- align eval dataset paths from `VOC_15_5` to `VOC_15+5`
- improve `tools/convert_dataset_class_ids.py`:
  - add `--no-use-link` option
  - replace directory-level symlink/copy behavior with per-file mirroring to avoid Path.resolve()-induced label path mismatch
  - add progress bars and recursive file counting for labels/images
- enhance `tools/utils.py` helpers:
  - recursive label conversion with nested directory preservation
  - new `mirror_image_files()` utility supporting symlink or copy mode
- update `tools/pca.py`:
  - support both `nn.Conv2d` and `CosineConv2d`
  - harden sampling edge cases for empty bbox batches
  - fix progress/memory monitor lifecycle and sample iteration bounds
- append latest ESPReg ablation notes to `experiments.md`

This commit mainly improves script reliability/reproducibility and fixes data-path correctness in
class-ID conversion, while reducing repository noise from obsolete experimental pipelines.