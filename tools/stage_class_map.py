"""Write the per-stage class-name lists used by training-time stage-wise mAP.

Usage:
    python tools/stage_class_map.py \
        --datasets data/VOC_10+10/task_1_cls_10/dataset.yaml \
                   data/VOC_10+10/task_2_cls_10/dataset.yaml \
        --output runs/<run>/task-2/stage_classes.json

The output is a JSON list of lists: stage i holds the class names introduced by the
i-th incremental task, in the dataset's class-id order. The trainer aggregates its
validation metrics over each list and selects the best checkpoint with the last
(current) stage's mAP.
"""

import argparse
import json

from utils import normalize_names

from ultralytics.utils import LOGGER, YAML


def main() -> None:
    """Read the task YAML sequence and write the stage class-name lists."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", required=True, help="Incremental task dataset YAMLs, in task order")
    parser.add_argument("--output", required=True, help="Destination JSON path")
    args = parser.parse_args()

    stages = []
    for dataset in args.datasets:
        config = YAML().load(dataset)
        if "names" not in config:
            raise KeyError(f"Dataset config has no 'names': {dataset}")
        names = normalize_names(config["names"], source=f"dataset '{dataset}'")
        stages.append([names[class_id] for class_id in sorted(names)])

    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(stages, file, ensure_ascii=False, indent=2)
    LOGGER.info(f"Saved {len(stages)} stage class lists to {args.output}")


if __name__ == "__main__":
    main()
