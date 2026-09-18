"""Point extra splits of a pruned dataset yaml at the source dataset.

Used by prune-vs-random comparison so only the compressed split (usually train)
changes, while val/test stay on the full original images. Paths written into
the destination yaml are relative to that yaml.

Usage:
    $ python tools/attach_source_splits.py \\
        --src_data data/VOC-YOLO/VOC.yaml \\
        --dst_yaml data/prune_subsets/VOC/coreset_r0p10/dataset.yaml \\
        --splits val test
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from prune_dataset import _resolve_split_dirs
from ultralytics.utils import LOGGER, YAML


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src_data", type=Path, required=True, help="Source dataset YAML")
    parser.add_argument("--dst_yaml", type=Path, required=True, help="Pruned dataset YAML to update")
    parser.add_argument("--splits", nargs="+", required=True, help="Source splits to attach")
    return parser.parse_args()


def main() -> None:
    """Write relative paths for the requested source splits into the destination yaml."""
    args = parse_args()
    source_cfg = args.src_data.resolve()
    dst_yaml = args.dst_yaml.resolve()
    if not dst_yaml.is_file():
        raise FileNotFoundError(f"Destination yaml does not exist: {dst_yaml}")
    dataset_config = YAML().load(source_cfg)
    output_config = YAML().load(dst_yaml)
    for split in args.splits:
        resolved = _resolve_split_dirs(source_cfg, dataset_config, split)
        if resolved is None:
            LOGGER.warning(f"Source dataset '{source_cfg}' has no '{split}' split; skipping attach")
            continue
        image_root, _label_root = resolved
        output_config[split] = Path(os.path.relpath(image_root, start=dst_yaml.parent)).as_posix()
        LOGGER.info(f"Attached {split}: {output_config[split]}")
    YAML().save(dst_yaml, output_config)


if __name__ == "__main__":
    main()
