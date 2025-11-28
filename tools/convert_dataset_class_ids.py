"""Convert dataset class IDs to adapt to model's classification head.

This tool converts the class IDs in a dataset to match the class IDs used by a trained model.
It reads the model's class list and the dataset's class list, creates a mapping between them,
and converts all label files accordingly.

Usage:
    $ python tools/convert_dataset_class_ids.py \
        --model <path/to/model.pt> \
        --dataset <path/to/dataset.yaml> \
        --output_dir <path/to/output_dir> \
        --splits <split1> <split2> ... (optional) \
        --keep_unrecognized_classes (optional)

    Arguments:
        --model: Path to the model checkpoint (.pt file)
        --dataset: Path to the source dataset YAML file
        --output_dir: Path to the output directory where converted dataset will be saved
        --splits: Dataset splits to convert class IDs for (default: "train val test")
        --keep_unrecognized_classes: Whether to keep classes not in the model's class list
            (default: False, unrecognized classes will be skipped)

Examples:
    $ python tools/convert_dataset_class_ids.py \
        --model runs/yolov8l_voc_inc_10_10_fromscratch_vspreg/task-2/task-1-best-expanded.pt \
        --dataset data/VOC_inc_10_10/task_2_cls_10/dataset.yaml \
        --output_dir runs/yolov8l_voc_inc_10_10_fromscratch_vspreg/task-2/task_2_cls_10_val-test_converted \
        --splits val test
"""


import argparse
import os.path as OSP
import os
import shutil

from ultralytics import YOLO
from ultralytics.utils import YAML, LOGGER

from utils import convert_class_ids_from_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to the model checkpoint (.pt file)")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset YAML file")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory")
    parser.add_argument("--splits", type=str, required=False, default=["train", "val", "test"], nargs="+",
        help="Dataset splits to convert class IDs for (default: ['train', 'val', 'test'])")
    parser.add_argument("--keep_unrecognized_classes", type=bool, required=False, default=False,
        help="Whether to keep unrecognized classes (default: False)")
    args = parser.parse_args()

    model = YOLO(args.model)
    model_classes = [model.names[i] for i in sorted(model.names.keys())]

    data_cfg = YAML.load(args.dataset)
    source_classes = [data_cfg["names"][i] for i in sorted(data_cfg["names"].keys())]
    
    class_id_map = {}
    for i, cls in enumerate(source_classes):
        if cls in model_classes:
            class_id_map[i] = model_classes.index(cls)
        else:
            if args.keep_unrecognized_classes:
                # Map the classes that are not in the model output class list
                # to the index space beyond the model output channel number
                class_id_map[i] = len(model_classes)
                model_classes.append(cls)
            else:
                LOGGER.warning(f"Class {cls} not found in model classes, skipped")
    
    config = {
        'names': {i: cls for i, cls in enumerate(model_classes)}
    } # target dataset config file

    if OSP.exists(args.output_dir):
        shutil.rmtree(args.output_dir)

    for split in args.splits:
        if split in data_cfg:
            source_images = OSP.join(data_cfg['path'], data_cfg[split]) if 'path' in data_cfg.keys() else \
                        OSP.join(OSP.dirname(args.dataset), data_cfg[split])
            source_labels = source_images.replace('images', 'labels')

            os.makedirs(OSP.join(args.output_dir, f"labels/{split}"), exist_ok=True)
            convert_class_ids_from_dir(source_labels, class_id_map, OSP.join(args.output_dir, f"labels/{split}"))

            shutil.copytree(source_images, OSP.join(args.output_dir, f"images/{split}"))
            config[split] = f"images/{split}"
        else:
            LOGGER.warning(f"Split {split} not found in dataset YAML file, skipped")
    
    config_path = OSP.join(args.output_dir, f"dataset.yaml")
    YAML.save(data=config, file=config_path)