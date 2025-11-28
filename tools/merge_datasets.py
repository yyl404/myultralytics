""" Merge multiple datasets into a single dataset.

Usage:
    $ python tools/merge_datasets.py \
        --datasets <path/to/dataset1.yaml> <path/to/dataset2.yaml> ... \
        --output_dir <path/to/output_dir>

Arguments:
    --datasets: Paths to the source dataset YAML files
    --output_dir: Path to the output directory where merged dataset will be saved

Examples:
    $ python tools/merge_datasets.py \
        --datasets runs/yolov8l_voc_inc_10_10_fromscratch_vspreg/task-2/task_2_cls_10_train_pseudo_labels/dataset.yaml \
            data/VOC_inc_10_10/task_2_cls_10/dataset.yaml \
        --output_dir runs/yolov8l_voc_inc_10_10_fromscratch_vspreg/task-2/task_2_cls_10_merged
"""

import argparse
import os
import os.path as OSP
import shutil

from ultralytics.utils import YAML, LOGGER
from utils import merge_labels_from_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, required=True, nargs="+",
        help="Paths to the source dataset YAML files")
    parser.add_argument("--output_dir", type=str, required=True,
        help="Path to the output directory where merged dataset will be saved")
    args = parser.parse_args()

    if len(args.datasets) < 2:
        raise ValueError("At least 2 datasets are required for merging")

    # Load all dataset configurations and collect classes
    dataset_configs = []
    merged_classes = set()
    
    for dataset_path in args.datasets:
        if not OSP.exists(dataset_path):
            raise ValueError(f"Dataset file not found: {dataset_path}")
        
        cfg = YAML.load(dataset_path)
        
        # Extract classes from this dataset
        if "names" in cfg:
            if isinstance(cfg["names"], dict):
                classes = [cfg["names"][i] for i in sorted(cfg["names"].keys())]
            else:
                classes = list(cfg["names"])
        else:
            classes = []
        
        dataset_configs.append((dataset_path, cfg, classes))
        
        # Collect all classes
        merged_classes.update(classes)
    
    # Create merged classes mapping
    merged_class_to_id = {cls: i for i, cls in enumerate(sorted(list(merged_classes)))}
    
    # Create class ID mappings for each dataset
    dataset_class_id_maps = []
    for dataset_path, cfg, classes in dataset_configs:
        class_id_map = {}
        for old_id, cls in enumerate(classes):
            if cls in merged_class_to_id:
                new_id = merged_class_to_id[cls]
                class_id_map[old_id] = new_id
            else:
                LOGGER.warning(f"Class '{cls}' from {dataset_path} not found in merged classes")
        dataset_class_id_maps.append(class_id_map)
    
    # Create output directory
    if OSP.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create merged dataset configuration
    merged_config = {
        'names': {i: cls for i, cls in enumerate(sorted(list(merged_classes)))}
    }
    
    # Process each split
    splits = ['train', 'val', 'test']
    for split in splits:
        label_dirs = []
        image_dirs = []
        split_class_id_maps = []  # Class ID maps for this split
        
        # Collect label and image directories for this split from all datasets
        for idx, (dataset_path, cfg, classes) in enumerate(dataset_configs):
            if split in cfg:
                dataset_dir = OSP.dirname(dataset_path)
                # Handle both absolute and relative paths
                if 'path' in cfg:
                    images_path = OSP.join(cfg['path'], cfg[split])
                else:
                    images_path = OSP.join(dataset_dir, cfg[split])
                
                labels_path = images_path.replace('images', 'labels')
                
                if OSP.exists(images_path):
                    image_dirs.append(images_path)
                if OSP.exists(labels_path):
                    label_dirs.append(labels_path)
                    split_class_id_maps.append(dataset_class_id_maps[idx])
        
        # Merge labels if any exist
        if label_dirs:
            output_labels_dir = OSP.join(args.output_dir, f"labels/{split}")
            os.makedirs(output_labels_dir, exist_ok=True)
            merge_labels_from_dir(label_dirs, output_labels_dir, class_id_maps=split_class_id_maps)
        
        # Merge images (copy from all datasets)
        if image_dirs:
            output_images_dir = OSP.join(args.output_dir, f"images/{split}")
            os.makedirs(output_images_dir, exist_ok=True)
            
            # Copy images from all datasets
            for image_dir in image_dirs:
                for image_file in os.listdir(image_dir):
                    src_path = OSP.join(image_dir, image_file)
                    dst_path = OSP.join(output_images_dir, image_file)
                    
                    # Skip if file already exists (from previous dataset)
                    if not OSP.exists(dst_path):
                        shutil.copy2(src_path, dst_path)
            
            merged_config[split] = f"images/{split}"
    
    # Save merged dataset configuration
    config_path = OSP.join(args.output_dir, "dataset.yaml")
    YAML.save(data=merged_config, file=config_path)
