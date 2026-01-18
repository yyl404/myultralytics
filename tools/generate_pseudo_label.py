"""Generate pseudo labels for a dataset.

Usage:
    $ python tools/generate_pseudo_label.py \
        --model <path/to/model.pt> \
        --dataset <path/to/dataset.yaml> \
        --output_dir <path/to/output_dir> \
        --conf_threshold <confidence_threshold> \
        --filter_iou_threshold <iou_threshold> (optional) \
        --splits <split1> <split2> ... (optional)

Arguments:
    --model: Path to the model checkpoint (.pt file)
    --dataset: Path to the dataset YAML file
    --output_dir: Path to the output directory
    --conf_threshold: Confidence threshold for generating pseudo labels
    --filter_iou_threshold: IoU threshold for filtering duplicate annotations when merging
        ground truth labels and pseudo labels. When a pseudo label has IoU > threshold
        with any ground truth label, the pseudo label is discarded (default: 0.5)
    --splits: Dataset splits to generate pseudo labels for (default: "train val test")

Examples:
    $ python tools/generate_pseudo_label.py \
        --model runs/yolov8l_voc_inc_10_10_fromscratch_vspreg/task-1/best.pt \
        --dataset data/VOC_inc_10_10/task_2_cls_10/dataset.yaml \
        --output_dir runs/yolov8l_voc_inc_10_10_fromscratch_vspreg/task-2/task_2_cls_10_pseudo_labels \
        --conf_threshold 0.25 \
        --splits train
"""


import argparse
import shutil
import os
from os import path as OSP
from tqdm import tqdm

from ultralytics import YOLO
from ultralytics.utils import YAML, LOGGER

from utils import convert_class_ids_from_dir, merge_labels_from_dir


def create_pseudo_labels_dataset(model, base_class_id_map, new_class_id_map, cfg_source,
                                 output_dir, all_classes, splits, conf_threshold=0.25,
                                 filter_iou_threshold=0.5):
    """Create a pseudo labels dataset by merging ground truth labels and pseudo labels.
    
    Args:
        model: YOLO model for generating pseudo labels
        base_class_id_map: Class ID mapping for base classes
        new_class_id_map: Class ID mapping for new classes
        cfg_source: Path to source dataset config file
        output_dir: Output directory path
        all_classes: List of all class names
        splits: Dataset splits to process
        conf_threshold: Confidence threshold for pseudo label generation
        filter_iou_threshold: IoU threshold for filtering duplicate annotations.
            When merging ground truth and pseudo labels, if a pseudo label has IoU > threshold
            with any ground truth label, the pseudo label is discarded (default: 0.5)
    """
    dir_source = OSP.dirname(cfg_source) # source dataset directory
    cfg_source = YAML.load(cfg_source) # source dataset config file

    config = {
        'names': {i: cls for i, cls in enumerate(all_classes)}
    } # target dataset config file
    
    for split in splits:
        if split in cfg_source:
            source_images = OSP.join(cfg_source['path'], cfg_source[split]) if 'path' in cfg_source.keys() else \
                    OSP.join(dir_source, cfg_source[split])
            source_labels = source_images.replace('images', 'labels')

            # 1. Predict pseudo labels
            results = model.predict(source_images, conf=conf_threshold, save_txt=True, save_conf=False,
                                    stream=True, project=output_dir, name=f"pseudo_labels/{split}", verbose=False)
            for result in tqdm(results, desc=f"Generating pseudo labels for {split}", total=len(os.listdir(source_images)),
                                    position=0, leave=True, ncols=80):
                pass # while iterating through the results generator, the results will be automatically saved

            # 2. Convert the class IDs of the pseudo labels to the class IDs of the target dataset
            pseudo_labels_dir = OSP.join(output_dir, f"pseudo_labels/{split}/labels_converted")
            os.makedirs(pseudo_labels_dir, exist_ok=True)
            convert_class_ids_from_dir(OSP.join(output_dir, f"pseudo_labels/{split}/labels"),
                                       base_class_id_map,
                                       pseudo_labels_dir)
            
            # 3. Read the ground truth labels of the source dataset and convert the class IDs
            gt_labels_dir = OSP.join(output_dir, f"gt_labels/{split}/labels_converted")
            os.makedirs(gt_labels_dir, exist_ok=True)
            convert_class_ids_from_dir(source_labels, new_class_id_map, gt_labels_dir)
            
            # 4. Merge the ground truth labels and the pseudo labels
            os.makedirs(OSP.join(output_dir, f"labels/{split}"), exist_ok=True)
            merge_labels_from_dir([gt_labels_dir, pseudo_labels_dir],
                                  output_dir=OSP.join(output_dir, f"labels/{split}"),
                                  filter_iou_threshold=filter_iou_threshold)

            # 5. Copy the images to the output directory
            images_output_dir = OSP.join(output_dir, f"images/{split}")
            shutil.copytree(source_images, images_output_dir)
            
            # 6. Add path to the config file
            config[split] = f"images/{split}"
        
        else:
            LOGGER.warning(f"WARNING ⚠️ Source dataset config file {cfg_source} does not have corresponding"+\
                           f" split {split}, skipping...")
    
    # Save the config file
    config_path = OSP.join(output_dir, f"dataset.yaml")
    YAML.save(data=config, file=config_path)
    
    # Delete the temporary generated pseudo labels
    pseudo_labels_dir = OSP.join(output_dir, f"pseudo_labels")
    if OSP.exists(pseudo_labels_dir):
        shutil.rmtree(pseudo_labels_dir)
    
    gt_labels_dir = OSP.join(output_dir, f"gt_labels")
    if OSP.exists(gt_labels_dir):
        shutil.rmtree(gt_labels_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to the teacher model checkpoint (.pt file)")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset YAML file")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory")
    parser.add_argument("--conf_threshold", type=float, required=False, default=0.25,
        help="Confidence threshold for generating pseudo labels (default: 0.25)")
    parser.add_argument("--filter_iou_threshold", type=float, required=False, default=0.5,
        help="IoU threshold for filtering duplicate annotations when merging ground truth "
             "labels and pseudo labels. When a pseudo label has IoU > threshold with any "
             "ground truth label, the pseudo label is discarded (default: 0.5)")
    parser.add_argument("--splits", type=str, required=False, default=["train", "val", "test"], nargs="+",
        help="Dataset splits to generate pseudo labels for (default: ['train', 'val', 'test'])")
    args = parser.parse_args()

    model = YOLO(args.model)
    base_classes = [model.names[i] for i in sorted(model.names.keys())]

    data_cfg = YAML.load(args.dataset)
    new_classes = [data_cfg["names"][i] for i in sorted(data_cfg["names"].keys())]

    all_classes = list(set(base_classes).union(new_classes))

    base_class_id_map = {}
    for i, cls in enumerate(base_classes):
        base_class_id_map[i] = all_classes.index(cls)

    new_class_id_map = {}
    for i, cls in enumerate(new_classes):
        new_class_id_map[i] = all_classes.index(cls)
    
    if OSP.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    
    create_pseudo_labels_dataset(model, base_class_id_map, new_class_id_map,
                                 args.dataset,
                                 args.output_dir, all_classes, args.splits, args.conf_threshold,
                                 args.filter_iou_threshold)