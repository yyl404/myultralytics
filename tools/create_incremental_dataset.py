"""Create class-incremental dataset by splitting source dataset into multiple tasks.

Usage:
    $ python tools/create_incremental_dataset.py \\
        --source_cfg <path/to/source_dataset.yaml> \\
        --output_dir <path/to/output_dir> \\
        --n_classes <n1> <n2> ... \\
        --classes <class_list> (optional) \\
        --split <split1> <split2> ... (optional)

    Arguments:
        --source_cfg: Path to the source dataset configuration file (.yaml file)
        --output_dir: Path to the output directory where incremental dataset will be created
        --n_classes: Number of classes for each task (space-separated list, e.g., "15 5" for two tasks)
        --classes: List of class names for each task as a Python list string, 
            e.g., "[['person', 'car'], ['bus', 'truck']]"
            (if not provided, source dataset classes will be split by --n_classes in default order)
        --split: Dataset splits to create (default: "train val test")
        
Examples:
    # Create incremental dataset with 15 classes in task 1 and 5 classes in task 2
    $ python tools/create_incremental_dataset.py \\
        --source_cfg data/VOC-YOLO/VOC.yaml \\
        --output_dir data/VOC_inc_15_5 \\
        --n_classes 15 5

    # Create incremental dataset with 5 tasks, each with 1 class
    $ python tools/create_incremental_dataset.py \\
        --source_cfg data/VOC-YOLO/VOC.yaml \\
        --output_dir data/VOC_inc_15_1x5 \\
        --n_classes 15 1 1 1 1 1

    # Create incremental dataset with custom class assignments
    $ python tools/create_incremental_dataset.py \\
        --source_cfg data/VOC-YOLO/VOC.yaml \\
        --output_dir data/VOC_inc_custom \\
        --classes "[['person', 'car'], ['bus', 'truck']]" \\
        --split train val
"""

import argparse
import sys
import traceback
import os
import shutil
import glob
from tqdm import tqdm

from ultralytics.utils import YAML, LOGGER

from utils import parse_list_string


SUPPORTED_IMAGE_EXTENSIONS = ['.jpg', '.png', '.jpeg', 'bmp']
SUPPORTED_LABEL_EXTENSIONS = ['.txt']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create class-incremental dataset")
    parser.add_argument("--source_cfg", type=str, required=True, help="The path to the source dataset's config file")
    parser.add_argument("--output_dir", type=str, required=True, help="The path to create output dataset's directory")
    parser.add_argument("--n_classes", type=int, nargs='+', required=True, help="The number of classes for each task")
    parser.add_argument("--classes", type=str, default=None, help="The classes of each task as a Python list string, "
        "e.g., \"[['person', 'car'], ['bus', 'truck']]\". "
        "If not provided, source dataset classes will be split by --n_classes in default order")
    parser.add_argument("--split", type=str, default=['train', 'val', 'test'], nargs='+', help="The splits to create the \
        dataset for, default is \"train val test\"")
    args = parser.parse_args()

    # Parse class assignments from --classes if provided
    if args.classes is not None:
        args.classes = parse_list_string(args.classes)

    # load source dataset classes
    source_dataset_yaml = YAML().load(args.source_cfg)
    source_classes = source_dataset_yaml["names"]

    # check if the source dataset has corresponding splits
    splits = []
    for _split in args.split:
        if _split in source_dataset_yaml.keys():
            splits.append(_split)
        else:
            LOGGER.warning(f"WARNING ⚠️ Source dataset config file {args.source_cfg} does not have corresponding split {_split}, \
                skipping...")

    # check if the output directory exists
    if os.path.exists(args.output_dir):
        LOGGER.info(f"INFO ℹ️ Output directory {args.output_dir} already exists, remove it or not? (Yes/No/Cancel)")
        answer = input()
        if answer.lower() == "yes" or answer.lower() == "y":
            shutil.rmtree(args.output_dir)
            LOGGER.info(f"Output directory {args.output_dir} removed.")
        elif answer.lower() == "cancel" or answer.lower() == "c":
            LOGGER.info("Aborting...")
            sys.exit(1)

    task_classes = []
    classes_id_map_source2task = []
    if args.classes is None:
        # split the source dataset classes by provided n_classes
        for t, n_classes in enumerate(args.n_classes):
            task_classes.append({})
            classes_id_map_source2task.append({}) # map from source class id to task class id
            for i in range(n_classes):
                class_name = source_classes[sum(args.n_classes[:t]) + i]
                task_classes[t][i] = class_name
                classes_id_map_source2task[t][sum(args.n_classes[:t]) + i] = i
    else:
        # use the provided classes
        for t, _classes in enumerate(args.classes):
            task_classes.append({})
            classes_id_map_source2task.append({})
            task_class_id = 0
            for _class_name in _classes:
                # Find matching source class id
                found = False
                for source_class_id in sorted(source_classes.keys()):
                    source_class_name = source_classes[source_class_id]
                    if source_class_name == _class_name:
                        task_classes[t][task_class_id] = _class_name
                        classes_id_map_source2task[t][source_class_id] = task_class_id
                        task_class_id += 1
                        found = True
                        break
                if not found:
                    LOGGER.warning(f"WARNING ⚠️ Class '{_class_name}' not found in source dataset classes, skipping...")

    # initialize the cumulative task classes and classes id map
    task_cumulative_classes = []
    classes_id_map_source2task_cumulative = []
    for t in range(len(task_classes)):
        task_cumulative_classes.append({})
        classes_id_map_source2task_cumulative.append({})
        class_counter = 0
        for i in range(t+1):
            # Iterate task classes in sorted order by class ID for consistency
            for task_class_id in sorted(task_classes[i].keys()):
                _class_name = task_classes[i][task_class_id]
                # Find matching source class id (search in sorted order for consistency)
                for source_class_id in sorted(source_classes.keys()):
                    source_class_name = source_classes[source_class_id]
                    if source_class_name == _class_name:
                        task_cumulative_classes[t][class_counter] = _class_name
                        classes_id_map_source2task_cumulative[t][source_class_id] = class_counter
                        class_counter += 1
                        break

    # initialize task image counts
    task_image_counts = {t: {_split: 0 for _split in splits} for t in range(len(task_classes))}
    
    # create output directory for each task
    for t in range(len(task_classes)):
        for _split in splits:
            os.makedirs(os.path.join(args.output_dir, f"task_{t+1}_cls_{len(task_classes[t].values())}/images/{_split}"), exist_ok=True)
            os.makedirs(os.path.join(args.output_dir, f"task_{t+1}_cls_{len(task_classes[t].values())}/labels/{_split}"), exist_ok=True)
            if t>0:
                os.makedirs(os.path.join(args.output_dir,
                    f"task_1-{t+1}_cls_{len(task_cumulative_classes[t].values())}/images/{_split}"),
                    exist_ok=True)
                os.makedirs(os.path.join(args.output_dir,
                    f"task_1-{t+1}_cls_{len(task_cumulative_classes[t].values())}/labels/{_split}"),
                    exist_ok=True)
    
    # initialize the task image paths list
    # which stores the image paths for each split in each task
    task_image_paths = {t: {_split: [] for _split in splits} for t in range(len(task_classes))}
    
    for _split in splits:
        if isinstance(source_dataset_yaml[_split], str):
            image_dirs = [source_dataset_yaml[_split]]
            label_dirs = [source_dataset_yaml[_split].replace("images", "labels")]
        elif isinstance(source_dataset_yaml[_split], list):
            image_dirs = source_dataset_yaml[_split]
            label_dirs = [_image_dir.replace("images", "labels") for _image_dir in image_dirs]
        else:
            raise ValueError(f"Invalid split configuration in source dataset config file \
                (must be a string or a list of strings).")
        
        # get all label files and image files in the directories
        label_files = []
        image_files = []
        for _image_dir, _label_dir in zip(image_dirs, label_dirs):
            # If the image or label directory does not exist, 
            # treat it as a relative path from the source dataset config file and try again
            if not os.path.exists(_image_dir):
                _image_dir = os.path.join(os.path.dirname(args.source_cfg), _image_dir)
            if not os.path.exists(_label_dir):
                _label_dir = os.path.join(os.path.dirname(args.source_cfg), _label_dir)
            # If the image or label directory still does not exist, raise an error
            if not os.path.exists(_image_dir):
                raise ValueError(f"Image directory {_image_dir} does not exist.")
            if not os.path.exists(_label_dir):
                raise ValueError(f"Label directory {_label_dir} does not exist.")
            # Get all image and label files in the directories
            for _img_ext in SUPPORTED_IMAGE_EXTENSIONS:
                image_files.extend(glob.glob(os.path.join(_image_dir, f'*{_img_ext.lower()}')))
                image_files.extend(glob.glob(os.path.join(_image_dir, f'*{_img_ext.upper()}')))
            for _label_ext in SUPPORTED_LABEL_EXTENSIONS:
                label_files.extend(glob.glob(os.path.join(_label_dir, f'*{_label_ext.lower()}')))
                label_files.extend(glob.glob(os.path.join(_label_dir, f'*{_label_ext.upper()}')))
            
        # process each label file
        for _label_file in tqdm(label_files, desc=f"Processing {_split} split"):
            # read the label file, find out all classes in the file
            classes_in_file = set()
            try:
                with open(_label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:  # YOLO format: class_id x_center y_center width height
                            class_id = int(parts[0])
                            classes_in_file.add(class_id)
            except:
                LOGGER.warning(f"WARNING ⚠️ Error reading label file {_label_file}, details as follows: \n{traceback.format_exc()}")
                continue
                
            # find the tasks that have at least one class in common with the current file
            compatible_tasks = []
            for t in range(len(task_classes)):
                task_class_ids = set(classes_id_map_source2task[t].keys())
                if classes_in_file.intersection(task_class_ids):
                    compatible_tasks.append(t)
            
            if compatible_tasks:
                # calculate the ratio of (image count / task class count) for each compatible task
                task_ratios = {}
                for t in compatible_tasks:
                    ratio = task_image_counts[t][_split] / len(task_classes[t].values()) if len(task_classes[t].values()) > 0 else float('inf')
                    task_ratios[t] = ratio
                
                # Find the task with the smallest ratio (the task in lack of images)
                selected_task = min(task_ratios, key=task_ratios.get)
                
                # Get the corresponding image file path
                for _image_file in image_files:
                    if os.path.basename(_image_file).split('.')[0] == os.path.basename(_label_file).split('.')[0]:
                        source_image_path = _image_file
                        source_label_path = _label_file
                        break
                
                if os.path.exists(source_image_path):
                    # copy the image file
                    task_dir = os.path.join(args.output_dir, f"task_{selected_task+1}_cls_{len(task_classes[selected_task].values())}")
                    dest_image_path = os.path.join(task_dir, "images", _split, os.path.basename(source_image_path))
                    shutil.copy2(source_image_path, dest_image_path)

                    # copy the label file and convert the class ids
                    dest_label_path = os.path.join(task_dir, "labels", _split, os.path.basename(source_label_path))
                    with open(source_label_path, 'r') as src_f, open(dest_label_path, 'w') as dst_f:
                        for i, line in enumerate(src_f):
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                source_class_id = int(parts[0])
                                # filter out instances whose class ids are not in the selected task
                                # and convert the class ids to the inner ids of the selected task
                                if source_class_id in classes_id_map_source2task[selected_task].keys():
                                    task_class_id = classes_id_map_source2task[selected_task][source_class_id]
                                    parts[0] = str(task_class_id)
                                    dst_f.write(' '.join(parts) + '\n')
                            else:
                                LOGGER.warning(f"WARNING ⚠️ Invalid label format in file {source_label_path} line {i+1}, skipping...")
                                continue

                    # copy the image file to the cumulative task directories
                    for end_task in range(max(selected_task, 1), len(task_classes)):
                        task_cumulative_dir = os.path.join(args.output_dir, f"task_1-{end_task+1}_cls_{len(task_cumulative_classes[end_task].values())}")
                        dest_cumulative_image_path = os.path.join(task_cumulative_dir, "images", _split, os.path.basename(source_image_path))
                        shutil.copy2(source_image_path, dest_cumulative_image_path)

                        dest_cumulative_label_path = os.path.join(task_cumulative_dir, "labels", _split, os.path.basename(source_label_path))
                        with open(source_label_path, 'r') as src_f, open(dest_cumulative_label_path, 'w') as dst_f:
                            for line in src_f:
                                parts = line.strip().split()
                                if len(parts) >= 5:
                                    source_class_id = int(parts[0])
                                    if source_class_id in classes_id_map_source2task_cumulative[end_task].keys():
                                        task_class_id = classes_id_map_source2task_cumulative[end_task][source_class_id]
                                        parts[0] = str(task_class_id)
                                        dst_f.write(' '.join(parts) + '\n')
                                else:
                                    LOGGER.warning(f"WARNING ⚠️ Invalid label format in file {source_label_path} line {i+1}, skipping...")
                                    continue

                    # update the image counts
                    task_image_counts[selected_task][_split] += 1
    
    # create the yaml config file for each task
    for t in range(len(task_classes)):
        task_dir = os.path.join(args.output_dir, f"task_{t+1}_cls_{len(task_classes[t].values())}")
        task_cumulative_dir = os.path.join(args.output_dir, f"task_1-{t+1}_cls_{len(task_cumulative_classes[t].values())}")
        
        # create the task config
        task_config = {
            'names': task_classes[t]
        }
        for _split in splits:
            task_config[_split] = f"images/{_split}"
        yaml_path = os.path.join(task_dir, 'dataset.yaml')
        YAML().save(yaml_path, task_config)

        # create the cumulative config
        if t > 0:
            cumulative_config = {
                'names': task_cumulative_classes[t]
            }
            for _split in splits:
                cumulative_config[_split] = f"images/{_split}"
            yaml_cumulative_path = os.path.join(task_cumulative_dir, 'dataset.yaml')
            YAML().save(yaml_cumulative_path, cumulative_config)

        LOGGER.info(f"Task {t+1} completed: {len(task_classes[t].values())} classes")
        for _split in splits:
            LOGGER.info(f"  {_split}: {task_image_counts[t][_split]} images")
        LOGGER.info(f"  Task config saved to: {yaml_path}")
        if t > 0:
            LOGGER.info(f"  Cumulative config saved to: {yaml_cumulative_path}")