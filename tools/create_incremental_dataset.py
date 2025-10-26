import argparse
import yaml
import os
import shutil
import glob
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create class-incremental dataset")
    parser.add_argument("--source_cfg", type=str, required=True, help="The path to the source dataset's config file")
    parser.add_argument("--output_dir", type=str, required=True, help="The path to create output dataset's directory")
    parser.add_argument("--n_classes", type=int, nargs='+', required=True, help="The number of classes for each task")
    args = parser.parse_args()

    source_dataset_yaml = yaml.load(open(args.source_cfg, "r"), Loader=yaml.FullLoader)
    source_classes = source_dataset_yaml["names"]
    splits = [_split for _split in ["train", "val", "test"] if _split in source_dataset_yaml.keys()]

    if os.path.exists(args.output_dir):
        print(f"Output directory {args.output_dir} already exists, remove it or not? (y/n)")
        answer = input()
        if answer == "y":
            shutil.rmtree(args.output_dir)
        else:
            print("Aborting...")
            exit(1)

    task_classes = []
    classes_id_map_source2task = []
    for t, n_classes in enumerate(args.n_classes):
        task_classes.append({})
        classes_id_map_source2task.append({}) # map from source class id to task class id
        for i in range(n_classes):
            class_name = source_classes[sum(args.n_classes[:t]) + i]
            task_classes[t][i] = class_name
            classes_id_map_source2task[t][sum(args.n_classes[:t]) + i] = i

    # initialize task image counts
    task_image_counts = {t: {_split: 0 for _split in splits} for t in range(len(args.n_classes))}
    
    # create output directory for each task
    for t in range(len(args.n_classes)):
        for _split in splits:
            os.makedirs(os.path.join(args.output_dir, f"task_{t+1}_cls_{args.n_classes[t]}/images/{_split}"), exist_ok=True)
            os.makedirs(os.path.join(args.output_dir, f"task_{t+1}_cls_{args.n_classes[t]}/labels/{_split}"), exist_ok=True)
            if t>0:
                os.makedirs(os.path.join(args.output_dir, f"task_1-{t+1}_cls_{sum(args.n_classes[:t+1])}/images/{_split}"), exist_ok=True)
                os.makedirs(os.path.join(args.output_dir, f"task_1-{t+1}_cls_{sum(args.n_classes[:t+1])}/labels/{_split}"), exist_ok=True)
    
    # store the image paths for each task in each split
    task_image_paths = {t: {_split: [] for _split in splits} for t in range(len(args.n_classes))}
    
    for _split in splits:
        print(f"Processing {_split} split...")
        if isinstance(source_dataset_yaml[_split], str):
            image_dirs = [source_dataset_yaml[_split]]
            label_dirs = [source_dataset_yaml[_split].replace("images", "labels")]
        elif isinstance(source_dataset_yaml[_split], list):
            image_dirs = source_dataset_yaml[_split]
            label_dirs = [_image_dir.replace("images", "labels") for _image_dir in image_dirs]
        else:
            raise ValueError(f"Invalid split configuration in source dataset config file (must be a string or a list of strings).")
        
        # get all label files and image files in the directories
        label_files = []
        image_files = []
        for _image_dir, _label_dir in zip(image_dirs, label_dirs):
            # If the image or label directory is not absolute, make it absolute
            if not os.path.exists(_image_dir) or not os.path.exists(_label_dir):
                _image_dir = os.path.join(os.path.dirname(args.source_cfg), _image_dir)
                _label_dir = os.path.join(os.path.dirname(args.source_cfg), _label_dir)
            # If the image or label directory still does not exist, raise an error
            if not os.path.exists(_image_dir) or not os.path.exists(_label_dir):
                raise ValueError(f"Image or label directory {_image_dir} or {_label_dir} does not exist.")
            # There are multiple image file extensions, so we need to use glob to get all of them
            image_files.extend(glob.glob(os.path.join(_image_dir, '*.jpg')))
            image_files.extend(glob.glob(os.path.join(_image_dir, '*.png')))
            image_files.extend(glob.glob(os.path.join(_image_dir, '*.jpeg')))
            image_files.extend(glob.glob(os.path.join(_image_dir, '*.JPG')))
            image_files.extend(glob.glob(os.path.join(_image_dir, '*.PNG')))
            image_files.extend(glob.glob(os.path.join(_image_dir, '*.JPEG')))

            label_files.extend(glob.glob(os.path.join(_label_dir, '*.txt')))
            
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
                print(f"Error reading label file {_label_file}, skipping...")
                continue
                
            # find the tasks that have at least one class in common with the current file
            compatible_tasks = []
            for t in range(len(args.n_classes)):
                task_class_ids = set(classes_id_map_source2task[t].keys())
                if classes_in_file.intersection(task_class_ids):
                    compatible_tasks.append(t)
            
            if compatible_tasks:
                # calculate the ratio of (image count / task class count) for each compatible task
                task_ratios = {}
                for t in compatible_tasks:
                    ratio = task_image_counts[t][_split] / args.n_classes[t] if args.n_classes[t] > 0 else float('inf')
                    task_ratios[t] = ratio
                
                # Find the task with the smallest ratio
                selected_task = min(task_ratios, key=task_ratios.get)
                
                # Get the corresponding image file path
                for _image_file in image_files:
                    if os.path.basename(_image_file).split('.')[0] == os.path.basename(_label_file).split('.')[0]:
                        source_image_path = _image_file
                        source_label_path = _label_file
                        break
                
                if os.path.exists(source_image_path):
                    # copy the image file
                    task_dir = os.path.join(args.output_dir, f"task_{selected_task+1}_cls_{args.n_classes[selected_task]}")
                    dest_image_path = os.path.join(task_dir, "images", _split, os.path.basename(source_image_path))
                    shutil.copy2(source_image_path, dest_image_path)

                    # copy the label file and convert the class ids
                    dest_label_path = os.path.join(task_dir, "labels", _split, os.path.basename(source_label_path))
                    with open(source_label_path, 'r') as src_f, open(dest_label_path, 'w') as dst_f:
                        for line in src_f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                source_class_id = int(parts[0])
                                if source_class_id in classes_id_map_source2task[selected_task].keys():
                                    task_class_id = classes_id_map_source2task[selected_task][source_class_id]
                                    parts[0] = str(task_class_id)
                                    dst_f.write(' '.join(parts) + '\n')

                    # copy the image file to the accumulative task directory
                    # filter out instances whose class ids are greater than the accumulative task class id upper bound
                    for task_id_upper_bound in range(max(selected_task, 1), len(args.n_classes)):
                        task_accum_dir = os.path.join(args.output_dir, f"task_1-{task_id_upper_bound+1}_cls_{sum(args.n_classes[:task_id_upper_bound+1])}")
                        dest_accum_image_path = os.path.join(task_accum_dir, "images", _split, os.path.basename(source_image_path))
                        shutil.copy2(source_image_path, dest_accum_image_path)

                        dest_accum_label_path = os.path.join(task_accum_dir, "labels", _split, os.path.basename(source_label_path))
                        with open(source_label_path, 'r') as src_f, open(dest_accum_label_path, 'w') as dst_f:
                            for line in src_f:
                                parts = line.strip().split()
                                if len(parts) >= 5:
                                    source_class_id = int(parts[0])
                                    max_accum_class_id = max(classes_id_map_source2task[task_id_upper_bound].keys())
                                    if source_class_id <= max_accum_class_id:
                                        dst_f.write(' '.join(parts) + '\n')

                    # update the image counts
                    task_image_counts[selected_task][_split] += 1
    
    # create the yaml config file for each task
    for t in range(len(args.n_classes)):
        task_dir = os.path.join(args.output_dir, f"task_{t+1}_cls_{args.n_classes[t]}")
        task_accum_dir = os.path.join(args.output_dir, f"task_1-{t+1}_cls_{sum(args.n_classes[:t+1])}")
        
        # create the task config
        task_config = {
            'names': task_classes[t]
        }
        for _split in splits:
            task_config[_split] = f"images/{_split}"
        yaml_path = os.path.join(task_dir, 'dataset.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(task_config, f, default_flow_style=False)

        # create the accumulative config
        if t > 0:
            accumulative_config = {
                'names': {i: class_name for i, class_name in enumerate(source_classes.values()) if i < sum(args.n_classes[:t+1])}
            }
            for _split in splits:
                accumulative_config[_split] = f"images/{_split}"
            yaml_accum_path = os.path.join(task_accum_dir, 'dataset.yaml')
            with open(yaml_accum_path, 'w') as f:
                yaml.dump(accumulative_config, f, default_flow_style=False)

        print(f"Task {t+1} completed: {len(task_config['names'])} classes")
        for _split in splits:
            print(f"  {_split}: {task_image_counts[t][_split]} images")
        print(f"  Config saved to: {yaml_path}")

    # create the incremental config file
    incremental_config = {"tasks": {}}
    for t in range(len(args.n_classes)):
        incremental_config["tasks"][t] = os.path.join(f"task_{t+1}_cls_{args.n_classes[t]}", "dataset.yaml")
    
    with open(os.path.join(args.output_dir, "incremental_config.yaml"), "w") as f:
        yaml.dump(incremental_config, f, default_flow_style=False)