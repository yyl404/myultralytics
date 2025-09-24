import os
import shutil
import yaml

def read_stage_file(stage_file_path, field_data_root_dir):
    """
    Reads a stage file and returns a list of (original_image_path, original_label_path, data_type) tuples.
    Adjusts paths to be absolute and uses forward slashes.
    """
    image_label_pairs = []
    with open(stage_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(' ')
            if len(parts) == 2:
                image_relative_path = parts[0].replace('\\', '/')
                label_relative_path = parts[1].replace('\\', '/')

                # Determine data type (train or val) from the relative path
                data_type = 'train' if 'data/train' in image_relative_path else 'val'

                # Construct absolute paths for original files
                # The relative paths in stage_X.txt start from 'data/train' or 'data/val'
                # So we join field_data_root_dir with the rest of the path after 'data/'
                original_image_path = os.path.join(field_data_root_dir, image_relative_path.split('data/', 1)[1])
                original_label_path = os.path.join(field_data_root_dir, label_relative_path.split('data/', 1)[1])

                image_label_pairs.append((original_image_path, original_label_path, data_type))
    return image_label_pairs

def generate_dataset_yaml(task_path, class_names, train_path, val_path, test_path):
    """
    Generates a dataset.yaml file for the given task.
    """
    names_dict = {i: name for i, name in enumerate(class_names)}
    dataset_content = {
        'names': names_dict,
        'train': train_path,
        'val': val_path,
        'test': test_path,
    }
    yaml_path = os.path.join(task_path, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_content, f, sort_keys=False)
    print(f"Generated {yaml_path}")

def create_dataset_structure(base_path, tasks, field_data_root_dir, stage_files_dir, classes_file):
    """
    Creates the directory structure for the field dataset similar to VOC_inc_15_5,
    and populates it with images and labels based on stage files.
    """
    # Read class names
    class_names = []
    if os.path.exists(classes_file):
        with open(classes_file, 'r') as f:
            class_names = [line.strip() for line in f if line.strip()]
    else:
        print(f"Warning: {classes_file} not found. dataset.yaml will have empty class names.")

    for task_index, task_name in enumerate(tasks):
        task_path = os.path.join(base_path, task_name)
        images_train_path = os.path.join(task_path, 'images', 'train')
        labels_train_path = os.path.join(task_path, 'labels', 'train')
        images_val_path = os.path.join(task_path, 'images', 'val')
        labels_val_path = os.path.join(task_path, 'labels', 'val')
        images_test_path = os.path.join(task_path, 'images', 'test')
        labels_test_path = os.path.join(task_path, 'labels', 'test')

        os.makedirs(images_train_path, exist_ok=True)
        os.makedirs(labels_train_path, exist_ok=True)
        os.makedirs(images_val_path, exist_ok=True)
        os.makedirs(labels_val_path, exist_ok=True)
        os.makedirs(images_test_path, exist_ok=True)
        os.makedirs(labels_test_path, exist_ok=True)
        print(f"Created directory structure for {task_name}")

        # Process stage files
        stage_file_name = f"stage_{task_index + 1}.txt"
        current_stage_file_path = os.path.join(stage_files_dir, stage_file_name)

        if os.path.exists(current_stage_file_path):
            print(f"Processing {stage_file_name}...")
            image_label_data = read_stage_file(current_stage_file_path, field_data_root_dir)

            for original_image_path, original_label_path, data_type in image_label_data:
                if data_type == 'train':
                    dest_image_path = os.path.join(images_train_path, os.path.basename(original_image_path))
                    dest_label_path = os.path.join(labels_train_path, os.path.basename(original_label_path))
                elif data_type == 'val':
                    dest_image_path = os.path.join(images_val_path, os.path.basename(original_image_path))
                    dest_label_path = os.path.join(labels_val_path, os.path.basename(original_label_path))
                else:
                    print(f"Warning: Unknown data type {data_type} for {original_image_path}. Skipping.")
                    continue

                # Copy or link files
                if os.path.exists(original_image_path):
                    shutil.copy(original_image_path, dest_image_path)
                else:
                    print(f"Warning: Image file not found: {original_image_path}")

                if os.path.exists(original_label_path):
                    shutil.copy(original_label_path, dest_label_path)
                else:
                    print(f"Warning: Label file not found: {original_label_path}")
        else:
            print(f"Warning: Stage file not found: {current_stage_file_path}")

        # Generate dataset.yaml
        generate_dataset_yaml(task_path, class_names, 'images/train', 'images/val', 'images/test')

    print("Dataset structure creation and population complete.")

if __name__ == "__main__":
    base_dataset_path = "/hy-tmp/liuzihan/myultralytics/data/field_voc_format" # New output directory
    field_data_root_dir = "/hy-tmp/liuzihan/myultralytics/data/field/data" # Root of the field dataset
    stage_files_dir = "/hy-tmp/liuzihan/myultralytics/data/field/domain_splits"
    classes_file = "/hy-tmp/liuzihan/myultralytics/tools/classes.txt"

    tasks_to_create = [
        "task_1",
        "task_2",
        "task_3",
    ]
    create_dataset_structure(base_dataset_path, tasks_to_create, field_data_root_dir, stage_files_dir, classes_file)