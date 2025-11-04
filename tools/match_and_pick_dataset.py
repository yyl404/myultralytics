"""
用于从A数据集当中，找到与B数据集文件名相同的图像和标签，并从A当中挑选出来，构建出新的数据集

python match_and_pick_dataset.py --dataset_a data/VOC/VOC.yaml --dataset_b data/VOC_inc_15_5/task_2_cls_5/dataset.yaml --save_dir data/VOC_inc_15_5/task_2_cls_5_full-labels

所有数据集均为yolo格式

对所有split都会进行处理，并且会复制数据集A当中的yaml文件到save_dir/dataset.yaml
"""
import argparse
import yaml
import os
import shutil
import glob
from tqdm import tqdm


def get_image_files(image_dir, base_dir=None):
    """获取图像目录中的所有图像文件"""
    if not os.path.isabs(image_dir) and base_dir:
        image_dir = os.path.join(base_dir, image_dir)
    
    if not os.path.exists(image_dir):
        return []
    
    image_files = []
    image_files.extend(glob.glob(os.path.join(image_dir, '*.jpg')))
    image_files.extend(glob.glob(os.path.join(image_dir, '*.png')))
    image_files.extend(glob.glob(os.path.join(image_dir, '*.jpeg')))
    image_files.extend(glob.glob(os.path.join(image_dir, '*.JPG')))
    image_files.extend(glob.glob(os.path.join(image_dir, '*.PNG')))
    image_files.extend(glob.glob(os.path.join(image_dir, '*.JPEG')))
    return image_files


def get_label_files(label_dir, base_dir=None):
    """获取标签目录中的所有标签文件"""
    if not os.path.isabs(label_dir) and base_dir:
        label_dir = os.path.join(base_dir, label_dir)
    
    if not os.path.exists(label_dir):
        return []
    
    return glob.glob(os.path.join(label_dir, '*.txt'))


def get_filename_without_ext(filepath):
    """获取不带扩展名的文件名"""
    return os.path.splitext(os.path.basename(filepath))[0]


def main():
    parser = argparse.ArgumentParser(description="Match and pick dataset from A based on B")
    parser.add_argument("--dataset_a", type=str, required=True, help="Path to dataset A's yaml file")
    parser.add_argument("--dataset_b", type=str, required=True, help="Path to dataset B's yaml file")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save the matched dataset")
    args = parser.parse_args()

    # Load yaml files
    dataset_a_yaml_path = args.dataset_a
    dataset_b_yaml_path = args.dataset_b
    
    dataset_a_yaml = yaml.load(open(dataset_a_yaml_path, "r"), Loader=yaml.FullLoader)
    dataset_b_yaml = yaml.load(open(dataset_b_yaml_path, "r"), Loader=yaml.FullLoader)
    
    # Get base directories for resolving relative paths
    dataset_a_base_dir = os.path.dirname(dataset_a_yaml_path)
    dataset_b_base_dir = os.path.dirname(dataset_b_yaml_path)
    
    # Get all splits (train, val, test, etc.)
    splits = []
    for split in ["train", "val", "test"]:
        if split in dataset_a_yaml.keys() or split in dataset_b_yaml.keys():
            splits.append(split)
    
    # Collect all filenames from dataset B (without extension)
    dataset_b_filenames = set()
    for split in splits:
        if split not in dataset_b_yaml:
            continue
        
        if isinstance(dataset_b_yaml[split], str):
            image_dirs = [dataset_b_yaml[split]]
        elif isinstance(dataset_b_yaml[split], list):
            image_dirs = dataset_b_yaml[split]
        else:
            continue
        
        for image_dir in image_dirs:
            image_files = get_image_files(image_dir, dataset_b_base_dir)
            for img_file in image_files:
                filename = get_filename_without_ext(img_file)
                dataset_b_filenames.add(filename)
    
    print(f"Found {len(dataset_b_filenames)} unique filenames in dataset B")
    
    # Create save directory structure
    os.makedirs(args.save_dir, exist_ok=True)
    for split in splits:
        os.makedirs(os.path.join(args.save_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(args.save_dir, "labels", split), exist_ok=True)
    
    # Process each split
    matched_counts = {split: 0 for split in splits}
    
    for split in splits:
        if split not in dataset_a_yaml:
            print(f"Split {split} not found in dataset A, skipping...")
            continue
        
        print(f"\nProcessing {split} split...")
        
        # Get image and label directories from dataset A
        if isinstance(dataset_a_yaml[split], str):
            image_dirs = [dataset_a_yaml[split]]
            label_dirs = [dataset_a_yaml[split].replace("images", "labels")]
        elif isinstance(dataset_a_yaml[split], list):
            image_dirs = dataset_a_yaml[split]
            label_dirs = [img_dir.replace("images", "labels") for img_dir in image_dirs]
        else:
            print(f"Invalid split configuration for {split} in dataset A, skipping...")
            continue
        
        # Collect all image and label files from dataset A
        all_image_files = []
        all_label_files = []
        
        for image_dir, label_dir in zip(image_dirs, label_dirs):
            # Resolve paths
            if not os.path.isabs(image_dir):
                image_dir = os.path.join(dataset_a_base_dir, image_dir)
            if not os.path.isabs(label_dir):
                label_dir = os.path.join(dataset_a_base_dir, label_dir)
            
            image_files = get_image_files(image_dir)
            label_files = get_label_files(label_dir)
            
            all_image_files.extend(image_files)
            all_label_files.extend(label_files)
        
        # Create a mapping from filename (without ext) to file paths
        image_dict = {}
        label_dict = {}
        
        for img_file in all_image_files:
            filename = get_filename_without_ext(img_file)
            image_dict[filename] = img_file
        
        for label_file in all_label_files:
            filename = get_filename_without_ext(label_file)
            label_dict[filename] = label_file
        
        # Match and copy files
        matched_in_split = 0
        for filename in tqdm(dataset_b_filenames, desc=f"Matching {split}"):
            if filename in image_dict and filename in label_dict:
                # Copy image file
                src_image = image_dict[filename]
                # Determine the extension from source file
                _, ext = os.path.splitext(src_image)
                dst_image = os.path.join(args.save_dir, "images", split, f"{filename}{ext}")
                shutil.copy2(src_image, dst_image)
                
                # Copy label file
                src_label = label_dict[filename]
                dst_label = os.path.join(args.save_dir, "labels", split, f"{filename}.txt")
                shutil.copy2(src_label, dst_label)
                
                matched_in_split += 1
        
        matched_counts[split] = matched_in_split
        print(f"  Matched {matched_in_split} files in {split} split")
    
    # Copy dataset A's yaml file to save_dir/dataset.yaml
    dst_yaml_path = os.path.join(args.save_dir, "dataset.yaml")
    
    # Update paths in the yaml to be relative to save_dir
    output_yaml = dataset_a_yaml.copy()
    for split in splits:
        if split in output_yaml:
            output_yaml[split] = f"images/{split}"
    
    with open(dst_yaml_path, 'w') as f:
        yaml.dump(output_yaml, f, default_flow_style=False, allow_unicode=True)
    
    print(f"\nDataset saved to: {args.save_dir}")
    print(f"YAML config saved to: {dst_yaml_path}")
    print("\nSummary:")
    for split in splits:
        print(f"  {split}: {matched_counts[split]} files")


if __name__ == "__main__":
    main()
