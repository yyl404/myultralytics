import argparse
import yaml
import os
import shutil
import glob
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--n_classes", type=int, nargs='+', required=True)
    args = parser.parse_args()

    source_dataset_yaml = yaml.load(open(args.dataset_path, "r"), Loader=yaml.FullLoader)
    source_classes = source_dataset_yaml["names"]
    # 处理每个数据划分
    splits = ['train', 'val', 'test']

    if os.path.exists(args.output_path):
        shutil.rmtree(args.output_path)

    task_classes = []
    classes_id_map = []

    for t, n_classes in enumerate(args.n_classes):
        if os.path.exists(os.path.join(args.output_path, f"task_{t+1}_cls_{n_classes}")):
            shutil.rmtree(os.path.join(args.output_path, f"task_{t+1}_cls_{n_classes}"))
        
        task_classes.append({})
        classes_id_map.append({}) # map from source class id to task class id
        for i in range(n_classes):
            class_name = source_classes[sum(args.n_classes[:t]) + i]
            task_classes[t][i] = class_name
            classes_id_map[t][sum(args.n_classes[:t]) + i] = i

    # 初始化任务统计信息
    task_image_counts = {t: {split: 0 for split in ['train', 'val', 'test']} for t in range(len(args.n_classes))}
    
    # 为每个任务创建目录结构
    for t in range(len(args.n_classes)):
        task_dir = os.path.join(args.output_path, f"task_{t+1}_cls_{args.n_classes[t]}")
    
    # 存储每个任务在每个划分中的图片路径
    task_split_images = {t: {split: [] for split in ['train', 'val', 'test']} for t in range(len(args.n_classes))}
    
    for split_name in splits:
        print(f"Processing {split_name} split...")
        label_dir = f"labels/{split_name}"
        image_dir = f"images/{split_name}"
        
        # 获取该目录下的所有标注文件
        if os.path.exists(os.path.join(os.path.dirname(args.dataset_path), label_dir)):
            label_files = glob.glob(os.path.join(os.path.dirname(args.dataset_path), label_dir, '*.txt'))
            
            for label_file in tqdm(label_files, desc=f"Processing {image_dir} images"):
                # 读取标注文件，统计包含的类别
                classes_in_file = set()
                try:
                    with open(label_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:  # YOLO格式：class_id x_center y_center width height
                                class_id = int(parts[0])
                                classes_in_file.add(class_id)
                except:
                    continue
                
                # 找到与当前文件类别有交集的任务
                compatible_tasks = []
                for t in range(len(args.n_classes)):
                    task_class_ids = set(classes_id_map[t].keys())
                    if classes_in_file.intersection(task_class_ids):
                        compatible_tasks.append(t)
                
                if compatible_tasks:
                    # 计算每个兼容任务的(图片数量/任务类别数)比例
                    task_ratios = {}
                    for t in compatible_tasks:
                        ratio = task_image_counts[t][split_name] / args.n_classes[t] if args.n_classes[t] > 0 else float('inf')
                        task_ratios[t] = ratio
                    
                    # 选择比例最小的任务
                    selected_task = min(task_ratios, key=task_ratios.get)
                    
                    # 获取对应的图片文件路径
                    image_file = os.path.basename(label_file).replace('.txt', '.jpg')
                    if not os.path.exists(os.path.join(os.path.dirname(args.dataset_path), image_dir, image_file)):
                        image_file = os.path.basename(label_file).replace('.txt', '.png')
                    if not os.path.exists(os.path.join(os.path.dirname(args.dataset_path), image_dir, image_file)):
                        image_file = os.path.basename(label_file).replace('.txt', '.jpeg')
                    
                    source_image_path = os.path.join(os.path.dirname(args.dataset_path), image_dir, image_file)
                    
                    if os.path.exists(source_image_path):
                        # 复制图片文件
                        task_dir = os.path.join(args.output_path, f"task_{selected_task+1}_cls_{args.n_classes[selected_task]}")
                        dest_image_path = os.path.join(task_dir, image_dir, image_file)
                        os.makedirs(os.path.dirname(dest_image_path), exist_ok=True)
                        shutil.copy2(source_image_path, dest_image_path)
                        
                        # 复制并转换标注文件
                        dest_label_path = os.path.join(task_dir, label_dir, os.path.basename(label_file))
                        os.makedirs(os.path.dirname(dest_label_path), exist_ok=True)
                        
                        # 转换类别ID并写入新文件
                        with open(label_file, 'r') as src_f, open(dest_label_path, 'w') as dst_f:
                            for line in src_f:
                                parts = line.strip().split()
                                if len(parts) >= 5:
                                    old_class_id = int(parts[0])
                                    if old_class_id in classes_id_map[selected_task].keys():
                                        new_class_id = classes_id_map[selected_task][old_class_id]
                                        parts[0] = str(new_class_id)
                                        dst_f.write(' '.join(parts) + '\n')
                        
                        # 更新统计信息
                        task_image_counts[selected_task][split_name] += 1
    
    # 为每个任务创建YAML配置文件
    for t in range(len(args.n_classes)):
        task_dir = os.path.join(args.output_path, f"task_{t+1}_cls_{args.n_classes[t]}")
        
        # 创建任务配置
        task_config = {
            'train': f"images/train",
            'val': f"images/val",
            'test': f"images/test",
            'names': task_classes[t]
        }
        
        # 保存YAML文件
        yaml_path = os.path.join(task_dir, 'dataset.yaml')
        os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
        with open(yaml_path, 'w') as f:
            yaml.dump(task_config, f, default_flow_style=False)
        
        print(f"Task {t+1} completed: {len(task_config['names'])} classes")
        print(f"  Train: {sum([task_image_counts[t]['train']])} images")
        print(f"  Val: {sum([task_image_counts[t]['val']])} images")
        print(f"  Test: {sum([task_image_counts[t]['test']])} images")  
        print(f"  Config saved to: {yaml_path}")

    # 创建总的yaml文件
    incremental_config = {"tasks": {}}
    for t in range(len(args.n_classes)):
        incremental_config["tasks"][t] = os.path.join(f"task_{t+1}_cls_{args.n_classes[t]}", "dataconfig.yaml")
    
    os.makedirs(args.output_path, exist_ok=True)
    with open(os.path.join(args.output_path, "incremental_config.yaml"), "w") as f:
        yaml.dump(incremental_config, f, default_flow_style=False)