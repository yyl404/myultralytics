import os
import json
import glob
from PIL import Image
from tqdm import tqdm
import random

def convert_yolo_to_json_dataset(img_dir, label_dir, output_json_path, classes=None):
    """
    将YOLO格式(txt)数据集转换为包含额外属性的JSON格式。
    """
    img_files = sorted(glob.glob(os.path.join(img_dir, '*.*')))
    dataset = []

    # 为了模拟"额外属性"，我们预定义一些状态词
    dummy_attributes = [
        {"state": "intact", "visibility": "clear"},
        {"state": "damaged", "visibility": "foggy"},
        {"state": "occluded", "visibility": "raining"},
        {"state": "on_fire", "visibility": "clear"}
    ]

    print(f"Converting {len(img_files)} images from {img_dir}...")

    for img_path in tqdm(img_files):
        filename = os.path.basename(img_path)
        basename = os.path.splitext(filename)[0]
        txt_path = os.path.join(label_dir, basename + '.txt')

        if not os.path.exists(txt_path):
            continue
            
        # 获取图像尺寸
        try:
            with Image.open(img_path) as img:
                width, height = img.size
        except:
            continue

        with open(txt_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            cls_id = int(parts[0])
            # YOLO format: center_x, center_y, w, h (normalized)
            x, y, w, h = map(float, parts[1:])
            
            # 随机赋予一个属性（应付创新点）
            attr = random.choice(dummy_attributes)
            
            cls_name = classes[cls_id] if classes else str(cls_id)

            item = {
                "image_path": os.path.abspath(img_path),
                "name": cls_name,
                "class_id": cls_id,
                "attributes": attr, # 这里就是你的额外分支输入
                "xywh": [x, y, w, h], # 归一化的
                "img_size": [width, height]
            }
            dataset.append(item)

    # 确保输出目录存在
    output_dir = os.path.dirname(output_json_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_json_path, 'w') as f:
        json.dump(dataset, f, indent=4)
    
    print(f"Saved formatted dataset to {output_json_path}")

# 使用示例 (请根据你的实际路径修改)
if __name__ == "__main__":
    # 假设你的数据集只有一类 'ship'
    CLASS_NAMES = [
        'aeroplane',
        'bicycle',
        'bird',
        'boat',
        'bottle',
        'bus',
        'car',
        'cat',
        'chair',
        'cow',
        'diningtable',
        'dog',
        'horse',
        'motorbike',
        'person',
        'pottedplant',
        'sheep',
        'sofa',
        'train',
        'tvmonitor'
    ] 
    
    # 训练集转换
    convert_yolo_to_json_dataset(
        img_dir='/root/myultralytics/data/VOC-YOLO-Small/images/train', 
        label_dir='/root/myultralytics/data/VOC-YOLO-Small/labels/train', 
        output_json_path='/root/myultralytics/data/VOC-YOLO-Small-Json/train_dataset.json',
        classes=CLASS_NAMES
    )
    
    # 验证集转换
    convert_yolo_to_json_dataset(
        img_dir='/root/myultralytics/data/VOC-YOLO-Small/images/val', 
        label_dir='/root/myultralytics/data/VOC-YOLO-Small/labels/val', 
        output_json_path='/root/myultralytics/data/VOC-YOLO-Small-Json/val_dataset.json',
        classes=CLASS_NAMES
    )