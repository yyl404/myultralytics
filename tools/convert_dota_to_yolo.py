import os
from PIL import Image
import numpy as np

class_to_id = {}
classes = []

def convert_dota_to_yolo(dota_ann_dir, img_dir, output_dir):
    """
    Converts DOTA dataset annotations (rotated bounding boxes) to YOLO format (horizontal bounding boxes).

    Args:
        dota_ann_dir (str): Path to the directory containing DOTA annotation files (.txt).
        img_dir (str): Path to the directory containing corresponding image files.
        output_dir (str): Path to the directory where YOLO annotation files will be saved.
                          If it's the same as dota_ann_dir, original files will be overwritten.
    """
    
    os.makedirs(output_dir, exist_ok=True)

    dota_ann_files = [f for f in os.listdir(dota_ann_dir) if f.endswith('.txt')]

    for ann_file in dota_ann_files:
        base_name = os.path.splitext(ann_file)[0]
        img_path = None
        # Try common image extensions
        for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
            potential_img_path = os.path.join(img_dir, base_name + ext)
            if os.path.exists(potential_img_path):
                img_path = potential_img_path
                break
        
        if not img_path:
            print(f"Warning: Image not found for {ann_file} in {img_dir}. Skipping.")
            continue

        try:
            with Image.open(img_path) as img:
                img_width, img_height = img.size
        except Exception as e:
            print(f"Error opening image {img_path}: {e}. Skipping.")
            continue

        yolo_lines = []
        input_ann_path = os.path.join(dota_ann_dir, ann_file)
        with open(input_ann_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                # DOTA format: x1 y1 x2 y2 x3 y3 x4 y4 class_name difficulty
                # We expect at least 9 parts (8 coords + class_name)
                if len(parts) < 9:
                    print(f"Warning: Skipping malformed line in {ann_file}: {line.strip()}")
                    continue

                try:
                    coords = [float(p) for p in parts[:8]]
                    class_name = parts[8]
                except ValueError:
                    print(f"Warning: Skipping malformed coordinates or class name in {ann_file}: {line.strip()}")
                    continue

                # Get class ID
                if class_name not in class_to_id:
                    class_to_id[class_name] = len(classes)
                    classes.append(class_name)
                class_id = class_to_id[class_name]

                # Convert 8 points to horizontal bounding box (xmin, ymin, xmax, ymax)
                xs = coords[0::2]
                ys = coords[1::2]
                
                xmin = min(xs)
                ymin = min(ys)
                xmax = max(xs)
                ymax = max(ys)

                # Convert to YOLO format (normalized x_center, y_center, width, height)
                x_center = (xmin + xmax) / 2.0
                y_center = (ymin + ymax) / 2.0
                width = xmax - xmin
                height = ymax - ymin

                # Normalize coordinates
                x_center_norm = x_center / img_width
                y_center_norm = y_center / img_height
                width_norm = width / img_width
                height_norm = height / img_height

                # Ensure values are within [0, 1] range
                x_center_norm = np.clip(x_center_norm, 0, 1)
                y_center_norm = np.clip(y_center_norm, 0, 1)
                width_norm = np.clip(width_norm, 0, 1)
                height_norm = np.clip(height_norm, 0, 1)

                yolo_lines.append(f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}")
        
        # Write YOLO annotation to file, overwriting original
        output_ann_path = os.path.join(output_dir, ann_file)
        with open(output_ann_path, 'w') as f:
            for yolo_line in yolo_lines:
                f.write(yolo_line + '\n')
        
        print(f"Converted {ann_file} to YOLO format.")

    print("\nConversion complete.")
    print("Detected classes and their IDs:")
    for cls_name, cls_id in class_to_id.items():
        print(f"  {cls_name}: {cls_id}")
    
    # Save the class list to a file for reference
    classes_file_path = os.path.join(output_dir, 'classes.txt')
    with open(classes_file_path, 'w') as f:
        for cls_name in classes:
            f.write(cls_name + '\n')
    print(f"Class list saved to {classes_file_path}")

if __name__ == '__main__':
    dota_ann_dir = '/hy-tmp/liuzihan/myultralytics/data/field/data/train/labelTxt'
    img_dir = '/hy-tmp/liuzihan/myultralytics/data/field/data/train/images/images'
    output_dir = '/hy-tmp/liuzihan/myultralytics/data/field/data/train/labelTxt' # Overwrite original

    convert_dota_to_yolo(dota_ann_dir, img_dir, output_dir)

    dota_ann_dir = '/hy-tmp/liuzihan/myultralytics/data/field/data/val/labelTxt'
    img_dir = '/hy-tmp/liuzihan/myultralytics/data/field/data/val/images/images'
    output_dir = '/hy-tmp/liuzihan/myultralytics/data/field/data/val/labelTxt' # Overwrite original

    convert_dota_to_yolo(dota_ann_dir, img_dir, output_dir)
