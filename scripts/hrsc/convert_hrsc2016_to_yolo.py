#!/usr/bin/env python3
"""
Convert HRSC2016 dataset to YOLO format with proper class names from sysdata.xml
"""
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict

def parse_xml(xml_path):
    """Parse HRSC2016 XML annotation file"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Get image dimensions
    width = int(root.find('Img_SizeWidth').text)
    height = int(root.find('Img_SizeHeight').text)
    
    # Get all objects
    objects = []
    for obj in root.findall('.//HRSC_Object'):
        class_id = obj.find('Class_ID').text
        xmin = float(obj.find('box_xmin').text)
        ymin = float(obj.find('box_ymin').text)
        xmax = float(obj.find('box_xmax').text)
        ymax = float(obj.find('box_ymax').text)
        difficult = int(obj.find('difficult').text) if obj.find('difficult') is not None else 0
        truncated = int(obj.find('truncated').text) if obj.find('truncated') is not None else 0
        
        objects.append({
            'class_id': class_id,
            'xmin': xmin,
            'ymin': ymin,
            'xmax': xmax,
            'ymax': ymax,
            'difficult': difficult,
            'truncated': truncated
        })
    
    return width, height, objects

def convert_to_yolo_format(xmin, ymin, xmax, ymax, img_width, img_height):
    """Convert bounding box from (xmin, ymin, xmax, ymax) to YOLO format (center_x, center_y, width, height)"""
    # Calculate center coordinates
    center_x = (xmin + xmax) / 2.0 / img_width
    center_y = (ymin + ymax) / 2.0 / img_height
    
    # Calculate width and height (normalized)
    width = (xmax - xmin) / img_width
    height = (ymax - ymin) / img_height
    
    # Ensure values are in [0, 1] range
    center_x = max(0, min(1, center_x))
    center_y = max(0, min(1, center_y))
    width = max(0, min(1, width))
    height = max(0, min(1, height))
    
    return center_x, center_y, width, height

def collect_all_classes(annotations_dir):
    """Collect all unique class IDs from annotations"""
    classes = set()
    for xml_file in Path(annotations_dir).rglob('*.xml'):
        try:
            tree = ET.parse(xml_file)
            for obj in tree.getroot().findall('.//HRSC_Object'):
                class_id = obj.find('Class_ID')
                if class_id is not None:
                    classes.add(class_id.text)
        except Exception as e:
            print(f"Error parsing {xml_file}: {e}")
    return sorted(classes)

def load_class_names_from_sysdata(sysdata_path):
    """Load class name mappings from sysdata.xml"""
    tree = ET.parse(sysdata_path)
    root = tree.getroot()
    
    class_mapping = {}
    for class_elem in root.findall('.//HRSC_Class'):
        class_id = class_elem.find('Class_ID').text
        short_name = class_elem.find('Class_ShortName')
        eng_name = class_elem.find('Class_EngName')
        
        # Use short name if available, otherwise use English name
        if short_name is not None and short_name.text:
            class_mapping[class_id] = short_name.text.strip()
        elif eng_name is not None and eng_name.text:
            class_mapping[class_id] = eng_name.text.strip()
        else:
            class_mapping[class_id] = f"class_{class_id}"
    
    return class_mapping

def needs_quotes(name):
    """Check if a name needs quotes in YAML"""
    special_chars = ['|', '[', ']', '(', ')', ':', '&', '*', '#', '?', '-', '<', '>', '=', '!', '%', '@', '`']
    return any(char in name for char in special_chars) or ' ' in name

def main():
    # Paths
    hrsc_root = Path('/root/myultralytics/data/HRSC2016_dataset/HRSC2016')
    output_root = Path('/root/myultralytics/data/HRSC2016_dataset/HRSC2016-YOLO')
    sysdata_path = hrsc_root / 'Train' / 'sysdata.xml'
    
    # Load class names from sysdata.xml
    print("Loading class names from sysdata.xml...")
    class_name_mapping = load_class_names_from_sysdata(sysdata_path)
    print(f"Loaded {len(class_name_mapping)} class definitions")
    
    # Remove existing output directory if it exists
    if output_root.exists():
        print(f"\nRemoving existing output directory: {output_root}")
        shutil.rmtree(output_root)
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        (output_root / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_root / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # Collect all classes from all splits
    print("\nCollecting all classes from annotations...")
    all_classes = set()
    for split_dir in ['Train', 'Test']:
        annotations_dir = hrsc_root / split_dir / 'Annotations'
        if annotations_dir.exists():
            classes = collect_all_classes(annotations_dir)
            all_classes.update(classes)
    
    # Create class mapping (sorted for consistency)
    class_list = sorted(all_classes)
    class_to_id = {cls: idx for idx, cls in enumerate(class_list)}
    id_to_class = {idx: cls for idx, cls in enumerate(class_list)}
    
    print(f"Found {len(class_list)} classes: {class_list}")
    
    # Process each split
    splits = {
        'train': ('Train', 'train.txt'),
        'val': ('Train', 'val.txt'),  # val is also in Train directory
        'test': ('Test', 'test.txt')
    }
    
    stats = defaultdict(int)
    
    for split_name, (data_dir, list_file) in splits.items():
        print(f"\nProcessing {split_name} split...")
        
        # Read image list
        list_path = hrsc_root / 'ImageSets' / list_file
        if not list_path.exists():
            print(f"Warning: {list_path} not found, skipping {split_name}")
            continue
        
        with open(list_path, 'r') as f:
            image_ids = [line.strip() for line in f if line.strip()]
        
        images_dir = hrsc_root / data_dir / 'AllImages'
        annotations_dir = hrsc_root / data_dir / 'Annotations'
        
        processed = 0
        skipped = 0
        
        for image_id in image_ids:
            # Find image file (could be .bmp)
            image_file = None
            for ext in ['.bmp', '.jpg', '.png']:
                candidate = images_dir / f"{image_id}{ext}"
                if candidate.exists():
                    image_file = candidate
                    break
            
            if image_file is None:
                print(f"Warning: Image {image_id} not found in {images_dir}")
                skipped += 1
                continue
            
            # Find annotation file
            xml_file = annotations_dir / f"{image_id}.xml"
            if not xml_file.exists():
                print(f"Warning: Annotation {image_id}.xml not found")
                skipped += 1
                continue
            
            try:
                # Parse XML
                img_width, img_height, objects = parse_xml(xml_file)
                
                if len(objects) == 0:
                    print(f"Warning: No objects found in {xml_file}")
                    skipped += 1
                    continue
                
                # Copy image
                output_image_dir = output_root / 'images' / split_name
                # Convert to jpg for consistency (or keep original format)
                output_image = output_image_dir / f"{image_id}.jpg"
                if image_file.suffix.lower() == '.bmp':
                    # Use PIL to convert BMP to JPG
                    from PIL import Image
                    img = Image.open(image_file)
                    img.convert('RGB').save(output_image, 'JPEG')
                else:
                    shutil.copy2(image_file, output_image)
                
                # Create YOLO format label file
                output_label = output_root / 'labels' / split_name / f"{image_id}.txt"
                with open(output_label, 'w') as f:
                    for obj in objects:
                        class_id_str = obj['class_id']
                        yolo_class_id = class_to_id[class_id_str]
                        
                        xmin, ymin = obj['xmin'], obj['ymin']
                        xmax, ymax = obj['xmax'], obj['ymax']
                        
                        center_x, center_y, width, height = convert_to_yolo_format(
                            xmin, ymin, xmax, ymax, img_width, img_height
                        )
                        
                        # Write YOLO format: class_id center_x center_y width height
                        f.write(f"{yolo_class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
                        stats[f"class_{yolo_class_id}"] += 1
                
                processed += 1
                stats[f"{split_name}_images"] += 1
                stats[f"{split_name}_objects"] += len(objects)
                
            except Exception as e:
                print(f"Error processing {image_id}: {e}")
                skipped += 1
                continue
        
        print(f"{split_name}: Processed {processed} images, skipped {skipped} images")
    
    # Create YAML configuration file with proper class names
    print("\nCreating YAML configuration file...")
    yaml_file = output_root / 'HRSC2016.yaml'
    with open(yaml_file, 'w', encoding='utf-8') as f:
        f.write("# HRSC2016 dataset in YOLO format\n\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/test\n\n")
        f.write("# Number of classes\n")
        f.write(f"nc: {len(class_list)}\n\n")
        f.write("# Class names\n")
        f.write("names:\n")
        
        for idx, class_id in enumerate(class_list):
            if class_id in class_name_mapping:
                name = class_name_mapping[class_id]
                if needs_quotes(name):
                    f.write(f"  {idx}: \"{name}\"\n")
                else:
                    f.write(f"  {idx}: {name}\n")
            else:
                f.write(f"  {idx}: class_{class_id}\n")
    
    print(f"\nConversion complete!")
    print(f"Output directory: {output_root}")
    print(f"Configuration file: {yaml_file}")
    print(f"\nStatistics:")
    for key, value in sorted(stats.items()):
        print(f"  {key}: {value}")
    
    # Print class name mapping summary
    print(f"\nClass name mapping (first 10):")
    for idx in range(min(10, len(class_list))):
        class_id = class_list[idx]
        yolo_idx = class_to_id[class_id]
        name = class_name_mapping.get(class_id, f"class_{class_id}")
        print(f"  {yolo_idx}: {class_id} -> {name}")

if __name__ == '__main__':
    main()

