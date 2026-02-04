#!/usr/bin/env python3
"""
Convert RSAR dataset to YOLO-OBB format.

RSAR only provides train labels. This script uses the train split exclusively and
splits it into 8:2 (80% train, 20% val) for training and validation.

Input:  data/RSAR/train/{images, annfiles}/
Output: data/RSAR-YOLO/{images/{train,val}, labels/{train,val}, data.yaml}
"""

import os
import random
import shutil
from pathlib import Path
from typing import Optional

import cv2

# Paths
RSAR_ROOT = Path(__file__).resolve().parents[2] / "data" / "RSAR"
OUT_ROOT = Path(__file__).resolve().parents[2] / "data" / "RSAR-YOLO"

# RSAR class names (alphabetical order for consistent IDs)
RSAR_CLASSES = ["aircraft", "bridge", "car", "harbor", "ship", "tank"]
CLASS_TO_ID = {c: i for i, c in enumerate(RSAR_CLASSES)}

# Train/val split ratio: 8:2 (80% train, 20% val)
TRAIN_RATIO = 0.8
RANDOM_SEED = 42


def convert_annotation(ann_path: Path, img_w: int, img_h: int, out_label_path: Path) -> bool:
    """
    Convert RSAR (DOTA-style) annotation to YOLO-OBB format.

    RSAR format: x1 y1 x2 y2 x3 y3 x4 y4 class_name difficulty
    YOLO-OBB:    class_id x1 y1 x2 y2 x3 y3 x4 y4 (normalized 0-1)
    """
    if not ann_path.exists():
        return False

    lines = []
    with open(ann_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            try:
                coords = [float(p) for p in parts[:8]]
                class_name = parts[8]
            except (ValueError, IndexError):
                continue
            if class_name not in CLASS_TO_ID:
                continue
            class_id = CLASS_TO_ID[class_name]
            # Normalize: x by width, y by height
            norm = [
                coords[i] / img_w if i % 2 == 0 else coords[i] / img_h
                for i in range(8)
            ]
            line_str = f"{class_id} " + " ".join(f"{v:.6g}" for v in norm) + "\n"
            lines.append(line_str)

    out_label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_label_path, "w") as f:
        f.writelines(lines)
    return True


def find_image(base_name: str, img_dir: Path) -> Optional[Path]:
    """Find image file by base name (supports jpg, png, bmp)."""
    for ext in (".jpg", ".jpeg", ".png", ".bmp"):
        p = img_dir / (base_name + ext)
        if p.exists():
            return p
    return None


def main():
    random.seed(RANDOM_SEED)

    train_img_dir = RSAR_ROOT / "train" / "images"
    train_ann_dir = RSAR_ROOT / "train" / "annfiles"

    if not train_img_dir.exists() or not train_ann_dir.exists():
        raise FileNotFoundError(
            f"RSAR train dirs not found: {train_img_dir}, {train_ann_dir}"
        )

    # Collect all samples with both image and annotation
    ann_files = list(train_ann_dir.glob("*.txt"))
    samples = []
    for ann_path in ann_files:
        base = ann_path.stem
        img_path = find_image(base, train_img_dir)
        if img_path is not None:
            samples.append((base, img_path, ann_path))

    random.shuffle(samples)
    n = len(samples)
    n_train = int(n * TRAIN_RATIO)
    train_samples = samples[:n_train]
    val_samples = samples[n_train:]

    print(f"Total samples: {n}, train: {len(train_samples)}, val: {len(val_samples)}")

    for split_name, split_samples in [("train", train_samples), ("val", val_samples)]:
        out_img_dir = OUT_ROOT / "images" / split_name
        out_label_dir = OUT_ROOT / "labels" / split_name
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_label_dir.mkdir(parents=True, exist_ok=True)

        for base, img_path, ann_path in split_samples:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Warning: Cannot read {img_path}, skip")
                continue
            h, w = img.shape[:2]

            # Copy image (preserve extension)
            dst_img = out_img_dir / img_path.name
            shutil.copy2(img_path, dst_img)

            # Convert and save label
            out_label = out_label_dir / (base + ".txt")
            convert_annotation(ann_path, w, h, out_label)

    # Write data.yaml
    data_yaml = OUT_ROOT / "data.yaml"
    names_str = "\n".join(f"  {i}: {c}" for i, c in enumerate(RSAR_CLASSES))
    yaml_content = f"""# RSAR dataset in YOLO-OBB format
# Converted from RSAR train only, split 8:2 (train/val)

path: {OUT_ROOT.absolute()}
train: images/train
val: images/val

names:
{names_str}
"""
    with open(data_yaml, "w") as f:
        f.write(yaml_content)

    print(f"Done. Output: {OUT_ROOT}")
    print(f"Config: {data_yaml}")


if __name__ == "__main__":
    main()
