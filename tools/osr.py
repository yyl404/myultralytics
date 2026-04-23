import argparse
import os
import random
import shutil
import subprocess
import types
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from ultralytics import YOLO
from ultralytics.utils import YAML


def _predict_once_custom(self, x, profile=False, visualize=False, embed=None):
    y, dt, embeddings = [], [], []
    for m in self.model:
        if m.f != -1:
            x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
        if profile:
            self._profile_one_layer(m, x, dt)
        x = m(x)
        y.append(x if m.i in self.save else None)
        if embed and m.i in embed:
            embeddings.append(x)
            if m.i == max(embed):
                return embeddings
    return x


def get_directory_size_system(directory_path):
    try:
        result = subprocess.run(
            ["du", "-sb", directory_path],
            capture_output=True,
            text=True,
            check=True,
        )
        return int(result.stdout.split("\t")[0])
    except Exception:
        return -1


def _resolve_root(cfg_path, cfg):
    cfg_dir = os.path.dirname(os.path.abspath(cfg_path))
    root = cfg.get("path", cfg_dir)
    if not os.path.isabs(root):
        root = os.path.join(cfg_dir, root)
    return os.path.abspath(root)


def _resolve_split_dirs(cfg_path, split):
    cfg = YAML.load(cfg_path)
    root = _resolve_root(cfg_path, cfg)

    rel_img_dir = cfg[split]
    img_dir = rel_img_dir if os.path.isabs(rel_img_dir) else os.path.join(root, rel_img_dir)

    if "images" in rel_img_dir:
        rel_label_dir = rel_img_dir.replace("images", "labels")
        label_dir = rel_label_dir if os.path.isabs(rel_label_dir) else os.path.join(root, rel_label_dir)
    else:
        label_dir = os.path.join(root, "labels", split)

    return cfg, root, img_dir, label_dir


def _get_class_names(cfg):
    names = cfg["names"]
    if isinstance(names, dict):
        return [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
    return list(names)


def _get_class_name_to_id(cfg):
    names = _get_class_names(cfg)
    return {name: i for i, name in enumerate(names)}


def _list_images(img_dir):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted([f for f in os.listdir(img_dir) if Path(f).suffix.lower() in exts])


def _read_labels(label_path):
    labels = []
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    labels.append(line.split())
    return labels


def _write_labels(label_path, labels):
    with open(label_path, "w") as f:
        for label in labels:
            f.write(" ".join(map(str, label)) + "\n")


def _crop_from_xywhn(image, box):
    h, w = image.shape[:2]
    cx, cy, bw, bh = box
    x1 = max(0, int((cx - bw / 2) * w))
    y1 = max(0, int((cy - bh / 2) * h))
    x2 = min(w, int((cx + bw / 2) * w))
    y2 = min(h, int((cy + bh / 2) * h))
    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2]


def calculate_iou(box1, box2):
    x1_1, y1_1 = box1[0] - box1[2] / 2, box1[1] - box1[3] / 2
    x2_1, y2_1 = box1[0] + box1[2] / 2, box1[1] + box1[3] / 2

    x1_2, y1_2 = box2[0] - box2[2] / 2, box2[1] - box2[3] / 2
    x2_2, y2_2 = box2[0] + box2[2] / 2, box2[1] + box2[3] / 2

    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0

    inter = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def parse_best_filename(filename):
    # {class_name}_best_sample_{index}.jpg
    stem = filename.rsplit(".", 1)[0]
    parts = stem.split("_best_sample_")
    if len(parts) != 2:
        return None, None
    class_name = parts[0]
    try:
        idx = int(parts[1])
    except ValueError:
        return None, None
    return class_name, idx


def load_memory_bank(memory_bank_dir):
    memory = {}
    if not os.path.exists(memory_bank_dir):
        return memory
    for fname in os.listdir(memory_bank_dir):
        if not fname.endswith(".jpg"):
            continue
        class_name, idx = parse_best_filename(fname)
        if class_name is None:
            continue
        memory.setdefault(class_name, []).append(os.path.join(memory_bank_dir, fname))
    return memory


def paste_patch_on_image(base_image, patch, x, y, pw, ph):
    patch = cv2.resize(patch, (pw, ph))
    h, w = base_image.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w, x + pw)
    y2 = min(h, y + ph)
    if x2 <= x1 or y2 <= y1:
        return base_image
    px1 = max(0, -x)
    py1 = max(0, -y)
    base_image[y1:y2, x1:x2] = patch[py1:py1 + (y2 - y1), px1:px1 + (x2 - x1)]
    return base_image


def generate_memory_bank(dataset_cfg, save_dir, model_path, k=1):
    cfg, _, train_images_dir, train_labels_dir = _resolve_split_dirs(dataset_cfg, "train")
    class_names = _get_class_names(cfg)

    model = YOLO(model_path)
    model.model._predict_once = types.MethodType(_predict_once_custom, model.model)

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    tmp_dir = os.path.join(save_dir, "_all_samples")
    os.makedirs(tmp_dir, exist_ok=True)

    class_feats = {name: [] for name in class_names}
    class_samples = {name: [] for name in class_names}

    image_files = _list_images(train_images_dir)
    for image_file in tqdm(image_files, desc="OSR: build memory candidates"):
        image_path = os.path.join(train_images_dir, image_file)
        label_path = os.path.join(train_labels_dir, Path(image_file).stem + ".txt")

        image = cv2.imread(image_path)
        if image is None or not os.path.exists(label_path):
            continue

        labels = _read_labels(label_path)
        for n, box in enumerate(labels):
            if len(box) < 5:
                continue
            cls = int(float(box[0]))
            if cls < 0 or cls >= len(class_names):
                continue
            crop = _crop_from_xywhn(image, list(map(float, box[1:5])))
            if crop is None or crop.size == 0:
                continue

            class_name = class_names[cls]
            crop_path = os.path.join(tmp_dir, f"{class_name}_sample_{len(class_samples[class_name])}.jpg")
            cv2.imwrite(crop_path, crop)

            embedding = model.embed(crop_path, verbose=False)[-1]
            feat = torch.mean(embedding, dim=(2, 3)).squeeze(0).cpu()
            class_feats[class_name].append(feat)
            class_samples[class_name].append(crop_path)

    for class_name in class_names:
        if len(class_samples[class_name]) == 0:
            continue
        proto = torch.stack(class_feats[class_name], dim=0).mean(dim=0)

        sims = []
        for crop_path, feat in zip(class_samples[class_name], class_feats[class_name]):
            sim = torch.cosine_similarity(feat, proto, dim=0).item()
            sims.append((crop_path, sim))

        sims = sorted(sims, key=lambda x: x[1], reverse=True)[:k]
        for kk, (crop_path, _) in enumerate(sims):
            shutil.copy(crop_path, os.path.join(save_dir, f"{class_name}_best_sample_{kk}.jpg"))

    shutil.rmtree(tmp_dir)

    size_bytes = get_directory_size_system(save_dir)
    if size_bytes >= 0:
        print(f"\033[94mINFO:\033[0m Memory bank occupies {size_bytes / 1024:.2f} KB")


def copy_paste_replay(dataset_cfg, memory_bank_dir, save_dir, split="train", max_paste=3):
    cfg, _, source_images_dir, source_labels_dir = _resolve_split_dirs(dataset_cfg, split)
    class_names = _get_class_names(cfg)
    class_name_to_id = _get_class_name_to_id(cfg)
    memory = load_memory_bank(memory_bank_dir)

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    images_save_dir = os.path.join(save_dir, "images", split)
    labels_save_dir = os.path.join(save_dir, "labels", split)
    os.makedirs(images_save_dir, exist_ok=True)
    os.makedirs(labels_save_dir, exist_ok=True)

    image_files = _list_images(source_images_dir)
    for image_file in tqdm(image_files, desc=f"OSR: copy-paste replay ({split})"):
        image_path = os.path.join(source_images_dir, image_file)
        label_path = os.path.join(source_labels_dir, Path(image_file).stem + ".txt")

        image = cv2.imread(image_path)
        if image is None:
            continue
        h, w = image.shape[:2]

        base_labels = _read_labels(label_path)
        aug_image = image.copy()
        aug_labels = [x[:] for x in base_labels]
        pasted_boxes = []
        remove_ids = set()

        num_paste = random.randint(1, max_paste)
        for _ in range(num_paste):
            avail_classes = [c for c in class_names if c in memory and len(memory[c]) > 0]
            if not avail_classes:
                break

            selected_class = random.choice(avail_classes)
            sample_path = random.choice(memory[selected_class])
            patch = cv2.imread(sample_path)
            if patch is None:
                continue

            ph0, pw0 = patch.shape[:2]
            max_pw = max(16, min(w // 3, pw0))
            max_ph = max(16, min(h // 3, ph0))
            pw = random.randint(max(8, max_pw // 2), max_pw)
            ph = random.randint(max(8, max_ph // 2), max_ph)

            ok = False
            for _try in range(50):
                x = random.randint(0, max(0, w - pw))
                y = random.randint(0, max(0, h - ph))
                box = [(x + pw / 2) / w, (y + ph / 2) / h, pw / w, ph / h]

                overlap_pasted = any(calculate_iou(box, b) > 1e-3 for b in pasted_boxes)
                if overlap_pasted:
                    continue

                for i, label in enumerate(aug_labels):
                    if len(label) >= 5:
                        existing_box = list(map(float, label[1:5]))
                        if calculate_iou(box, existing_box) > 0.5:
                            remove_ids.add(i)

                ok = True
                break

            if not ok:
                continue

            aug_image = paste_patch_on_image(aug_image, patch, x, y, pw, ph)
            pasted_boxes.append(box)
            aug_labels.append([
                str(class_name_to_id[selected_class]),
                f"{box[0]:.6f}",
                f"{box[1]:.6f}",
                f"{box[2]:.6f}",
                f"{box[3]:.6f}",
            ])

        final_labels = [lb for i, lb in enumerate(aug_labels) if i not in remove_ids]

        out_img = os.path.join(images_save_dir, f"cp_{image_file}")
        out_lbl = os.path.join(labels_save_dir, f"cp_{Path(image_file).stem}.txt")
        cv2.imwrite(out_img, aug_image)
        _write_labels(out_lbl, final_labels)

    cfg_out = {
        "names": cfg["names"],
        "train": f"images/{split}",
    }
    YAML.save(data=cfg_out, file=os.path.join(save_dir, "dataset.yaml"))
    print(f"OSR copy-paste replay saved to {save_dir}")


def _collect_new_crops(dataset_cfg, split="train"):
    cfg, _, img_dir, lbl_dir = _resolve_split_dirs(dataset_cfg, split)
    class_names = _get_class_names(cfg)

    crops = []
    for image_file in tqdm(_list_images(img_dir), desc="OSR: collect new crops"):
        image_path = os.path.join(img_dir, image_file)
        label_path = os.path.join(lbl_dir, Path(image_file).stem + ".txt")
        image = cv2.imread(image_path)
        if image is None or not os.path.exists(label_path):
            continue

        labels = _read_labels(label_path)
        for box in labels:
            if len(box) < 5:
                continue
            cls = int(float(box[0]))
            crop = _crop_from_xywhn(image, list(map(float, box[1:5])))
            if crop is None or crop.size == 0:
                continue
            crops.append({
                "class_name": class_names[cls],
                "crop": crop,
            })
    return crops


def feature_augmentation_replay(dataset_cfg, memory_bank_dir, save_dir, split="train", num_generations=0, mixup_alpha=1.0):
    cfg, _, source_images_dir, source_labels_dir = _resolve_split_dirs(dataset_cfg, split)
    class_names = _get_class_names(cfg)
    class_name_to_id = _get_class_name_to_id(cfg)
    memory = load_memory_bank(memory_bank_dir)
    new_crops = _collect_new_crops(dataset_cfg, split)

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    images_save_dir = os.path.join(save_dir, "images", split)
    labels_save_dir = os.path.join(save_dir, "labels", split)
    os.makedirs(images_save_dir, exist_ok=True)
    os.makedirs(labels_save_dir, exist_ok=True)

    bg_files = _list_images(source_images_dir)
    if num_generations <= 0:
        num_generations = len(bg_files)

    if len(bg_files) == 0 or len(new_crops) == 0:
        raise RuntimeError("OSR feature augmentation cannot run: empty background set or empty new-crop set.")

    for idx in tqdm(range(num_generations), desc="OSR: feature augmentation replay"):
        bg_file = random.choice(bg_files)
        bg_path = os.path.join(source_images_dir, bg_file)
        bg_label_path = os.path.join(source_labels_dir, Path(bg_file).stem + ".txt")

        bg = cv2.imread(bg_path)
        if bg is None:
            continue
        h, w = bg.shape[:2]

        labels = _read_labels(bg_label_path)
        aug_image = bg.copy()
        aug_labels = [x[:] for x in labels]
        pasted_boxes = []
        remove_ids = set()

        num_mix = random.randint(1, 4)
        for _ in range(num_mix):
            avail_old = [c for c in class_names if c in memory and len(memory[c]) > 0]
            if not avail_old:
                break

            old_class = random.choice(avail_old)
            old_patch = cv2.imread(random.choice(memory[old_class]))
            new_patch = random.choice(new_crops)["crop"]
            if old_patch is None or new_patch is None:
                continue

            target_w = random.randint(24, max(24, min(w // 3, old_patch.shape[1], new_patch.shape[1])))
            target_h = random.randint(24, max(24, min(h // 3, old_patch.shape[0], new_patch.shape[0])))

            old_rs = cv2.resize(old_patch, (target_w, target_h))
            new_rs = cv2.resize(new_patch, (target_w, target_h))
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            mixed_patch = cv2.addWeighted(old_rs, lam, new_rs, 1 - lam, 0)

            ok = False
            for _try in range(50):
                x = random.randint(0, max(0, w - target_w))
                y = random.randint(0, max(0, h - target_h))
                box = [(x + target_w / 2) / w, (y + target_h / 2) / h, target_w / w, target_h / h]

                overlap_pasted = any(calculate_iou(box, b) > 1e-3 for b in pasted_boxes)
                if overlap_pasted:
                    continue

                for i, label in enumerate(aug_labels):
                    if len(label) >= 5:
                        existing_box = list(map(float, label[1:5]))
                        if calculate_iou(box, existing_box) > 0.5:
                            remove_ids.add(i)

                ok = True
                break

            if not ok:
                continue

            aug_image = paste_patch_on_image(aug_image, mixed_patch, x, y, target_w, target_h)
            pasted_boxes.append(box)

            # mixed patch uses old-class supervision
            aug_labels.append([
                str(class_name_to_id[old_class]),
                f"{box[0]:.6f}",
                f"{box[1]:.6f}",
                f"{box[2]:.6f}",
                f"{box[3]:.6f}",
            ])

        final_labels = [lb for i, lb in enumerate(aug_labels) if i not in remove_ids]

        out_img = os.path.join(images_save_dir, f"fa_{idx:06d}.jpg")
        out_lbl = os.path.join(labels_save_dir, f"fa_{idx:06d}.txt")
        cv2.imwrite(out_img, aug_image)
        _write_labels(out_lbl, final_labels)

    cfg_out = {
        "names": cfg["names"],
        "train": f"images/{split}",
    }
    YAML.save(data=cfg_out, file=os.path.join(save_dir, "dataset.yaml"))
    print(f"OSR feature augmentation replay saved to {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_dataset_cfg", type=str, default="", help="Task1/base dataset yaml")
    parser.add_argument("--new_dataset_cfg", type=str, default="", help="Task2/converted dataset yaml")
    parser.add_argument("--memory_bank_dir", type=str, default="", help="Memory bank dir")
    parser.add_argument("--save_dir", type=str, default="", help="Save dir")
    parser.add_argument("--split", type=str, default="train", help="train/val/test")
    parser.add_argument("--num_generations", type=int, default=0, help="0 means auto = len(train)")
    parser.add_argument("--model_path", type=str, default="", help="Task1 model path")
    parser.add_argument("--k", type=int, default=1, help="One-shot => keep 1 sample per old class")
    parser.add_argument("--mixup_alpha", type=float, default=1.0, help="Beta(alpha, alpha) for feature augmentation")

    parser.add_argument("--generate_memory_bank", action="store_true")
    parser.add_argument("--copy_paste_replay", action="store_true")
    parser.add_argument("--feature_augmentation_replay", action="store_true")
    args = parser.parse_args()

    if args.generate_memory_bank:
        generate_memory_bank(args.base_dataset_cfg, args.memory_bank_dir, args.model_path, args.k)

    if args.copy_paste_replay:
        copy_paste_replay(args.new_dataset_cfg, args.memory_bank_dir, args.save_dir, args.split)

    if args.feature_augmentation_replay:
        feature_augmentation_replay(
            args.new_dataset_cfg,
            args.memory_bank_dir,
            args.save_dir,
            args.split,
            args.num_generations,
            args.mixup_alpha,
        )