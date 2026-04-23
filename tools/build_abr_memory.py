# tools/build_abr_memory.py
import argparse, json, random, os
from pathlib import Path
import cv2
from ultralytics.utils import YAML

VOC_NAMES = [
    "aeroplane","bicycle","bird","boat","bottle",
    "bus","car","cat","chair","cow",
    "diningtable","dog","horse","motorbike","person",
    "pottedplant","sheep","sofa","train","tvmonitor"
]

def load_dataset_yaml(yaml_path):
    yaml_path = Path(yaml_path).resolve()
    cfg = YAML.load(str(yaml_path))
    root = yaml_path.parent

    def resolve_path(x):
        if not x:
            return x
        p = Path(x)
        return str(p if p.is_absolute() else (root / p).resolve())

    cfg["train"] = resolve_path(cfg.get("train"))
    if "val" in cfg:
        cfg["val"] = resolve_path(cfg.get("val"))
    if "test" in cfg:
        cfg["test"] = resolve_path(cfg.get("test"))

    names = cfg["names"]
    if isinstance(names, dict):
        names = [names[i] for i in range(len(names))]

    return cfg, cfg["train"], names

def collect_images(train_field):
    # 兼容 txt 列表 或 images 目录
    p = Path(train_field)
    if p.suffix == ".txt":
        with open(p) as f:
            return [x.strip() for x in f if x.strip()]
    exts = {".jpg",".jpeg",".png",".bmp"}
    return [str(x) for x in p.rglob("*") if x.suffix.lower() in exts]

def img2label_path(img_path):
    p = Path(img_path)
    # 常见 YOLO 结构: images/... -> labels/...
    return str(Path(str(p).replace("/images/","/labels/")).with_suffix(".txt"))

def crop_xywhn(img, xc, yc, w, h):
    H, W = img.shape[:2]
    x1 = max(0, int((xc - w/2) * W))
    y1 = max(0, int((yc - h/2) * H))
    x2 = min(W, int((xc + w/2) * W))
    y2 = min(H, int((yc + h/2) * H))
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]

def main(args):
    random.seed(args.seed)
    out_dir = Path(args.output)
    crop_dir = out_dir / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)

    cfg, train_field, names = load_dataset_yaml(args.data)
    img_files = collect_images(train_field)

    cls_buckets = {}
    for cls_id in range(len(names)):
        cls_buckets[cls_id] = []

    for img_path in img_files:
        label_path = img2label_path(img_path)
        if not os.path.exists(label_path):
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue

        with open(label_path) as f:
            lines = [x.strip().split() for x in f if x.strip()]

        for li, row in enumerate(lines):
            cls_id = int(float(row[0]))
            xc, yc, w, h = map(float, row[1:5])
            crop = crop_xywhn(img, xc, yc, w, h)
            if crop is None or crop.size == 0:
                continue
            cls_buckets[cls_id].append((img_path, label_path, crop, li))

    old_num = args.old_class_num
    per_cls = (args.memory_size + old_num - 1) // old_num
    selected = []

    # 只从旧类里取
    for cls_id in range(old_num):
        items = cls_buckets[cls_id]
        random.shuffle(items)
        items = items[:per_cls]
        for k, (img_path, label_path, crop, li) in enumerate(items):
            crop_name = f"{len(selected):06d}_cls_{cls_id}.jpg"
            crop_path = crop_dir / crop_name
            cv2.imwrite(str(crop_path), crop)
            selected.append({
                "crop_path": str(crop_path),
                "cls": cls_id,
                "src_img": img_path,
                "src_label": label_path
            })

    with open(out_dir / "memory.json", "w") as f:
        json.dump(selected, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--memory_size", type=int, default=2000)
    parser.add_argument("--old_class_num", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args)