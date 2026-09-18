"""Convert the 13 ODinW domains from COCO json to YOLO images/labels layout.

Each domain is written under ``--out_dir/<domain>/`` with ``images/{train,val,test}``,
``labels/{train,val,test}`` (splits that exist in the source), and ``dataset.yaml``
whose paths are relative to that yaml.

Usage:
    $ python tools/convert_odinw13_to_yolo.py \\
        --src_dir data/OdinW-13 \\
        --out_dir data/OdinW-13-yolo
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from ultralytics.utils import LOGGER, YAML

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Canonical Roboflow export root of each ODinW-13 domain, relative to --src_dir.
DOMAIN_RELATIVE_ROOTS = {
    "AerialMaritimeDrone": "AerialMaritimeDrone/large",
    "Aquarium": "Aquarium/Aquarium Combined.v2-raw-1024.coco",
    "CottontailRabbits": "CottontailRabbits",
    "EgoHands": "EgoHands/generic",
    "NorthAmericaMushrooms": "NorthAmericaMushrooms/North American Mushrooms.v1-416x416.coco",
    "Packages": "Packages/Raw",
    "PascalVOC": "PascalVOC",
    "Raccoon": "Raccoon/Raccoon.v2-raw.coco",
    "ShellfishOpenImages": "ShellfishOpenImages/416x416",
    "VehiclesOpenImages": "VehiclesOpenImages/416x416",
    "pistols": "pistols/export",
    "pothole": "pothole",
    "thermalDogsAndPeople": "thermalDogsAndPeople",
}

SPLIT_DIR_NAMES = {
    "train": ("train",),
    "val": ("val", "valid", "validation"),
    "test": ("test",),
}

COCO_JSON_CANDIDATES = (
    "_annotations.coco.json",
    "annotations.coco.json",
    "annotations_without_background.json",
)


def _find_coco_json(split_dir: Path) -> Path:
    """Return the canonical COCO json in a split directory."""
    for name in COCO_JSON_CANDIDATES:
        candidate = split_dir / name
        if candidate.is_file():
            return candidate
    json_files = sorted(
        path for path in split_dir.glob("*.json") if "fewshot" not in path.name.lower()
    )
    if not json_files:
        raise FileNotFoundError(f"No COCO json found under {split_dir}")
    return json_files[0]


def _discover_split_dirs(domain_root: Path) -> dict[str, Path]:
    """Map YOLO split names to COCO split directories that exist under a domain root."""
    found: dict[str, Path] = {}
    if (domain_root / "_annotations.coco.json").is_file() and not any(
        (domain_root / name).is_dir() for names in SPLIT_DIR_NAMES.values() for name in names
    ):
        found["train"] = domain_root
        return found
    for split, aliases in SPLIT_DIR_NAMES.items():
        for alias in aliases:
            candidate = domain_root / alias
            if candidate.is_dir():
                found[split] = candidate
                break
    if not found:
        raise FileNotFoundError(f"No train/val/test split directories under {domain_root}")
    return found


def _category_id_map(categories: list[dict], used_ids: set[int]) -> tuple[dict[int, int], dict[int, str]]:
    """Map original COCO category ids that appear in annotations onto contiguous YOLO ids."""
    ordered = []
    for category in categories:
        category_id = int(category["id"])
        if category_id in used_ids:
            ordered.append((category_id, str(category["name"])))
    if not ordered:
        raise RuntimeError("No COCO categories appear in the annotations")
    ordered.sort(key=lambda item: item[0])
    id_map = {original: index for index, (original, _) in enumerate(ordered)}
    names = {index: name for index, (_, name) in enumerate(ordered)}
    return id_map, names


def _yolo_line(class_id: int, bbox: list[float], image_width: int, image_height: int) -> str | None:
    """Convert a COCO xywh box to one YOLO label line; None when the box is invalid."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError(f"Invalid image size {image_width}x{image_height}")
    x, y, width, height = bbox
    if width <= 0 or height <= 0:
        return None
    x_center = (x + width / 2.0) / image_width
    y_center = (y + height / 2.0) / image_height
    return (
        f"{class_id} {x_center:.6f} {y_center:.6f} "
        f"{width / image_width:.6f} {height / image_height:.6f}"
    )


def _convert_split(split_dir: Path, output_dir: Path, split: str) -> dict[int, str]:
    """Convert one COCO split into YOLO images/labels and return names for this split."""
    coco_path = _find_coco_json(split_dir)
    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    images = coco.get("images") or []
    annotations = coco.get("annotations") or []
    categories = coco.get("categories") or []
    if not images:
        raise RuntimeError(f"COCO json has no images: {coco_path}")

    used_ids = {int(ann["category_id"]) for ann in annotations if "category_id" in ann}
    id_map, names = _category_id_map(categories, used_ids)
    images_by_id = {int(image["id"]): image for image in images}

    image_dir = output_dir / "images" / split
    label_dir = output_dir / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    boxes_by_image: dict[int, list[str]] = {image_id: [] for image_id in images_by_id}
    skipped_boxes = 0
    for annotation in annotations:
        if annotation.get("iscrowd", 0):
            continue
        category_id = int(annotation["category_id"])
        class_id = id_map.get(category_id)
        if class_id is None:
            skipped_boxes += 1
            continue
        image_id = int(annotation["image_id"])
        image_info = images_by_id.get(image_id)
        if image_info is None:
            skipped_boxes += 1
            continue
        line = _yolo_line(
            class_id,
            annotation["bbox"],
            int(image_info["width"]),
            int(image_info["height"]),
        )
        if line is None:
            skipped_boxes += 1
            continue
        boxes_by_image[image_id].append(line)

    written = 0
    for image_info in images:
        file_name = Path(image_info["file_name"]).name
        source_image = split_dir / file_name
        if not source_image.is_file():
            source_image = split_dir / image_info["file_name"]
        if not source_image.is_file():
            LOGGER.warning(f"Missing image listed in {coco_path}: {image_info['file_name']}")
            continue
        if source_image.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            LOGGER.warning(f"Skipping unsupported image type: {source_image}")
            continue
        destination_image = image_dir / source_image.name
        if not destination_image.exists():
            destination_image.symlink_to(source_image.resolve())
        label_path = label_dir / f"{source_image.stem}.txt"
        label_path.write_text("\n".join(boxes_by_image[int(image_info["id"])]) + ("\n" if boxes_by_image[int(image_info["id"])] else ""), encoding="utf-8")
        written += 1
    if skipped_boxes:
        LOGGER.warning(f"{split_dir}: skipped {skipped_boxes} annotations")
    LOGGER.info(f"{split_dir} -> {split}: {written} images, {len(names)} classes")
    return names


def convert_domain(src_dir: Path, domain: str, relative_root: str, out_dir: Path) -> Path:
    """Convert one ODinW domain and write its dataset.yaml. Returns the yaml path."""
    domain_root = src_dir / relative_root
    if not domain_root.is_dir():
        raise FileNotFoundError(f"Domain root does not exist: {domain_root}")
    domain_out = out_dir / domain
    if domain_out.exists() or domain_out.is_symlink():
        if domain_out.is_symlink() or domain_out.is_file():
            domain_out.unlink()
        else:
            shutil.rmtree(domain_out)
    domain_out.mkdir(parents=True)
    split_dirs = _discover_split_dirs(domain_root)
    names_by_split = {}
    output_config: dict = {}
    for split, split_dir in split_dirs.items():
        names_by_split[split] = _convert_split(split_dir, domain_out, split)
        output_config[split] = f"images/{split}"
    names = names_by_split.get("train") or next(iter(names_by_split.values()))
    output_config["names"] = names
    yaml_path = domain_out / "dataset.yaml"
    YAML().save(yaml_path, output_config)
    return yaml_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src_dir", type=Path, default=Path("data/OdinW-13"))
    parser.add_argument("--out_dir", type=Path, default=Path("data/OdinW-13-yolo"))
    parser.add_argument(
        "--domains",
        nargs="+",
        default=list(DOMAIN_RELATIVE_ROOTS),
        help="Domain names to convert (default: all 13)",
    )
    return parser.parse_args()


def main() -> None:
    """Convert requested ODinW-13 domains to YOLO format."""
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for domain in args.domains:
        relative_root = DOMAIN_RELATIVE_ROOTS.get(domain)
        if relative_root is None:
            raise KeyError(f"Unknown ODinW-13 domain '{domain}'. Known: {list(DOMAIN_RELATIVE_ROOTS)}")
        yaml_path = convert_domain(args.src_dir, domain, relative_root, args.out_dir)
        LOGGER.info(f"Wrote {yaml_path}")


if __name__ == "__main__":
    main()
