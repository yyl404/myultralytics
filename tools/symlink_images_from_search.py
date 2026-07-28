#!/usr/bin/env python3
"""Link images into a label-only YOLO tree from one or more search roots.

Given a reference dataset that keeps labels/yaml but may lack image files
(e.g. /hy-tmp/data/COCO_40+40), find matching images under search directories by
basename and create one symlink per label at the corresponding images/ path:

    <ref>/.../labels/<split>/<stem>.txt
  -> <ref>/.../images/<split>/<stem>.<ext>  ->  <found image>

Example:
  python tools/symlink_images_from_search.py \\
    --ref-dir /hy-tmp/data/COCO_40+40 \\
    --search-dirs /hy-tmp/data/coco-yolo/images \\
                  /root/myultralytics/data/coco-yolo/images \\
    --workers 16
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path

from tqdm import tqdm

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
LABEL_EXT = ".txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create image symlinks in a label-only YOLO reference tree."
    )
    parser.add_argument(
        "--ref-dir",
        type=Path,
        required=True,
        help="Reference dataset root with labels/ (and empty or partial images/).",
    )
    parser.add_argument(
        "--search-dirs",
        type=Path,
        nargs="+",
        required=True,
        help="Directories (and their subtrees) to search for image files.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(16, (os.cpu_count() or 1) * 2),
        help="Number of worker threads for indexing and linking (default: %(default)s).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing files/symlinks at the destination path.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned links without creating them.",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    return args


def _collect_images_under(root: Path) -> list[tuple[str, Path]]:
    """Walk one search root; return (stem, absolute_path) pairs."""
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Search directory not found: {root}")

    found: list[tuple[str, Path]] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            suffix = Path(name).suffix.lower()
            if suffix not in IMAGE_EXTS:
                continue
            path = Path(dirpath, name)
            # Prefer lstat-fast path; resolve only once for the symlink target.
            found.append((path.stem, path.resolve()))
    return found


def index_search_images(
    search_dirs: list[Path], workers: int
) -> dict[str, list[Path]]:
    """Map image stem -> list of absolute paths found under search dirs."""
    index: dict[str, list[Path]] = defaultdict(list)
    seen: set[Path] = set()

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="index") as pool:
        futures = [pool.submit(_collect_images_under, root) for root in search_dirs]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Indexing"):
            for stem, resolved in future.result():
                if resolved in seen:
                    continue
                seen.add(resolved)
                index[stem].append(resolved)
    return index


def iter_label_files(ref_dir: Path) -> list[Path]:
    labels: list[Path] = []
    for path in ref_dir.rglob(f"*{LABEL_EXT}"):
        if not path.is_file():
            continue
        if "labels" not in path.parts:
            continue
        labels.append(path)
    return labels


def label_to_image_dir(label_path: Path) -> Path:
    """Map .../labels/<split>/<stem>.txt -> .../images/<split>/."""
    parts = list(label_path.parts)
    try:
        idx = parts.index("labels")
    except ValueError as exc:
        raise ValueError(f"Label path has no 'labels' component: {label_path}") from exc
    parts[idx] = "images"
    return Path(*parts[:-1])


def create_symlink(target: Path, link_path: Path, *, force: bool, dry_run: bool) -> str:
    """Create link_path -> target. Returns status: created|exists|replaced|skipped."""
    if link_path.exists() or link_path.is_symlink():
        if link_path.is_symlink() and link_path.resolve() == target.resolve():
            return "exists"
        if not force:
            return "skipped"
        if dry_run:
            return "replaced"
        if link_path.is_symlink() or link_path.is_file():
            link_path.unlink()
        else:
            raise IsADirectoryError(f"Destination is a directory: {link_path}")
        status = "replaced"
    else:
        status = "created"

    if dry_run:
        return status

    link_path.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(target, link_path)
    return status


def _link_one(
    label_path: Path,
    index: dict[str, list[Path]],
    *,
    force: bool,
    dry_run: bool,
) -> dict:
    """Process one label. Returns a small result dict (thread-safe to return)."""
    stem = label_path.stem
    candidates = index.get(stem, [])
    if not candidates:
        return {"status": "missing", "label": label_path, "candidates": []}

    ambiguous = len(candidates) > 1
    target = candidates[0]
    link_path = label_to_image_dir(label_path) / f"{stem}{target.suffix.lower()}"
    try:
        status = create_symlink(target, link_path, force=force, dry_run=dry_run)
    except OSError as exc:
        return {
            "status": "failed",
            "label": label_path,
            "link": link_path,
            "target": target,
            "error": str(exc),
            "ambiguous": ambiguous,
            "candidates": candidates if ambiguous else [],
        }
    return {
        "status": status,
        "label": label_path,
        "link": link_path,
        "target": target,
        "ambiguous": ambiguous,
        "candidates": candidates if ambiguous else [],
    }


def _print_summary(
    *,
    total_labels: int,
    indexed_images: int,
    unique_stems: int,
    counts: dict[str, int],
    missing: list[Path],
    ambiguous: list[tuple[Path, list[Path]]],
    failures: list[dict],
    dry_run: bool,
) -> None:
    found = total_labels - counts["missing"]
    print("\n========== Summary ==========")
    print(f"  Search index : {indexed_images} image files ({unique_stems} unique stems)")
    print(f"  Labels total : {total_labels}")
    print(f"  Found        : {found}")
    print(f"  Not found    : {counts['missing']}")
    print("  ---- link actions ----")
    for key in ("created", "replaced", "exists", "skipped", "failed"):
        print(f"  {key:12}: {counts[key]}")
    if counts["ambiguous"]:
        print(f"  ambiguous   : {counts['ambiguous']} (first match used)")
    print("=============================")

    if failures:
        print(f"\nFailed links, showing up to 10/{len(failures)}:")
        for item in failures[:10]:
            print(f"  {item['link']} -> {item['target']}: {item['error']}")

    if ambiguous:
        print(f"\nAmbiguous stems (used first match), showing up to 10/{len(ambiguous)}:")
        for label_path, cands in ambiguous[:10]:
            print(f"  {label_path.stem}:")
            for c in cands:
                print(f"    - {c}")

    if missing:
        print(f"\nMissing images for {len(missing)} labels, showing up to 20:")
        for path in missing[:20]:
            print(f"  {path}")

    if dry_run:
        print("\nDry-run only; no symlinks were written.")


def main() -> int:
    args = parse_args()
    ref_dir = args.ref_dir.resolve()
    if not ref_dir.is_dir():
        print(f"Reference directory not found: {ref_dir}", file=sys.stderr)
        return 1

    print(f"Indexing images under {len(args.search_dirs)} search dir(s) "
          f"with {args.workers} workers...")
    index = index_search_images(args.search_dirs, workers=args.workers)
    indexed_images = sum(len(v) for v in index.values())
    print(f"Indexed {indexed_images} images ({len(index)} unique stems).")

    labels = iter_label_files(ref_dir)
    print(f"Found {len(labels)} label files under {ref_dir}")
    print(f"Linking with {args.workers} workers...")

    counts: dict[str, int] = defaultdict(int)
    missing: list[Path] = []
    ambiguous: list[tuple[Path, list[Path]]] = []
    failures: list[dict] = []

    worker = partial(_link_one, index=index, force=args.force, dry_run=args.dry_run)
    # chunksize keeps the future queue bounded for large label sets
    chunksize = max(32, len(labels) // (args.workers * 32) or 1)
    with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="link") as pool:
        for result in tqdm(
            pool.map(worker, labels, chunksize=chunksize),
            total=len(labels),
            desc="Linking",
        ):
            status = result["status"]
            counts[status] += 1
            if status == "missing":
                missing.append(result["label"])
            elif status == "failed":
                failures.append(result)
            if result.get("ambiguous"):
                counts["ambiguous"] += 1
                ambiguous.append((result["label"], result["candidates"]))

    _print_summary(
        total_labels=len(labels),
        indexed_images=indexed_images,
        unique_stems=len(index),
        counts=counts,
        missing=missing,
        ambiguous=ambiguous,
        failures=failures,
        dry_run=args.dry_run,
    )

    return 1 if counts["missing"] or counts["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
