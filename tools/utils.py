"""Utility functions for tools."""

import ast
import os
import os.path as OSP
import threading
import time
import shutil
from collections.abc import Mapping

from torch import nn
import torch.nn.functional as F

import psutil
import torch


def parse_list_string(list_str):
    """Parse a Python list string into a list with arbitrary elements, length, and nesting.
    
    This function safely parses a string representation of a Python list into an actual
    Python list structure. It supports lists of any length, with any element types, and
    arbitrary levels of nesting.
    
    Args:
        list_str (str): String representation of a Python list, e.g., 
            "['class1', 'class2']" or "[['person', 'car'], ['bus', 'truck']]"
    
    Returns:
        list: A Python list with the parsed structure. The list can contain any elements
            and have arbitrary nesting levels.
            Examples:
            - parse_list_string("['class1', 'class2']") -> ['class1', 'class2']
            - parse_list_string("[['person', 'car'], ['bus', 'truck']]") -> [['person', 'car'], ['bus', 'truck']]
    
    Raises:
        ValueError: If the input is not a string.
        ValueError: If the parsed result is not a list.
        SyntaxError: If the string cannot be parsed as a valid Python literal.
    
    Example:
        >>> parse_list_string("['class1', 'class2', 'class3']")
        ['class1', 'class2', 'class3']
        >>> parse_list_string("[['person', 'car'], ['bus', 'truck']]")
        [['person', 'car'], ['bus', 'truck']]
        >>> parse_list_string("[[[1, 2], [3, 4]], [[5, 6]]]")
        [[[1, 2], [3, 4]], [[5, 6]]]
    """
    if not isinstance(list_str, str):
        raise ValueError(f"Expected string, got {type(list_str).__name__}")
    
    result = ast.literal_eval(list_str)
    if not isinstance(result, list):
        raise ValueError(f"Parsed result must be a list, got {type(result).__name__}")
    
    return result


# ====== Dataset Utils ======
def normalize_names(names, source):
    """Return class names keyed by contiguous integer IDs.

    Args:
        names: Class names as a list (index = class id) or a mapping of class id -> name.
        source: Description of where the names come from, used in error messages
            (e.g. "model 'best.pt'" or "dataset 'dataset.yaml'").

    Returns:
        dict: {int class_id: class name}, validated to be contiguous from 0.
    """
    if isinstance(names, list):
        return dict(enumerate(names))
    if not isinstance(names, Mapping):
        raise TypeError(f"Class names in {source} must be a list or mapping, got {type(names)}")
    normalized = {int(class_id): class_name for class_id, class_name in names.items()}
    expected_ids = list(range(len(normalized)))
    if sorted(normalized) != expected_ids:
        raise ValueError(
            f"Class IDs in {source} must be contiguous from 0, got {sorted(normalized)}"
        )
    return normalized


def calculate_iou_xywh(box1, box2):
    """Calculate IoU between two xywh format boxes.
    
    Args:
        box1 (list): [x_center, y_center, width, height]
        box2 (list): [x_center, y_center, width, height]
    
    Returns:
        float: IoU value between 0 and 1
    """
    x1_center, y1_center, w1, h1 = box1
    x2_center, y2_center, w2, h2 = box2
    
    # Convert to corner coordinates
    x1_min = x1_center - w1 / 2
    x1_max = x1_center + w1 / 2
    y1_min = y1_center - h1 / 2
    y1_max = y1_center + h1 / 2
    
    x2_min = x2_center - w2 / 2
    x2_max = x2_center + w2 / 2
    y2_min = y2_center - h2 / 2
    y2_max = y2_center + h2 / 2
    
    # Calculate intersection
    inter_x_min = max(x1_min, x2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_min = max(y1_min, y2_min)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # Calculate union
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / (union_area+1e-6)


def parse_label_line(line):
    """Parse a YOLO detection format label line.

    Args:
        line (str): Label line in YOLO format: "class_id x_center y_center width height"

    Returns:
        tuple: (class_id, x_center, y_center, width, height) or None if invalid
    """
    parts = line.strip().split()
    if len(parts) != 5:
        return None
    try:
        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
        return (class_id, x_center, y_center, width, height)
    except (ValueError, IndexError):
        return None


def parse_label_line_obb(line):
    """Parse a YOLO OBB format label line (oriented bounding box).

    OBB format: class_id x1 y1 x2 y2 x3 y3 x4 y4 (normalized polygon corners).

    Args:
        line (str): Label line in YOLO OBB format.

    Returns:
        tuple: (class_id, [x1, y1, x2, y2, x3, y3, x4, y4]) or None if invalid
    """
    parts = line.strip().split()
    if len(parts) != 9:
        return None
    try:
        class_id = int(parts[0])
        coords = [float(parts[i]) for i in range(1, 9)]
        return (class_id, coords)
    except (ValueError, IndexError):
        return None


def convert_class_ids(label_lines, class_id_map, task="detect"):
    """Convert the class IDs in the label lines by the class_id_map.

    Args:
        label_lines: List of label file lines.
        class_id_map: Dict mapping old class id -> new class id.
        task: "detect" for xywh format, "obb" for xyxyxyxy (4 corner) format.
    """
    converted_lines = []
    for line in label_lines:
        s = line.strip()
        if not s:
            continue
        if task == "obb":
            parsed = parse_label_line_obb(s)
            if parsed is None:
                continue
            old_cat_id, coords = parsed
            if old_cat_id in class_id_map:
                new_cat_id = class_id_map[old_cat_id]
                converted_lines.append(f"{new_cat_id} " + " ".join(str(c) for c in coords) + "\n")
        else:
            parsed = parse_label_line(s)
            if parsed is None:
                continue
            old_cat_id, x, y, w, h = parsed
            if old_cat_id in class_id_map:
                new_cat_id = class_id_map[old_cat_id]
                converted_lines.append(f"{new_cat_id} {x} {y} {w} {h}\n")
    return converted_lines


def convert_class_ids_from_dir(labels_dir, class_id_map, output_dir, task="detect", pbar=None):
    """Read all .txt label files under labels_dir (recursively) and convert class IDs.

    Nested layout under ``labels_dir`` is mirrored under ``output_dir`` (same relative paths).

    Args:
        labels_dir: Root directory of YOLO-format ``*.txt`` labels (flat or nested).
        class_id_map: Dict mapping old class id -> new class id.
        output_dir: Root directory to write converted labels (subdirs created as needed).
        task: "detect" for detection (xywh) labels, "obb" for OBB (xyxyxyxy) labels.
        pbar: Optional progress bar with an ``update()`` method; incremented once per label file.
    """
    labels_dir = OSP.abspath(labels_dir)
    output_dir = OSP.abspath(output_dir)
    for root, _, files in os.walk(labels_dir):
        rel_root = OSP.relpath(root, labels_dir)
        out_sub = output_dir if rel_root == "." else OSP.join(output_dir, rel_root)
        for label_file in files:
            if not label_file.endswith(".txt"):
                continue
            label_path = OSP.join(root, label_file)
            os.makedirs(out_sub, exist_ok=True)
            with open(label_path, "r") as f:
                lines = f.readlines()
            converted_lines = convert_class_ids(lines, class_id_map, task=task)
            output_path = OSP.join(out_sub, label_file)
            with open(output_path, "w") as f:
                f.writelines(converted_lines)
            if pbar is not None:
                pbar.update()


def mirror_image_files(source_dir: str, dest_dir: str, pbar=None, *, no_use_link: bool) -> None:
    """Mirror ``source_dir`` into ``dest_dir`` with the same subtree layout.

    Each leaf file is either a symlink to the source file (default) or a full copy. We do **not**
    symlink the whole ``source_dir`` as ``dest_dir``: Ultralytics resolves dataset image roots
    with ``Path.resolve()``, which follows a **directory** symlink and makes ``img2label_paths()``
    point at the source ``labels/`` tree instead of converted labels under ``output_dir``.

    Args:
        source_dir: Root of the image tree to mirror (walked recursively).
        dest_dir: Root output directory; any existing path at this name is removed first.
        pbar: Optional progress object with ``update()``; called once per mirrored file.
        no_use_link: If True, copy each file with ``shutil.copy2``. If False, create a symlink with
            ``os.symlink`` pointing at the absolute source path (saves disk space).
    """
    source_dir = OSP.abspath(source_dir)
    if OSP.lexists(dest_dir):
        if OSP.islink(dest_dir):
            os.unlink(dest_dir)
        elif OSP.isdir(dest_dir):
            shutil.rmtree(dest_dir)
        else:
            os.remove(dest_dir)
    for root, _, files in os.walk(source_dir):
        rel = OSP.relpath(root, source_dir)
        dest_sub = dest_dir if rel == "." else OSP.join(dest_dir, rel)
        os.makedirs(dest_sub, exist_ok=True)
        for name in files:
            src = OSP.join(root, name)
            dst = OSP.join(dest_sub, name)
            # Replace existing file or symlink (lexists covers broken symlinks on some OSes).
            if OSP.lexists(dst):
                os.unlink(dst)
            if no_use_link:
                shutil.copy2(src, dst)
            else:
                os.symlink(OSP.abspath(src), dst)
            if pbar is not None:
                pbar.update()


def merge_labels_from_dir(label_dirs, output_dir, class_id_maps=None, filter_iou_threshold=None):
    """Merge labels from multiple directories.
    
    This function reads label files from multiple directories and merges them.
    For each label file (same filename across directories), all label lines are
    combined into a single file in the output directory.
    
    When filter_iou_threshold is in [0, 1], labels are processed in order of label_dirs.
    For each new label, if it has IoU > threshold with any previously merged label,
    the new label is adopted (considered as duplicate annotation of the same object).
    
    Args:
        label_dirs (list): List of label directory paths to merge from.
        output_dir (str): Output directory path where merged labels will be saved.
        class_id_maps (list | None): List of class ID maps to apply to the labels.
        filter_iou_threshold (float | None): IoU threshold for filtering duplicate annotations.
            If in [0, 1], enables IoU-based filtering. Default: None.

    Example:
        >>> merge_labels_from_dir(['dir1/labels', 'dir2/labels'], 'output/labels')
        # Merges all .txt files from dir1/labels and dir2/labels into output/labels
        >>> merge_labels_from_dir(['dir1/labels', 'dir2/labels'], 'output/labels', filter_iou_threshold=0.5)
        # Merges with IoU filtering: new labels with IoU > 0.5 is discarded
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect all label files from all directories
    all_label_files = set()
    for label_dir in label_dirs:
        for label_file in os.listdir(label_dir):
            if label_file.endswith('.txt'):
                all_label_files.add(label_file)
    
    # Check if IoU filtering is enabled
    use_iou_filter = filter_iou_threshold is not None and 0 <= filter_iou_threshold <= 1
    
    # Merge labels for each file
    for label_file in all_label_files:
        merged_labels = []
        
        # Process labels from each directory in order
        label_iter = zip(label_dirs, class_id_maps) if class_id_maps else zip(label_dirs, [None] * len(label_dirs))
        
        for label_dir, class_id_map in label_iter:
            label_path = OSP.join(label_dir, label_file)
            if not OSP.exists(label_path):
                continue
            
            # Read and convert labels
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            if class_id_map is not None:
                lines = convert_class_ids(lines, class_id_map)
            
            # Process each label line
            for line in lines:
                line_stripped = line.strip()
                if len(line_stripped) < 5:
                    continue
                
                # Parse label for both IoU filtering and deduplication
                class_id, x_center, y_center, width, height = parse_label_line(line_stripped)
                box = [x_center, y_center, width, height]
                
                if use_iou_filter:
                    # Filter label with existing labels
                    keep_this_label = True
                    for _, existing_box in merged_labels:
                        iou = calculate_iou_xywh(box, existing_box)
                        
                        if iou > filter_iou_threshold:
                            keep_this_label = False
                            break
                    
                    if keep_this_label:
                        merged_labels.append((class_id, box))
                else:
                    merged_labels.append((class_id, box))
        
        # Save merged labels
        if len(merged_labels) > 0:
            output_path = OSP.join(output_dir, label_file)
            with open(output_path, 'w') as f:
                for class_id, (x, y, w, h) in merged_labels:
                    f.write(f'{class_id} {x} {y} {w} {h}\n')


# ====== Memory Monitor Utils ======
class RealTimeMemoryMonitor:
    """Real-time memory monitor for GPU and system memory.
    
    This class monitors GPU memory and system memory usage in real-time using
    a background thread. It can update a progress bar with memory information.
    
    Example:
        >>> monitor = RealTimeMemoryMonitor(update_interval=0.5)
        >>> pbar = tqdm(range(100))
        >>> monitor.set_progress_bar(pbar)
        >>> monitor.start_monitoring()
        >>> # ... do work ...
        >>> monitor.stop_monitoring()
    """
    def __init__(self, update_interval=0.5):
        """Initialize the memory monitor.
        
        Args:
            update_interval (float): Time interval in seconds between memory updates. Default: 0.5
        """
        self.update_interval = update_interval
        self.monitoring = False
        self.monitor_thread = None
        self.gpu_mem = 0
        self.mem = 0
        self.pbar = None  # store progress bar reference
        
    def get_gpu_mem_mb(self):
        """Get current GPU memory usage in MB.
        
        Returns:
            int: GPU memory usage in MB, or 0 if CUDA is not available.
        """
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() // (1024 * 1024)
        return 0

    def get_mem_mb(self):
        """Get current system memory usage in MB.
        
        Returns:
            int: System memory usage in MB.
        """
        return psutil.Process().memory_info().rss // (1024 * 1024)
    
    def set_progress_bar(self, pbar):
        """Set the progress bar to update with memory information.
        
        Args:
            pbar: Progress bar object (e.g., from tqdm) that has a set_description method.
        """
        self.pbar = pbar
    
    def _monitor_loop(self):
        """Internal monitoring loop that runs in a background thread."""
        while self.monitoring:
            self.gpu_mem = self.get_gpu_mem_mb()
            self.mem = self.get_mem_mb()
            
            # Real-time update progress bar description
            if self.pbar is not None:
                self.pbar.set_description(f"GPU Mem: {self.gpu_mem:.2f} MB, Mem: {self.mem:.2f} MB")
            
            time.sleep(self.update_interval)
    
    def start_monitoring(self):
        """Start the memory monitoring in a background thread."""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop the memory monitoring and wait for the thread to finish."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def get_status(self):
        """Get current memory status as a string.
        
        Returns:
            str: Formatted string with current GPU and system memory usage.
        """
        return f"GPU Mem: {self.gpu_mem:.2f} MB, Mem: {self.mem:.2f} MB"


class RegDistributionLoss(nn.Module):
    """Criterion class for computing loss between two distributions (for prototype replay)."""

    def __init__(self, reg_max: int = 16, reduction: str = "mean") -> None:
        """
        Initialize the DistributionLoss module.
        
        Args:
            reg_max: Maximum value for regression (number of bins in distribution)
            reduction: Reduction method ('mean', 'sum', or 'none')
        """
        super().__init__()
        self.reg_max = reg_max
        self.reduction = reduction

    def __call__(self, pred_dist: torch.Tensor, target_dist: torch.Tensor) -> torch.Tensor:
        """
        Compute loss between two distributions.
        
        Args:
            pred_dist: Predicted distribution, shape (N, 4*reg_max) or (N, 4, reg_max)
            target_dist: Target distribution, shape (N, 4*reg_max) or (N, 4, reg_max)
        
        Returns:
            Loss value (scalar for 'mean' or 'sum', tensor for 'none')
        """
        # Ensure both tensors have the same shape
        assert pred_dist.shape == target_dist.shape, f"Shape mismatch: pred_dist {pred_dist.shape} vs target_dist {target_dist.shape}"
        
        # Reshape if needed: (N, 4*reg_max) -> (N, 4, reg_max)
        if pred_dist.dim() == 2 and pred_dist.shape[1] == 4 * self.reg_max:
            pred_dist = pred_dist.view(-1, 4, self.reg_max)
            target_dist = target_dist.view(-1, 4, self.reg_max)
        elif pred_dist.dim() == 2:
            # If not the expected shape, assume it's already in correct format or reshape accordingly
            raise ValueError(f"Unexpected pred_dist shape: {pred_dist.shape}, expected (N, {4 * self.reg_max}) or (N, 4, {self.reg_max})")
        
        target_prob = F.softmax(target_dist, dim=-1)  # (N, 4, reg_max)
        log_pred_prob = F.log_softmax(pred_dist, dim=-1)  # (N, 4, reg_max)
        
        # Compute cross-entropy: -sum(target * log(pred))
        # This is equivalent to KL divergence up to a constant (entropy of target)
        loss = -(target_prob * log_pred_prob).sum(dim=-1)  # (N, 4)
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss