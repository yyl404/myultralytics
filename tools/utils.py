"""Utility functions for tools."""

import ast
import os
import os.path as OSP
import threading
import time

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
def convert_class_ids(label_lines, class_id_map):
    """Convert the class IDs in the label lines by the class_id_map"""
    converted_lines = []
    for line in label_lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            old_cat_id = int(parts[0])
            if old_cat_id in class_id_map:
                new_cat_id = class_id_map[old_cat_id]
                parts[0] = str(new_cat_id)
                converted_lines.append(' '.join(parts) + '\n')
            # else:
                # LOGGER.warning(f"Class ID {old_cat_id} not found in class_id_map")
    return converted_lines


def convert_class_ids_from_dir(labels_dir, class_id_map, output_dir):
    """Read all label files in a directory and convert class IDs."""
    for label_file in os.listdir(labels_dir):
        if label_file.endswith('.txt'):
            label_path = OSP.join(labels_dir, label_file)
            with open(label_path, 'r') as f:
                lines = f.readlines()
            converted_lines = convert_class_ids(lines, class_id_map)

            output_path = OSP.join(output_dir, label_file)
            with open(output_path, 'w') as f:
                f.writelines(converted_lines)


def merge_labels_from_dir(label_dirs, output_dir, class_id_maps=None):
    """Merge labels from multiple directories.
    
    This function reads label files from multiple directories and merges them.
    For each label file (same filename across directories), all label lines are
    combined into a single file in the output directory.
    
    Args:
        label_dirs (list): List of label directory paths to merge from.
        output_dir (str): Output directory path where merged labels will be saved.
        class_id_maps (list | None): List of class ID maps to apply to the labels.

    Example:
        >>> merge_labels_from_dir(['dir1/labels', 'dir2/labels'], 'output/labels')
        # Merges all .txt files from dir1/labels and dir2/labels into output/labels
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect all label files from all directories
    all_label_files = set()
    for label_dir in label_dirs:
        for label_file in os.listdir(label_dir):
            if label_file.endswith('.txt'):
                all_label_files.add(label_file)
    
    # Merge labels for each file
    for label_file in all_label_files:
        merged_lines = set()
        
        # Read labels from each directory
        if class_id_maps is None:
            for label_dir in label_dirs:
                label_path = OSP.join(label_dir, label_file)
                if OSP.exists(label_path):
                    with open(label_path, 'r') as f:
                        lines = f.readlines()
                        merged_lines.update(lines)
        else:
            for label_dir, class_id_map in zip(label_dirs, class_id_maps):
                label_path = OSP.join(label_dir, label_file)
                if OSP.exists(label_path):
                    with open(label_path, 'r') as f:
                        lines = f.readlines()
                        converted_lines = convert_class_ids(lines, class_id_map)
                        merged_lines.update(converted_lines)
        
        # Save merged labels
        if merged_lines:
            output_path = OSP.join(output_dir, label_file)
            with open(output_path, 'w') as f:
                f.writelines(merged_lines)


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