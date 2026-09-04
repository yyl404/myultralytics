""" Expand the model classification head to support more classes.

Usage:
    $ python tools/expand_model_head.py \
        --model <path/to/model.pt> \
        --model_cfg <path/to/model.yaml> \
        --dataset <path/to/dataset.yaml> \
        --new_classes <class_list> \
        --save_path <path/to/save.pt> \
        --zero_weight_init (optional)

    Class ordering: existing model classes keep their ids and their order in the
    classification head; classes from --dataset/--new_classes that are not already
    known are appended after them in their dataset order. A class already known to
    the model keeps its existing id and is never duplicated; annotations of that
    class in the new dataset are aligned to the existing id by class name during
    dataset conversion (tools/convert_dataset_class_ids.py).

    Arguments:
        --model: Path to the trained model checkpoint (.pt file)
        --model_cfg: Path to the model configuration file (.yaml file)
        --dataset: Path to the dataset YAML file (alternative to --new_classes)
        --new_classes: List of new class names as a Python list string, 
            e.g., "['class1', 'class2', ...]" (alternative to --dataset)
        --save_path: Path where the expanded model will be saved
        --zero_weight_init: Whether to initialize the weights of the new classes to 0
        --class_embedding_init: Whether to initialize the weights of new classes using their text embeddings.
            Requires --yoloe_model to provide a YOLOE model for generating embeddings.
        --yoloe_model: Path to YOLOE model weights (required if --class_embedding_init is used)
        
Examples:
    # Expand by specifying incremental dataset yaml file
    $ python tools/expand_model_head.py \
        --model runs/yolov8l_voc_inc_10_10_fromscratch_vspreg/task-1/best.pt \
        --model_cfg yolov8l.yaml \
        --dataset data/VOC_inc_10_10/task_2_cls_10/dataset.yaml \
        --save_path runs/yolov8l_voc_inc_10_10_fromscratch_vspreg/task-2/task-1-best-expanded.pt

    # Expand by directly specifying new classes list
    $ python tools/expand_model_head.py \
        --model runs/yolov8l_voc_10_10_fromscratch_vspreg/task-1/best.pt \
        --model_cfg yolov8l.yaml \
        --new_classes "['diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']" \
        --save_path runs/yolov8l_voc_inc_10_10_fromscratch_vspreg/task-2/task-1-best-expanded.pt

    # Expand with embedding initialization for new classes
    $ python tools/expand_model_head.py \
        --model runs/yolov8l_voc_inc_10_10_fromscratch_vspreg/task-1/best.pt \
        --model_cfg yolov8l.yaml \
        --new_classes "['diningtable', 'dog', 'horse', 'motorbike', 'person']" \
        --save_path runs/yolov8l_voc_inc_10_10_fromscratch_vspreg/task-2/task-1-best-expanded.pt \
        --class_embedding_init \
        --yoloe_model yoloe-v8l-seg.pt
"""

import argparse
from os import path as OSP

from torch.nn import Sequential

from ultralytics import YOLO, YOLOE
from ultralytics.nn.tasks import DetectionModel, Detect, yaml_model_load
from ultralytics.utils import YAML

from utils import parse_list_string


def expand_detection_head(ckpt_path, model_cfg, channel_map, classes_names, save_dir, output_name,
                          zero_weight_init=False, class_embedding_init=False, yoloe_model_path=None,
                          incremental_history=None):
    """Expand the detection head output channels and migrate weights from old to new channels.
    
    This function expands the model's detection head to support more classes by allocating
    new channels for new classes and migrating weights from the original detection head to
    corresponding channels in the new model.
    
    Args:
        ckpt_path (str): Path to the model checkpoint file.
        model_cfg (str): Path to the model configuration file.
        channel_map (dict): Channel mapping dictionary from old model channels to new model
            corresponding channels (maps old_idx -> new_idx).
        classes_names (list): List of class names for the expanded model.
        save_dir (str): Directory path where the expanded model will be saved.
        output_name (str): Output filename for the expanded model.
        zero_weight_init (bool, optional): Whether to initialize weights of new channels to zero.
            Defaults to False.
        class_embedding_init (bool, optional): Whether to initialize weights of new classes using text embeddings.
            Defaults to False.
        yoloe_model_path (str, optional): Path to YOLOE model for generating embeddings.
            Required if class_embedding_init is True.
        incremental_history (list[dict], optional): Incremental class-space history to attach to
            the expanded model ([{"task": int, "names": list[str]}, ...], one entry per stage).
            Stored as a plain module attribute so every later checkpoint carries it.
    
    Returns:
        None: The function saves the expanded model to disk but does not return anything.
    """
    model = YOLO(ckpt_path).eval()
    assert isinstance(model.model, DetectionModel) and isinstance(model.model.model, Sequential)\
        and isinstance(model.model.model[-1], Detect), "Only support DetectionModel with Detect in the last layer"
    weight = model.model.state_dict()
    
    model_name = model_cfg.split(".")[0]
    model_cfg = yaml_model_load(model_cfg)
    model_cfg["nc"] = len(classes_names)
    YAML.save(data=model_cfg, file=OSP.join(save_dir, f"{model_name}-nc{len(classes_names)}.yaml"))
    new_model = YOLO(OSP.join(save_dir, f"{model_name}-nc{len(classes_names)}.yaml"))
    if zero_weight_init:
        for name, param in new_model.model.named_parameters():
            if 'cv3' in name and name.endswith('.2.weight'):
                param.data.zero_()
            if 'cv3' in name and name.endswith('.2.bias'):
                param.data.zero_()
    if class_embedding_init:
        # Load fused yoloe weights
        yoloe_model = YOLOE(yoloe_model_path).eval()
        tpe = yoloe_model.get_text_pe(classes_names)
        yoloe_model.model.model[-1].fuse(tpe)
        new_model.model.load_state_dict(yoloe_model.model.state_dict(), strict=False)
    new_weight = new_model.model.state_dict()

    # Migrate weights from old to new channels
    for key in new_weight.keys():
        if key in weight:
            layer_id = int(key.split('.')[1])
            # Handle the classification layer weights
            if layer_id == len(model.model.model) - 1 and 'cv3' in key and key.endswith('.2.weight'):
                # Migrate weights according to the transfer_map
                for old_idx, new_idx in channel_map.items():
                    if old_idx < weight[key].shape[0] and new_idx < new_weight[key].shape[0]:
                        new_weight[key][new_idx] = weight[key][old_idx].clone()
            
            # Handle the classification layer bias
            elif layer_id == len(model.model.model) - 1 and 'cv3' in key and key.endswith('.2.bias'):
                # Migrate bias according to the transfer_map
                for old_idx, new_idx in channel_map.items():
                    if old_idx < weight[key].shape[0] and new_idx < new_weight[key].shape[0]:
                        new_weight[key][new_idx] = weight[key][old_idx].clone()
            
            # Other layers are directly copied
            else:
                new_weight[key] = weight[key].clone()

    new_model.model.load_state_dict(new_weight)
    # `_end2end` / `agnostic_nms` / `max_det` live on the Detect head as Python attributes,
    # not in the state_dict. A yaml-rebuilt YOLO26 head defaults to end2end=True, so without
    # copying them the expanded model (and any teacher loaded from it) would decode the
    # untrained one2one branch instead of the one2many+NMS path the source model used.
    new_model.model.end2end = model.model.end2end
    src_head, dst_head = model.model.model[-1], new_model.model.model[-1]
    dst_head.agnostic_nms = src_head.agnostic_nms
    dst_head.max_det = src_head.max_det
    new_model.model.names = {k: v for k, v in enumerate(classes_names)}
    new_model.model.incremental_history = incremental_history
    new_model.save(OSP.join(save_dir, output_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model checkpoint (.pt file)")
    parser.add_argument("--model_cfg", type=str, required=True, help="Path to the model configuration file (.yaml file)")
    parser.add_argument("--dataset", type=str, required=False, default=None, help="Path to the dataset YAML file (alternative to --new_classes)")
    parser.add_argument("--new_classes", type=str, required=False, default=None, help="List of new class names as a Python list string, e.g., \"['class1', 'class2', ...]\" (alternative to --dataset)")
    parser.add_argument("--save_path", type=str, required=True, help="Path where the expanded model will be saved")
    parser.add_argument("--zero_weight_init", action="store_true", help="Whether to initialize the weights of the new classes to 0")
    parser.add_argument("--class_embedding_init", action="store_true", help="Whether to initialize the weights of new classes using their text embeddings")
    parser.add_argument("--yoloe_model", type=str, default=None, help="Path to YOLOE model weights (required if --class_embedding_init is used)")
    args = parser.parse_args()

    base_model = YOLO(args.model)
    base_classes = [base_model.names[i] for i in sorted(base_model.names.keys())]

    if args.new_classes is None:
        assert args.dataset is not None, "Either --new_classes or --dataset must be provided"
        data_cfg = YAML.load(args.dataset)
        new_classes = [data_cfg["names"][i] for i in sorted(data_cfg["names"].keys())]
    else:
        new_classes = parse_list_string(args.new_classes)

    # Existing classes keep their ids and order; unseen classes are appended after them
    # in dataset order. A class already known to the base model keeps the existing id
    # (its annotations are aligned to that id by class name during dataset conversion),
    # so no duplicate channel is created.
    all_classes = list(base_classes)
    for cls in new_classes:
        if cls not in all_classes:
            all_classes.append(cls)

    # Base class channels are preserved 1:1 (identity map); appended classes get
    # fresh channels initialized by expand_detection_head.
    base_class_id_map = {i: i for i in range(len(base_classes))}

    # Carry the incremental class-space history forward: one entry per stage, so the
    # expanded model remembers which classes each completed task introduced. A model
    # without a recorded history has seen a single stage: its whole class space.
    base_history = getattr(base_model.model, "incremental_history", None)
    if base_history is None:
        base_history = [{"task": 1, "names": base_classes}]
    history_names = [name for stage in base_history for name in stage["names"]]
    if history_names != base_classes:
        raise ValueError(
            f"incremental_history of {args.model} does not match its class space: "
            f"history covers {history_names}, model names are {base_classes}"
        )
    incremental_history = [*base_history, {"task": len(base_history) + 1, "names": all_classes[len(base_classes):]}]

    root_dir, model_name = OSP.split(args.save_path)
    expand_detection_head(args.model, args.model_cfg, base_class_id_map, all_classes,
                          root_dir, model_name, args.zero_weight_init,
                          args.class_embedding_init, args.yoloe_model,
                          incremental_history=incremental_history)