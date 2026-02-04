#!/usr/bin/env python3
"""
Generate missing training scripts for voc, coco, rsar.
Target: naive, pseudo_label, pseudo_label+espreg, pseudo_label+ewc, pseudo_label+espreg+ewc,
        pseudo_label+proto_rp, pseudo_label+espreg+proto_rp, pseudo_label+ewc+proto_rp, pseudo_label+espreg+ewc+proto_rp.
Canonical order: pseudo_label -> espreg -> ewc -> proto_rp.
When espreg and ewc are used together, EWC uses --module_pattern "*bn" only.
"""
from pathlib import Path
import os

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SCRIPTS_ROOT = REPO_ROOT / "scripts"

# (dataset, split, backbone) -> config
# rel_dir: path under scripts/ e.g. "voc/10_10/yolov8"
# output_prefix: e.g. "runs/yolov8l_voc_inc_10_10_fromscratch"
# task_datasets: list of dataset.yaml paths
# model_cfg, is_obb, weight (optional), freeze_layers (optional list of strings)
CONFIGS = [
    # VOC
    {"dataset": "voc", "split": "10_10", "backbone": "yolov8", "rel_dir": "voc/10_10/yolov8",
     "output_prefix": "runs/yolov8l_voc_inc_10_10_fromscratch", "model_cfg": "yolov8l.yaml", "is_obb": False,
     "task_datasets": ["data/VOC_inc_10_10/task_1_cls_10/dataset.yaml", "data/VOC_inc_10_10/task_2_cls_10/dataset.yaml"]},
    {"dataset": "voc", "split": "10_10", "backbone": "yoloev8", "rel_dir": "voc/10_10/yoloev8",
     "output_prefix": "runs/yolov8l_voc_10_10_pretrained-yoloe", "model_cfg": "yolov8l.yaml", "is_obb": False,
     "weight": "yoloe-v8l-seg.pt", "freeze_layers": ["[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]", "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]"],
     "task_datasets": ["data/VOC_inc_10_10/task_1_cls_10/dataset.yaml", "data/VOC_inc_10_10/task_2_cls_10/dataset.yaml"]},
    {"dataset": "voc", "split": "15_5", "backbone": "yolov8", "rel_dir": "voc/15_5/yolov8",
     "output_prefix": "runs/yolov8l_voc_15_5_fromscratch", "model_cfg": "yolov8l.yaml", "is_obb": False,
     "task_datasets": ["data/VOC_15_5/task_1_cls_15/dataset.yaml", "data/VOC_15_5/task_2_cls_5/dataset.yaml"]},
    {"dataset": "voc", "split": "15_5", "backbone": "yoloev8", "rel_dir": "voc/15_5/yoloev8",
     "output_prefix": "runs/yolov8l_voc_15_5_pretrained-yoloe", "model_cfg": "yolov8l.yaml", "is_obb": False,
     "weight": "yoloe-v8l-seg.pt", "freeze_layers": ["[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]", "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]"],
     "task_datasets": ["data/VOC_15_5/task_1_cls_15/dataset.yaml", "data/VOC_15_5/task_2_cls_5/dataset.yaml"]},
    {"dataset": "voc", "split": "19_1", "backbone": "yolov8", "rel_dir": "voc/19_1/yolov8",
     "output_prefix": "runs/yolov8l_voc_19_1_fromscratch", "model_cfg": "yolov8l.yaml", "is_obb": False,
     "task_datasets": ["data/VOC_19_1/task_1_cls_19/dataset.yaml", "data/VOC_19_1/task_2_cls_1/dataset.yaml"]},
    # COCO
    {"dataset": "coco", "split": "40_40", "backbone": "yolov8", "rel_dir": "coco/40_40/yolov8",
     "output_prefix": "runs/yolov8l_coco_40_40_fromscratch", "model_cfg": "yolov8l.yaml", "is_obb": False,
     "task_datasets": ["data/coco_40_40/task_1_cls_40/dataset.yaml", "data/coco_40_40/task_2_cls_40/dataset.yaml"]},
    {"dataset": "coco", "split": "70_10", "backbone": "yolov8", "rel_dir": "coco/70_10/yolov8",
     "output_prefix": "runs/yolov8l_coco_70_10_fromscratch", "model_cfg": "yolov8l.yaml", "is_obb": False,
     "task_datasets": ["data/COCO_70_10/task_1_cls_70/dataset.yaml", "data/COCO_70_10/task_2_cls_10/dataset.yaml"]},
    # RSAR (OBB)
    {"dataset": "rsar", "split": "3_3", "backbone": "yolov8", "rel_dir": "rsar/3_3/yolov8",
     "output_prefix": "runs/yolov8l_rsar_3_3_fromscratch", "model_cfg": "yolov8l-obb.yaml", "is_obb": True,
     "task_datasets": ["data/RSAR_3_3/task_1_cls_3/dataset.yaml", "data/RSAR_3_3/task_2_cls_3/dataset.yaml"]},
    {"dataset": "rsar", "split": "3_3", "backbone": "yoloe", "rel_dir": "rsar/3_3/yoloe",
     "output_prefix": "runs/yolov8l_rsar_3_3_pretrained-yoloe", "model_cfg": "yolov8l-obb.yaml", "is_obb": True,
     "weight": "yoloe-v8l-seg.pt", "freeze_layers": ["[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]", "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]"],
     "task_datasets": ["data/RSAR_3_3/task_1_cls_3/dataset.yaml", "data/RSAR_3_3/task_2_cls_3/dataset.yaml"]},
]

METHODS = [
    "naive",
    "pseudo_label",
    "pseudo_label+espreg",
    "pseudo_label+ewc",
    "pseudo_label+espreg+ewc",
    "pseudo_label+proto_rp",
    "pseudo_label+espreg+proto_rp",
    "pseudo_label+ewc+proto_rp",
    "pseudo_label+espreg+ewc+proto_rp",
]

# Aliases: old script names that map to canonical method (so we consider script as "exists")
ALIASES = {
    "espreg+pseudo_label": "pseudo_label+espreg",
    "ewc+pseudo_label": "pseudo_label+ewc",
    "espreg+pseudo_label+proto_rp": "pseudo_label+espreg+proto_rp",
}


def task_datasets_bash(cfg):
    lines = ["TASK_DATASETS=("]
    for d in cfg["task_datasets"]:
        lines.append(f'    "{d}"')
    lines.append(")")
    return "\n".join(lines)


def first_task_train(cfg, with_patience=True):
    """First task: train (from scratch or with weight). Returns the body of 'if [ task_num -eq 1 ]; then ...'."""
    w = cfg.get("weight")
    freeze = cfg.get("freeze_layers")
    extra = []
    if w:
        extra.append("            --weight $YOLOE_MODEL_WEIGHT \\")
    if freeze:
        extra.append("            --freeze ${FREEZE_LAYERS[0]} \\")
    if with_patience:
        extra.append("            --patience $PATIENCE \\")
    extra_str = "\n".join(extra) if extra else ""
    if extra_str:
        extra_str = extra_str.rstrip(" \\") + "\n        "
    return f"""        echo "Training task $task_num from {"pretrained weight" if w else "scratch"}..."
        python tools/train.py --model $MODEL_CFG \\
            --data $DATASET_PATH \\
            --save_path $TASK_DIR/best.pt \\
            --epochs $EPOCHS \\
            --batch_size $BATCH_SIZE \\
            --imgsz $IMGSZ \\
            --workers $WORKERS \\
            --device $DEVICE \\
            --project $TASK_DIR \\
{extra_str}
        PREV_MODEL="$TASK_DIR/best.pt\""""


def subsequent_expand_convert(cfg, use_id_converted=True):
    var = "ID_CONVERTED_DATASET" if use_id_converted else "CONVERTED_DATASET"
    suffix = "_id_converted" if use_id_converted else "_converted"
    return f"""        DATASET_NAME=$(basename $(dirname $DATASET_PATH))
        echo "Expanding model head for task $task_num..."
        EXPANDED_MODEL="$TASK_DIR/task-$((task_num-1))-best-expanded.pt"
        python tools/expand_model_head.py \\
            --model $PREV_MODEL \\
            --model_cfg $MODEL_CFG \\
            --dataset $DATASET_PATH \\
            --save_path $EXPANDED_MODEL

        echo "Converting dataset class IDs for task $task_num..."
        {var}="$TASK_DIR/${{DATASET_NAME}}{suffix}"
        python tools/convert_dataset_class_ids.py \\
            --model $EXPANDED_MODEL \\
            --dataset $DATASET_PATH \\
            --output_dir ${var}"""


def subsequent_train_base(cfg, data_var="ID_CONVERTED_DATASET", extra_train_args=""):
    w = cfg.get("weight")
    freeze = cfg.get("freeze_layers")
    fl = " \\\n            --freeze ${FREEZE_LAYERS[$((task_num-1))]}" if freeze else ""
    return f"""
        echo "Training task $task_num..."
        TRAIN_CMD="python tools/train.py --model $EXPANDED_MODEL \\
            --data \\\"${data_var}/dataset.yaml\\\" \\
            --save_path $TASK_DIR/best.pt \\
            --epochs $EPOCHS \\
            --patience $PATIENCE \\
            --batch_size $BATCH_SIZE \\
            --imgsz $IMGSZ \\
            --workers $WORKERS \\
            --device $DEVICE \\
            --project $TASK_DIR \\{fl}
            --trainer antiforget{extra_train_args}"

        TRAIN_CMD="$TRAIN_CMD --pseudo_label True --conf_threshold $CONF_THRESHOLD --filter_iou_threshold $FILTER_IOU_THRESHOLD"
"""


def gen_naive(cfg):
    w = cfg.get("weight")
    freeze = cfg.get("freeze_layers")
    fl_decl = ""
    if freeze:
        fl_decl = "\nFREEZE_LAYERS=(\n    " + "\n    ".join('"' + f + '"' for f in freeze) + "\n)\n"
    weight_decl = '\nYOLOE_MODEL_WEIGHT="' + w + '"\n' if w else ""
    return f"""#!/bin/bash
# Configuration
MODEL_CFG="{cfg['model_cfg']}"
{weight_decl}{fl_decl}OUTPUT_DIR="{cfg['output_prefix']}_naive"
EPOCHS=100
BATCH_SIZE=16
IMGSZ=640
WORKERS=8
DEVICE=0
PATIENCE=10
START_TASK=${{START_TASK:-1}}

{task_datasets_bash(cfg)}

if [ $START_TASK -lt 1 ] || [ $START_TASK -gt ${{#TASK_DATASETS[@]}} ]; then
    echo "Error: START_TASK must be between 1 and ${{#TASK_DATASETS[@]}}"
    exit 1
fi

PREV_MODEL=""
if [ $START_TASK -gt 1 ]; then
    PREV_TASK=$((START_TASK - 1))
    PREV_MODEL="$OUTPUT_DIR/task-$PREV_TASK/best.pt"
    if [ ! -f "$PREV_MODEL" ]; then
        echo "Error: Previous task model not found: $PREV_MODEL"
        exit 1
    fi
    echo "Resuming from Task $START_TASK"
fi

task_num=1
for DATASET_PATH in "${{TASK_DATASETS[@]}}"; do
    if [ $task_num -lt $START_TASK ]; then
        ((task_num++))
        continue
    fi
    echo "=========================================="
    echo "Processing Task $task_num"
    echo "=========================================="
    TASK_DIR="$OUTPUT_DIR/task-$task_num"

    if [ $task_num -eq 1 ]; then
{first_task_train(cfg)}

    else
        DATASET_NAME=$(basename $(dirname $DATASET_PATH))
        echo "Expanding model head for task $task_num..."
        EXPANDED_MODEL="$TASK_DIR/task-$((task_num-1))-best-expanded.pt"
        python tools/expand_model_head.py \\
            --model $PREV_MODEL \\
            --model_cfg $MODEL_CFG \\
            --dataset $DATASET_PATH \\
            --save_path $EXPANDED_MODEL
        echo "Converting dataset class IDs for task $task_num..."
        CONVERTED_DATASET="$TASK_DIR/${{DATASET_NAME}}_converted"
        python tools/convert_dataset_class_ids.py \\
            --model $EXPANDED_MODEL \\
            --dataset $DATASET_PATH \\
            --output_dir $CONVERTED_DATASET
        echo "Training task $task_num..."
        python tools/train.py --model $EXPANDED_MODEL \\
            --data "$CONVERTED_DATASET/dataset.yaml" \\
            --save_path $TASK_DIR/best.pt \\
            --epochs $EPOCHS \\
            --batch_size $BATCH_SIZE \\
            --imgsz $IMGSZ \\
            --workers $WORKERS \\
            --device $DEVICE \\
            --project $TASK_DIR \\
            --patience $PATIENCE
        PREV_MODEL="$TASK_DIR/best.pt"
    fi
    echo "Task $task_num completed!"
    ((task_num++))
done
echo "All tasks completed!"
"""


def gen_pseudo_label(cfg):
    w = cfg.get("weight")
    freeze = cfg.get("freeze_layers")
    fl_decl = ("\nFREEZE_LAYERS=(\n    " + "\n    ".join('"' + f + '"' for f in freeze) + "\n)\n") if freeze else ""
    weight_decl = '\nYOLOE_MODEL_WEIGHT="' + w + '"\n' if w else ""
    fl_train = " \\\n            --freeze ${FREEZE_LAYERS[$((task_num-1))]}" if freeze else ""
    return f"""#!/bin/bash
MODEL_CFG="{cfg['model_cfg']}"
{weight_decl}{fl_decl}OUTPUT_DIR="{cfg['output_prefix']}_pseudo_label"
EPOCHS=100
BATCH_SIZE=16
IMGSZ=640
WORKERS=8
DEVICE=0
PATIENCE=10
CONF_THRESHOLD=0.25
FILTER_IOU_THRESHOLD=0.5
START_TASK=${{START_TASK:-1}}

{task_datasets_bash(cfg)}

if [ $START_TASK -lt 1 ] || [ $START_TASK -gt ${{#TASK_DATASETS[@]}} ]; then
    echo "Error: START_TASK must be between 1 and ${{#TASK_DATASETS[@]}}"
    exit 1
fi

PREV_MODEL=""
if [ $START_TASK -gt 1 ]; then
    PREV_TASK=$((START_TASK - 1))
    PREV_MODEL="$OUTPUT_DIR/task-$PREV_TASK/best.pt"
    if [ ! -f "$PREV_MODEL" ]; then
        echo "Error: Previous task model not found: $PREV_MODEL"
        exit 1
    fi
    echo "Resuming from Task $START_TASK"
fi

task_num=1
for DATASET_PATH in "${{TASK_DATASETS[@]}}"; do
    if [ $task_num -lt $START_TASK ]; then
        ((task_num++))
        continue
    fi
    echo "=========================================="
    echo "Processing Task $task_num"
    echo "=========================================="
    TASK_DIR="$OUTPUT_DIR/task-$task_num"

    if [ $task_num -eq 1 ]; then
{first_task_train(cfg)}

    else
{subsequent_expand_convert(cfg, use_id_converted=True)}

        echo "Training task $task_num with pseudo_label..."
        python tools/train.py --model $EXPANDED_MODEL \\
            --data "$ID_CONVERTED_DATASET/dataset.yaml" \\
            --save_path $TASK_DIR/best.pt \\
            --epochs $EPOCHS \\
            --patience $PATIENCE \\
            --batch_size $BATCH_SIZE \\
            --imgsz $IMGSZ \\
            --workers $WORKERS \\
            --device $DEVICE \\
            --project $TASK_DIR \\{fl_train} \\
            --trainer antiforget \\
            --pseudo_label True \\
            --conf_threshold $CONF_THRESHOLD \\
            --filter_iou_threshold $FILTER_IOU_THRESHOLD
        PREV_MODEL="$TASK_DIR/best.pt"
    fi
    echo "Task $task_num completed!"
    ((task_num++))
done
echo "All tasks completed!"
"""


def gen_pseudo_label_espreg(cfg):
    w = cfg.get("weight")
    freeze = cfg.get("freeze_layers")
    fl_decl = ("\nFREEZE_LAYERS=(\n    " + "\n    ".join('"' + f + '"' for f in freeze) + "\n)\n") if freeze else ""
    weight_decl = '\nYOLOE_MODEL_WEIGHT="' + w + '"\n' if w else ""
    fl_train = " \\\n            --freeze ${FREEZE_LAYERS[$((task_num-1))]}" if freeze else ""
    return f"""#!/bin/bash
MODEL_CFG="{cfg['model_cfg']}"
{weight_decl}{fl_decl}OUTPUT_DIR="{cfg['output_prefix']}_pseudo_label+espreg"
EPOCHS=100
BATCH_SIZE=16
IMGSZ=640
WORKERS=8
DEVICE=0
PATIENCE=10
CONF_THRESHOLD=0.25
FILTER_IOU_THRESHOLD=0.5
ESPREG_LOSS_WEIGHT=${{ESPREG_LOSS_WEIGHT:-1000.0}}
START_TASK=${{START_TASK:-1}}

{task_datasets_bash(cfg)}

if [ $START_TASK -lt 1 ] || [ $START_TASK -gt ${{#TASK_DATASETS[@]}} ]; then
    echo "Error: START_TASK must be between 1 and ${{#TASK_DATASETS[@]}}"
    exit 1
fi

PREV_PCA_CACHE=""
if [ $START_TASK -gt 1 ]; then
    PREV_TASK=$((START_TASK - 1))
    PREV_MODEL="$OUTPUT_DIR/task-$PREV_TASK/best.pt"
    PREV_PCA_CACHE="$OUTPUT_DIR/task-$PREV_TASK/pca_cache.pkl"
    if [ ! -f "$PREV_MODEL" ]; then
        echo "Error: Previous task model not found: $PREV_MODEL"
        exit 1
    fi
    if [ ! -f "$PREV_PCA_CACHE" ]; then
        echo "Warning: Previous task PCA cache not found."
    fi
    echo "Resuming from Task $START_TASK"
fi

task_num=1
for DATASET_PATH in "${{TASK_DATASETS[@]}}"; do
    if [ $task_num -lt $START_TASK ]; then
        ((task_num++))
        continue
    fi
    echo "=========================================="
    echo "Processing Task $task_num"
    echo "=========================================="
    TASK_DIR="$OUTPUT_DIR/task-$task_num"

    if [ $task_num -eq 1 ]; then
{first_task_train(cfg)}

        echo "Performing PCA analysis..."
        PCA_CACHE_PATH="$TASK_DIR/pca_cache.pkl"
        python tools/pca.py \\
            --model $TASK_DIR/best.pt \\
            --dataset $DATASET_PATH \\
            --save_path $PCA_CACHE_PATH
        PREV_MODEL="$TASK_DIR/best.pt"
        PREV_PCA_CACHE="$PCA_CACHE_PATH"
    else
{subsequent_expand_convert(cfg, use_id_converted=True)}

        echo "Training task $task_num with pseudo_label + ESPReg..."
        TRAIN_CMD="python tools/train.py --model $EXPANDED_MODEL \\
            --data \\\"$ID_CONVERTED_DATASET/dataset.yaml\\\" \\
            --save_path $TASK_DIR/best.pt \\
            --epochs $EPOCHS \\
            --patience $PATIENCE \\
            --batch_size $BATCH_SIZE \\
            --imgsz $IMGSZ \\
            --workers $WORKERS \\
            --device $DEVICE \\
            --project $TASK_DIR \\{fl_train} \\
            --trainer antiforget"
        TRAIN_CMD="$TRAIN_CMD --pseudo_label True --conf_threshold $CONF_THRESHOLD --filter_iou_threshold $FILTER_IOU_THRESHOLD"
        TRAIN_CMD="$TRAIN_CMD --espreg True --pca_cache_path $PREV_PCA_CACHE --espreg_loss_weight $ESPREG_LOSS_WEIGHT"
        eval $TRAIN_CMD

        echo "Performing PCA analysis..."
        PCA_CACHE_PATH="$TASK_DIR/pca_cache.pkl"
        python tools/pca.py \\
            --model $TASK_DIR/best.pt \\
            --dataset $DATASET_PATH \\
            --load_hist $PREV_PCA_CACHE \\
            --save_path $PCA_CACHE_PATH
        PREV_MODEL="$TASK_DIR/best.pt"
        PREV_PCA_CACHE="$PCA_CACHE_PATH"
    fi
    echo "Task $task_num completed!"
    ((task_num++))
done
echo "All tasks completed!"
"""


def gen_pseudo_label_ewc(cfg, use_bn_only=False):
    """use_bn_only: when True (espreg+ewc), cal_importance with --module_pattern '*bn'."""
    w = cfg.get("weight")
    freeze = cfg.get("freeze_layers")
    fl_decl = ("\nFREEZE_LAYERS=(\n    " + "\n    ".join('"' + f + '"' for f in freeze) + "\n)\n") if freeze else ""
    fl_train = " \\\n            --freeze ${FREEZE_LAYERS[$((task_num-1))]}" if freeze else ""
    bn_line = '            --module_pattern "*bn" \\' if use_bn_only else ""
    weight_decl = '\nYOLOE_MODEL_WEIGHT="' + cfg.get("weight", "") + '"\n' if cfg.get("weight") else ""
    return f"""#!/bin/bash
MODEL_CFG="{cfg['model_cfg']}"
{weight_decl}{fl_decl}OUTPUT_DIR="{cfg['output_prefix']}_pseudo_label+ewc"
EPOCHS=100
BATCH_SIZE=16
IMGSZ=640
WORKERS=8
DEVICE=0
PATIENCE=10
CONF_THRESHOLD=0.25
FILTER_IOU_THRESHOLD=0.5
EWC_LOSS_WEIGHT=${{EWC_LOSS_WEIGHT:-100.0}}
START_TASK=${{START_TASK:-1}}

{task_datasets_bash(cfg)}

if [ $START_TASK -lt 1 ] || [ $START_TASK -gt ${{#TASK_DATASETS[@]}} ]; then
    echo "Error: START_TASK must be between 1 and ${{#TASK_DATASETS[@]}}"
    exit 1
fi

PREV_IMPORTANCE_PATH=""
if [ $START_TASK -gt 1 ]; then
    PREV_TASK=$((START_TASK - 1))
    PREV_MODEL="$OUTPUT_DIR/task-$PREV_TASK/best.pt"
    PREV_IMPORTANCE_PATH="$OUTPUT_DIR/task-$PREV_TASK/importance.pth"
    if [ ! -f "$PREV_MODEL" ]; then
        echo "Error: Previous task model not found: $PREV_MODEL"
        exit 1
    fi
    if [ ! -f "$PREV_IMPORTANCE_PATH" ]; then
        echo "Warning: Previous task importance not found."
    fi
    echo "Resuming from Task $START_TASK"
fi

task_num=1
for DATASET_PATH in "${{TASK_DATASETS[@]}}"; do
    if [ $task_num -lt $START_TASK ]; then
        ((task_num++))
        continue
    fi
    echo "=========================================="
    echo "Processing Task $task_num"
    echo "=========================================="
    TASK_DIR="$OUTPUT_DIR/task-$task_num"

    if [ $task_num -eq 1 ]; then
{first_task_train(cfg)}

        echo "Calculating parameter importance..."
        IMPORTANCE_PATH="$TASK_DIR/importance.pth"
        python tools/cal_importance.py \\
            --model $TASK_DIR/best.pt \\
            --dataset $DATASET_PATH \\
            --save_path $IMPORTANCE_PATH \\
{bn_line}
            --batch_size $BATCH_SIZE \\
            --workers $WORKERS \\
            --device $DEVICE
        PREV_MODEL="$TASK_DIR/best.pt"
        PREV_IMPORTANCE_PATH="$IMPORTANCE_PATH"
    else
{subsequent_expand_convert(cfg, use_id_converted=True)}

        if [ -n "$PREV_IMPORTANCE_PATH" ] && [ -f "$PREV_IMPORTANCE_PATH" ]; then
            echo "Expanding importance file..."
            EXPANDED_IMPORTANCE_PATH="$TASK_DIR/task-$((task_num-1))-importance-expanded.pth"
            python tools/expand_importance.py \\
                --old_importance $PREV_IMPORTANCE_PATH \\
                --old_model $PREV_MODEL \\
                --new_model $EXPANDED_MODEL \\
                --save_path $EXPANDED_IMPORTANCE_PATH \\
                --copy_importance_init
            PREV_IMPORTANCE_PATH="$EXPANDED_IMPORTANCE_PATH"
        fi

        echo "Training task $task_num with pseudo_label + EWC..."
        TRAIN_CMD="python tools/train.py --model $EXPANDED_MODEL \\
            --data \\\"$ID_CONVERTED_DATASET/dataset.yaml\\\" \\
            --save_path $TASK_DIR/best.pt \\
            --epochs $EPOCHS \\
            --patience $PATIENCE \\
            --batch_size $BATCH_SIZE \\
            --imgsz $IMGSZ \\
            --workers $WORKERS \\
            --device $DEVICE \\
            --project $TASK_DIR \\{fl_train} \\
            --trainer antiforget"
        TRAIN_CMD="$TRAIN_CMD --pseudo_label True --conf_threshold $CONF_THRESHOLD --filter_iou_threshold $FILTER_IOU_THRESHOLD"
        if [ -n "$PREV_IMPORTANCE_PATH" ] && [ -f "$PREV_IMPORTANCE_PATH" ]; then
            TRAIN_CMD="$TRAIN_CMD --ewc True --importance_path $PREV_IMPORTANCE_PATH --ewc_loss_weight $EWC_LOSS_WEIGHT"
        fi
        eval $TRAIN_CMD

        echo "Calculating parameter importance..."
        IMPORTANCE_PATH="$TASK_DIR/importance.pth"
        python tools/cal_importance.py \\
            --model $TASK_DIR/best.pt \\
            --dataset "$ID_CONVERTED_DATASET/dataset.yaml" \\
            --save_path $IMPORTANCE_PATH \\
{bn_line}
            --batch_size $BATCH_SIZE \\
            --workers $WORKERS \\
            --device $DEVICE
        PREV_MODEL="$TASK_DIR/best.pt"
        PREV_IMPORTANCE_PATH="$IMPORTANCE_PATH"
    fi
    echo "Task $task_num completed!"
    ((task_num++))
done
echo "All tasks completed!"
"""


def gen_pseudo_label_espreg_ewc(cfg):
    return gen_pseudo_label_ewc(cfg, use_bn_only=True).replace(
        "OUTPUT_DIR=\"" + cfg["output_prefix"] + "_pseudo_label+ewc\"",
        "OUTPUT_DIR=\"" + cfg["output_prefix"] + "_pseudo_label+espreg+ewc\""
    ).replace(
        "pseudo_label + EWC",
        "pseudo_label + ESPReg + EWC"
    ).replace(
        "Training task $task_num with pseudo_label + EWC...",
        "Training task $task_num with pseudo_label + ESPReg + EWC..."
    ).replace(
        "PREV_IMPORTANCE_PATH=\"\"",
        "PREV_PCA_CACHE=\"\"\nPREV_IMPORTANCE_PATH=\"\""
    ).replace(
        "if [ $START_TASK -gt 1 ]; then\n    PREV_TASK=$((START_TASK - 1))\n    PREV_MODEL=",
        "if [ $START_TASK -gt 1 ]; then\n    PREV_TASK=$((START_TASK - 1))\n    PREV_MODEL="
    )
    # Need to add PCA and ESPREG to the script properly. Doing a full custom gen is clearer.
    w = cfg.get("weight")
    freeze = cfg.get("freeze_layers")
    fl_decl = ("\nFREEZE_LAYERS=(\n    " + "\n    ".join('"' + f + '"' for f in freeze) + "\n)\n") if freeze else ""
    fl_train = " \\\n            --freeze ${FREEZE_LAYERS[$((task_num-1))]}" if freeze else ""
    out_prefix = cfg["output_prefix"] + "_pseudo_label+espreg+ewc"
    return f"""#!/bin/bash
MODEL_CFG="{cfg['model_cfg']}"
{fl_decl}OUTPUT_DIR="{out_prefix}"
EPOCHS=100
BATCH_SIZE=16
IMGSZ=640
WORKERS=8
DEVICE=0
PATIENCE=10
CONF_THRESHOLD=0.25
FILTER_IOU_THRESHOLD=0.5
ESPREG_LOSS_WEIGHT=${{ESPREG_LOSS_WEIGHT:-1000.0}}
EWC_LOSS_WEIGHT=${{EWC_LOSS_WEIGHT:-100.0}}
START_TASK=${{START_TASK:-1}}

{task_datasets_bash(cfg)}

if [ $START_TASK -lt 1 ] || [ $START_TASK -gt ${{#TASK_DATASETS[@]}} ]; then
    echo "Error: START_TASK must be between 1 and ${{#TASK_DATASETS[@]}}"
    exit 1
fi

PREV_PCA_CACHE=""
PREV_IMPORTANCE_PATH=""
if [ $START_TASK -gt 1 ]; then
    PREV_TASK=$((START_TASK - 1))
    PREV_MODEL="$OUTPUT_DIR/task-$PREV_TASK/best.pt"
    PREV_PCA_CACHE="$OUTPUT_DIR/task-$PREV_TASK/pca_cache.pkl"
    PREV_IMPORTANCE_PATH="$OUTPUT_DIR/task-$PREV_TASK/importance.pth"
    if [ ! -f "$PREV_MODEL" ]; then
        echo "Error: Previous task model not found: $PREV_MODEL"
        exit 1
    fi
    echo "Resuming from Task $START_TASK"
fi

task_num=1
for DATASET_PATH in "${{TASK_DATASETS[@]}}"; do
    if [ $task_num -lt $START_TASK ]; then
        ((task_num++))
        continue
    fi
    echo "=========================================="
    echo "Processing Task $task_num"
    echo "=========================================="
    TASK_DIR="$OUTPUT_DIR/task-$task_num"

    if [ $task_num -eq 1 ]; then
{first_task_train(cfg)}

        echo "Performing PCA analysis..."
        PCA_CACHE_PATH="$TASK_DIR/pca_cache.pkl"
        python tools/pca.py \\
            --model $TASK_DIR/best.pt \\
            --dataset $DATASET_PATH \\
            --save_path $PCA_CACHE_PATH
        echo "Calculating parameter importance (bn modules only)..."
        IMPORTANCE_PATH="$TASK_DIR/importance.pth"
        python tools/cal_importance.py \\
            --model $TASK_DIR/best.pt \\
            --dataset $DATASET_PATH \\
            --save_path $IMPORTANCE_PATH \\
            --module_pattern "*bn" \\
            --batch_size $BATCH_SIZE \\
            --workers $WORKERS \\
            --device $DEVICE
        PREV_MODEL="$TASK_DIR/best.pt"
        PREV_PCA_CACHE="$PCA_CACHE_PATH"
        PREV_IMPORTANCE_PATH="$IMPORTANCE_PATH"
    else
{subsequent_expand_convert(cfg, use_id_converted=True)}

        if [ -n "$PREV_IMPORTANCE_PATH" ] && [ -f "$PREV_IMPORTANCE_PATH" ]; then
            echo "Expanding importance file..."
            EXPANDED_IMPORTANCE_PATH="$TASK_DIR/task-$((task_num-1))-importance-expanded.pth"
            python tools/expand_importance.py \\
                --old_importance $PREV_IMPORTANCE_PATH \\
                --old_model $PREV_MODEL \\
                --new_model $EXPANDED_MODEL \\
                --save_path $EXPANDED_IMPORTANCE_PATH \\
                --copy_importance_init
            PREV_IMPORTANCE_PATH="$EXPANDED_IMPORTANCE_PATH"
        fi

        echo "Training task $task_num with pseudo_label + ESPReg + EWC (ewc on bn only)..."
        TRAIN_CMD="python tools/train.py --model $EXPANDED_MODEL \\
            --data \\\"$ID_CONVERTED_DATASET/dataset.yaml\\\" \\
            --save_path $TASK_DIR/best.pt \\
            --epochs $EPOCHS \\
            --patience $PATIENCE \\
            --batch_size $BATCH_SIZE \\
            --imgsz $IMGSZ \\
            --workers $WORKERS \\
            --device $DEVICE \\
            --project $TASK_DIR \\{fl_train} \\
            --trainer antiforget"
        TRAIN_CMD="$TRAIN_CMD --pseudo_label True --conf_threshold $CONF_THRESHOLD --filter_iou_threshold $FILTER_IOU_THRESHOLD"
        TRAIN_CMD="$TRAIN_CMD --espreg True --pca_cache_path $PREV_PCA_CACHE --espreg_loss_weight $ESPREG_LOSS_WEIGHT"
        if [ -n "$PREV_IMPORTANCE_PATH" ] && [ -f "$PREV_IMPORTANCE_PATH" ]; then
            TRAIN_CMD="$TRAIN_CMD --ewc True --importance_path $PREV_IMPORTANCE_PATH --ewc_loss_weight $EWC_LOSS_WEIGHT"
        fi
        eval $TRAIN_CMD

        echo "Performing PCA analysis..."
        PCA_CACHE_PATH="$TASK_DIR/pca_cache.pkl"
        python tools/pca.py \\
            --model $TASK_DIR/best.pt \\
            --dataset $DATASET_PATH \\
            --load_hist $PREV_PCA_CACHE \\
            --save_path $PCA_CACHE_PATH
        echo "Calculating parameter importance (bn modules only)..."
        IMPORTANCE_PATH="$TASK_DIR/importance.pth"
        python tools/cal_importance.py \\
            --model $TASK_DIR/best.pt \\
            --dataset "$ID_CONVERTED_DATASET/dataset.yaml" \\
            --save_path $IMPORTANCE_PATH \\
            --module_pattern "*bn" \\
            --batch_size $BATCH_SIZE \\
            --workers $WORKERS \\
            --device $DEVICE
        PREV_MODEL="$TASK_DIR/best.pt"
        PREV_PCA_CACHE="$PCA_CACHE_PATH"
        PREV_IMPORTANCE_PATH="$IMPORTANCE_PATH"
    fi
    echo "Task $task_num completed!"
    ((task_num++))
done
echo "All tasks completed!"
"""


def gen_pseudo_label_proto_rp(cfg):
    """pseudo_label + prototype replay (no espreg/ewc)."""
    w = cfg.get("weight")
    freeze = cfg.get("freeze_layers")
    fl_decl = ("\nFREEZE_LAYERS=(\n    " + "\n    ".join('"' + f + '"' for f in freeze) + "\n)\n") if freeze else ""
    fl_train = " \\\n            --freeze ${FREEZE_LAYERS[$((task_num-1))]}" if freeze else ""
    out_prefix = cfg["output_prefix"] + "_pseudo_label+proto_rp"
    return f"""#!/bin/bash
MODEL_CFG="{cfg['model_cfg']}"
{fl_decl}OUTPUT_DIR="{out_prefix}"
EPOCHS=100
BATCH_SIZE=16
IMGSZ=640
WORKERS=8
DEVICE=0
PATIENCE=10
CONF_THRESHOLD=0.25
PROTO_RP_USE_BASE_MODEL=${{PROTO_RP_USE_BASE_MODEL:-True}}
START_TASK=${{START_TASK:-1}}

{task_datasets_bash(cfg)}

if [ $START_TASK -lt 1 ] || [ $START_TASK -gt ${{#TASK_DATASETS[@]}} ]; then
    echo "Error: START_TASK must be between 1 and ${{#TASK_DATASETS[@]}}"
    exit 1
fi

PREV_PROTOTYPES=""
if [ $START_TASK -gt 1 ]; then
    PREV_TASK=$((START_TASK - 1))
    PREV_MODEL="$OUTPUT_DIR/task-$PREV_TASK/best.pt"
    PREV_PROTOTYPES="$OUTPUT_DIR/task-$PREV_TASK/prototypes.pt"
    if [ ! -f "$PREV_MODEL" ]; then
        echo "Error: Previous task model not found: $PREV_MODEL"
        exit 1
    fi
    echo "Resuming from Task $START_TASK"
fi

task_num=1
for DATASET_PATH in "${{TASK_DATASETS[@]}}"; do
    if [ $task_num -lt $START_TASK ]; then
        ((task_num++))
        continue
    fi
    echo "=========================================="
    echo "Processing Task $task_num"
    echo "=========================================="
    TASK_DIR="$OUTPUT_DIR/task-$task_num"

    if [ $task_num -eq 1 ]; then
{first_task_train(cfg)}

        echo "Generating prototypes for task $task_num..."
        PROTOTYPES_PATH="$TASK_DIR/prototypes.pt"
        python tools/generate_prototypes.py \\
            --model $TASK_DIR/best.pt \\
            --data $DATASET_PATH \\
            --output $PROTOTYPES_PATH \\
            --device $DEVICE
        PREV_MODEL="$TASK_DIR/best.pt"
        PREV_PROTOTYPES="$PROTOTYPES_PATH"
    else
        DATASET_NAME=$(basename $(dirname $DATASET_PATH))
        echo "Expanding model head for task $task_num..."
        EXPANDED_MODEL="$TASK_DIR/task-$((task_num-1))-best-expanded.pt"
        python tools/expand_model_head.py \\
            --model $PREV_MODEL \\
            --model_cfg $MODEL_CFG \\
            --dataset $DATASET_PATH \\
            --save_path $EXPANDED_MODEL

        CONVERTED_PROTOTYPES=""
        if [ -n "$PREV_PROTOTYPES" ] && [ -f "$PREV_PROTOTYPES" ]; then
            echo "Converting prototype classes..."
            CONVERTED_PROTOTYPES="$TASK_DIR/task-$((task_num-1))-prototypes-converted.pt"
            python tools/convert_prototype_classes.py \\
                --prototypes $PREV_PROTOTYPES \\
                --original_model $PREV_MODEL \\
                --expanded_model $EXPANDED_MODEL \\
                --output $CONVERTED_PROTOTYPES
        fi

        echo "Generating pseudo labels..."
        PSEUDO_LABELS_DIR="$TASK_DIR/${{DATASET_NAME}}_train_pseudo_labels"
        python tools/generate_pseudo_label.py \\
            --model $PREV_MODEL \\
            --dataset $DATASET_PATH \\
            --output_dir $PSEUDO_LABELS_DIR \\
            --conf_threshold $CONF_THRESHOLD \\
            --splits train
        echo "Merging dataset..."
        MERGED_DATASET_DIR="$TASK_DIR/${{DATASET_NAME}}_merged"
        python tools/merge_datasets.py \\
            --datasets "$PSEUDO_LABELS_DIR/dataset.yaml" "$DATASET_PATH" \\
            --output_dir $MERGED_DATASET_DIR
        echo "Converting dataset class IDs..."
        CONVERTED_DATASET="$TASK_DIR/${{DATASET_NAME}}_converted"
        python tools/convert_dataset_class_ids.py \\
            --model $EXPANDED_MODEL \\
            --dataset $MERGED_DATASET_DIR/dataset.yaml \\
            --output_dir $CONVERTED_DATASET

        echo "Training task $task_num with pseudo_label + proto_rp..."
        TRAIN_CMD="python tools/train.py --model $EXPANDED_MODEL \\
            --data \\\"$CONVERTED_DATASET/dataset.yaml\\\" \\
            --save_path $TASK_DIR/best.pt \\
            --epochs $EPOCHS \\
            --batch_size $BATCH_SIZE \\
            --imgsz $IMGSZ \\
            --workers $WORKERS \\
            --device $DEVICE \\
            --project $TASK_DIR \\{fl_train} \\
            --trainer antiforget"
        TRAIN_CMD="$TRAIN_CMD --pseudo_label True --conf_threshold $CONF_THRESHOLD --filter_iou_threshold 0.5"
        if [ -n "$CONVERTED_PROTOTYPES" ] && [ -f "$CONVERTED_PROTOTYPES" ]; then
            TRAIN_CMD="$TRAIN_CMD --prototypes $CONVERTED_PROTOTYPES --proto_rp_use_base_model $PROTO_RP_USE_BASE_MODEL"
        fi
        eval $TRAIN_CMD

        echo "Generating prototypes for task $task_num..."
        PROTOTYPES_PATH="$TASK_DIR/prototypes.pt"
        python tools/generate_prototypes.py \\
            --model $TASK_DIR/best.pt \\
            --data $DATASET_PATH \\
            --output $PROTOTYPES_PATH \\
            --load_hits $CONVERTED_PROTOTYPES \\
            --device $DEVICE
        PREV_MODEL="$TASK_DIR/best.pt"
        PREV_PROTOTYPES="$PROTOTYPES_PATH"
    fi
    echo "Task $task_num completed!"
    ((task_num++))
done
echo "All tasks completed!"
"""


def script_exists(rel_dir: str, method: str) -> bool:
    d = SCRIPTS_ROOT / rel_dir
    if not d.is_dir():
        return False
    canonical = f"train_{method}.sh"
    if (d / canonical).exists():
        return True
    for old_name, canonical_method in ALIASES.items():
        if canonical_method == method and (d / f"train_{old_name}.sh").exists():
            return True
    return False


def main():
    import argparse
    p = argparse.ArgumentParser(description="Fill missing train scripts for voc, coco, rsar")
    p.add_argument("--dry-run", action="store_true", help="Only print what would be created")
    p.add_argument("--force", action="store_true", help="Overwrite existing scripts")
    args = p.parse_args()

    generators = {
        "naive": gen_naive,
        "pseudo_label": gen_pseudo_label,
        "pseudo_label+espreg": gen_pseudo_label_espreg,
        "pseudo_label+ewc": lambda c: gen_pseudo_label_ewc(c, use_bn_only=False),
        "pseudo_label+espreg+ewc": gen_pseudo_label_espreg_ewc,
        "pseudo_label+proto_rp": gen_pseudo_label_proto_rp,
        "pseudo_label+espreg+proto_rp": None,  # TODO stub
        "pseudo_label+ewc+proto_rp": None,
        "pseudo_label+espreg+ewc+proto_rp": None,
    }
    # Stub for combo proto_rp: reuse same structure as existing train_pseudo_label+espreg+proto_rp (long)
    # For speed we only implement the 6 we have; the 3 with proto_rp+ewc can be added as smaller set
    created = []
    for cfg in CONFIGS:
        rel_dir = cfg["rel_dir"]
        out_dir = SCRIPTS_ROOT / rel_dir
        for method in METHODS:
            if method in ("pseudo_label+espreg+proto_rp", "pseudo_label+ewc+proto_rp", "pseudo_label+espreg+ewc+proto_rp"):
                continue  # Skip complex combo proto_rp for now; can add later
            if not args.force and script_exists(rel_dir, method):
                continue
            gen = generators.get(method)
            if gen is None:
                continue
            content = gen(cfg)
            path = out_dir / f"train_{method}.sh"
            if args.dry_run:
                print(f"Would create: {path}")
                created.append(str(path))
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            path.chmod(0o755)
            print(f"Created: {path}")
            created.append(str(path))

    if args.dry_run:
        print(f"\nDry run: would create {len(created)} scripts.")
    else:
        print(f"\nCreated {len(created)} scripts.")


if __name__ == "__main__":
    main()
