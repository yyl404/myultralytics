import argparse
import os
import shutil

from ultralytics.utils import YAML
from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DistillationDetectionTrainer

from incremental_utils import expand_detection_head, create_classes_expanded_dataset, create_pseudo_labels_dataset
from osr_utils import copy_paste_replay, mix_up_augmentation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--checkpoint", type=str)

    args = parser.parse_args()
    cfg = YAML.load(args.cfg)
    save_dir = args.save_dir
    checkpoint = args.checkpoint

    if checkpoint is not None:
        model = YOLO(checkpoint)
        new_classes = cfg["new_classes"] if cfg["new_classes"] is not None else []
        all_classes = model.names

        new_class_id_map = {} # new_class_id -> all_class_id
        for i, cls in enumerate(new_classes):
            if cls in all_classes:
                new_class_id_map[i] = all_classes.index(cls)
        
        base_class_id_map = {} # base_class_id -> all_class_id
        for i, cls in enumerate(base_classes):
            if cls in all_classes:
                base_class_id_map[i] = all_classes.index(cls)

        if base_model is not None:
            training_dataset_cfg = f"{save_dir}/training_dataset_expanded/dataconfig.yaml"
            if not os.path.exists(training_dataset_cfg):
                Warning(f"Resumed training dataset {training_dataset_cfg} does not exist, recreating...")
                if os.path.exists(os.path.join(save_dir, "training_dataset_expanded")):
                    shutil.rmtree(os.path.join(save_dir, "training_dataset_expanded"))
                create_classes_expanded_dataset(data_cfg, new_class_id_map, save_dir, f"training_dataset_expanded", all_classes)
            model_teacher = YOLO(base_model).model
            model.train(trainer=DistillationDetectionTrainer, data=training_dataset_cfg, epochs=cfg["epochs"], batch=cfg["batch"],
                        workers=cfg["workers"], device=cfg["device"], project=save_dir, teacher_model=model_teacher)
        else:
            training_dataset_cfg = f"{save_dir}/training_dataset_with_copy_paste_replay/dataconfig.yaml"
            if not os.path.exists(training_dataset_cfg):
                Warning(f"Resumed training dataset {training_dataset_cfg} does not exist, recreating...")
                # # 先进行mix-up增强
                # if os.path.exists(os.path.join(save_dir, "training_dataset_with_mixed_up_samples")):
                #     shutil.rmtree(os.path.join(save_dir, "training_dataset_with_mixed_up_samples"))
                # os.makedirs(os.path.join(save_dir, "training_dataset_with_mixed_up_samples", "images", "train"))
                # mix_up_augmentation(data_cfg, cfg["memory_bank_dir"], os.path.join(save_dir, "training_dataset_with_mixed_up_samples", "images", "train"),
                #                     "train", num_generations=cfg["num_mixup"])
                # # 进行伪标签生成
                # shutil.copytree(os.path.dirname(data_cfg), os.path.join(save_dir, "training_dataset_with_mixed_up_samples"), dirs_exist_ok=True)
                # if os.path.exists(os.path.join(save_dir, "training_dataset_with_pseudo_labels")):
                #     shutil.rmtree(os.path.join(save_dir, "training_dataset_with_pseudo_labels"))
                # create_pseudo_labels_dataset(model, base_class_id_map, new_class_id_map, 
                #                              os.path.join(save_dir, "training_dataset_with_mixed_up_samples", os.path.basename(data_cfg)),
                #                              os.path.join(save_dir, "training_dataset_with_pseudo_labels"), all_classes, conf_threshold=0.25)
                # # 进行copy-paste回放
                # if os.path.exists(os.path.join(save_dir, "training_dataset_with_copy_paste_replay")):
                #     shutil.rmtree(os.path.join(save_dir, "training_dataset_with_copy_paste_replay"))
                # copy_paste_replay(os.path.join(save_dir, "training_dataset_with_pseudo_labels", "dataconfig.yaml"),
                #                   os.path.join(save_dir, "osr_memory_bank"),
                #                   os.path.join(save_dir, "training_dataset_with_copy_paste_replay"),
                #                   "train")
                if os.path.exists(os.path.join(save_dir, "training_dataset_with_pseudo_labels")):
                    shutil.rmtree(os.path.join(save_dir, "training_dataset_with_pseudo_labels"))
                create_pseudo_labels_dataset(model, base_class_id_map, new_class_id_map, 
                                             data_cfg,
                                             os.path.join(save_dir, "training_dataset_with_pseudo_labels"), all_classes, conf_threshold=0.25)
            model.train(data=training_dataset_cfg, epochs=cfg["epochs"], batch=cfg["batch"],
                        workers=cfg["workers"], device=cfg["device"], project=save_dir, resume=True)
        model.save(os.path.join(save_dir, "best.pt"))
        return
    
    base_model = cfg["base_model"]
    model_cfg = cfg["model_cfg"]
    model = YOLO(base_model) if base_model is not None else YOLO(model_cfg)
    model_name = model_cfg.split("/")[-1].split(".")[0]

    if base_model is not None:
        base_classes = [model.names[i] for i in sorted(model.names.keys())]
    else:
        base_classes = []

    new_classes = cfg["new_classes"] if cfg["new_classes"] is not None else []
    all_classes = list(set(base_classes).union(new_classes))

    model_channel_map = {}
    for i, cls in model.names.items():
        if cls in all_classes:
            model_channel_map[i] = all_classes.index(cls)

    if base_model is not None:
        expand_detection_head(base_model, model_cfg, model_channel_map, all_classes, save_dir, f"{model_name}_expanded.pt")
    else:
        os.makedirs(save_dir, exist_ok=True)
        model.save(os.path.join(save_dir, f"{model_name}_expanded.pt"))

    new_class_id_map = {} # new_class_id -> all_class_id
    for i, cls in enumerate(new_classes):
        if cls in all_classes:
            new_class_id_map[i] = all_classes.index(cls)
    
    base_class_id_map = {} # base_class_id -> all_class_id
    for i, cls in enumerate(base_classes):
        if cls in all_classes:
            base_class_id_map[i] = all_classes.index(cls)
    
    data_cfg = cfg["data_cfg"]
    
    if base_model is None:
        # 如果base_model为None，说明是训练第一个任务，不需要生成伪标签，直接创建数据集
        if os.path.exists(os.path.join(save_dir, "training_dataset_expanded")):
            shutil.rmtree(os.path.join(save_dir, "training_dataset_expanded"))
        create_classes_expanded_dataset(data_cfg, new_class_id_map, save_dir, f"training_dataset_expanded", all_classes)
        training_dataset_cfg = f"{save_dir}/training_dataset_expanded/dataconfig.yaml"
    else:
        # # 先进行mix-up增强
        # if os.path.exists(os.path.join(save_dir, "training_dataset_with_mixed_up_samples")):
        #     shutil.rmtree(os.path.join(save_dir, "training_dataset_with_mixed_up_samples"))
        # os.makedirs(os.path.join(save_dir, "training_dataset_with_mixed_up_samples", "images", "train"))
        # mix_up_augmentation(data_cfg, cfg["memory_bank_dir"], os.path.join(save_dir, "training_dataset_with_mixed_up_samples", "images", "train"),
        #                     "train", num_generations=cfg["num_mixup"])
        # # 进行伪标签生成
        # shutil.copytree(os.path.dirname(data_cfg), os.path.join(save_dir, "training_dataset_with_mixed_up_samples"), dirs_exist_ok=True)
        # if os.path.exists(os.path.join(save_dir, "training_dataset_with_pseudo_labels")):
        #     shutil.rmtree(os.path.join(save_dir, "training_dataset_with_pseudo_labels"))
        # create_pseudo_labels_dataset(model, base_class_id_map, new_class_id_map, 
        #                              os.path.join(save_dir, "training_dataset_with_mixed_up_samples", os.path.basename(data_cfg)),
        #                              os.path.join(save_dir, "training_dataset_with_pseudo_labels"), all_classes, conf_threshold=0.25)
        # # 进行copy-paste回放
        # if os.path.exists(os.path.join(save_dir, "training_dataset_with_copy_paste_replay")):
        #     shutil.rmtree(os.path.join(save_dir, "training_dataset_with_copy_paste_replay"))
        # copy_paste_replay(os.path.join(save_dir, "training_dataset_with_pseudo_labels", "dataconfig.yaml"),
        #                   os.path.join(save_dir, "osr_memory_bank"),
        #                   os.path.join(save_dir, "training_dataset_with_copy_paste_replay"),
        #                   "train")
        # 进行伪标签生成
        if os.path.exists(os.path.join(save_dir, "training_dataset_with_pseudo_labels")):
            shutil.rmtree(os.path.join(save_dir, "training_dataset_with_pseudo_labels"))
        create_pseudo_labels_dataset(model, base_class_id_map, new_class_id_map, 
                                     data_cfg,
                                     os.path.join(save_dir, "training_dataset_with_pseudo_labels"), all_classes, conf_threshold=0.25)
        training_dataset_cfg = os.path.join(save_dir, "training_dataset_with_pseudo_labels", "dataconfig.yaml")

    model = YOLO(f"{save_dir}/{model_name}_expanded.pt")
    
    if base_model is not None:
        model_teacher = YOLO(base_model).model
        model.train(trainer=DistillationDetectionTrainer, data=training_dataset_cfg, epochs=cfg["epochs"], batch=cfg["batch"],
                    workers=cfg["workers"], device=cfg["device"], project=save_dir, teacher_model=model_teacher, freeze=[i for i in range(9)])
    else:
        model.train(data=training_dataset_cfg, epochs=cfg["epochs"], batch=cfg["batch"],
                    workers=cfg["workers"], device=cfg["device"], project=save_dir)

    model.save(os.path.join(save_dir, "best.pt"))


if __name__ == "__main__":
    main()