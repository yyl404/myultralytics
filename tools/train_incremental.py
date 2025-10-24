import argparse
import os
import os.path as OSP
import shutil

from ultralytics.utils import YAML
from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import VSPRegDetectionTrainer

from incremental_utils import expand_detection_head, create_classes_expanded_dataset, create_pseudo_labels_dataset, build_vae_replay_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--checkpoint", type=str)
    # VAE 回放相关可选参数
    parser.add_argument("--vae_ckpt", type=str, default=None, help="VAE 权重路径，如 /root/vae-search/logs/best.pt")
    parser.add_argument("--vae_arch", type=str, default="vq", choices=["vq", "vanilla"], help="VAE 架构类型")
    parser.add_argument("--replay_source_images", type=str, default=None, help="用于回放构建的源图像目录")
    parser.add_argument("--replay_enable", action="store_true", help="启用基于VAE的生成式回放")
    parser.add_argument("--replay_conf_threshold", type=float, default=0.25)

    args = parser.parse_args()
    cfg = YAML.load(args.cfg)
    save_dir = args.save_dir
    checkpoint = args.checkpoint
    
    base_model_path = cfg["base_model"]
    model_cfg = cfg["model_cfg"]
    model_name = model_cfg.split("/")[-1].split(".")[0]

    data_cfg_path = cfg["data_cfg"]

    if base_model_path is not None:
        base_model = YOLO(base_model_path)
        base_classes = [base_model.names[i] for i in sorted(base_model.names.keys())]
        
        data_cfg = YAML.load(data_cfg_path)
        new_classes = [data_cfg["names"][i] for i in sorted(data_cfg["names"].keys())]

        all_classes = list(set(base_classes).union(new_classes))
        
        base_class_id_map = {}
        for i, cls in enumerate(base_classes):
            base_class_id_map[i] = all_classes.index(cls)

        new_class_id_map = {}
        for i, cls in enumerate(new_classes):
            new_class_id_map[i] = all_classes.index(cls)

        if checkpoint is not None:
            if not OSP.exists(OSP.join(save_dir, "training_dataset_with_pseudo_labels", "dataconfig.yaml")):
                print("Warning: checkpoint is provided, but training dataset with pseudo labels is not found. Recreating...")
                if OSP.exists(OSP.join(save_dir, "training_dataset_with_pseudo_labels")):
                    shutil.rmtree(OSP.join(save_dir, "training_dataset_with_pseudo_labels"))
                create_pseudo_labels_dataset(base_model, base_class_id_map, new_class_id_map, 
                                             data_cfg_path,
                                             OSP.join(save_dir, "training_dataset_with_pseudo_labels"), all_classes)
            model = YOLO(checkpoint)
            resume = True
        else:
            expand_detection_head(base_model_path, model_cfg, base_class_id_map, all_classes, save_dir, f"{model_name}_expanded.pt")
            if OSP.exists(OSP.join(save_dir, "training_dataset_with_pseudo_labels")):
                shutil.rmtree(OSP.join(save_dir, "training_dataset_with_pseudo_labels"))
            create_pseudo_labels_dataset(base_model, base_class_id_map, new_class_id_map, 
                                         data_cfg_path,
                                         OSP.join(save_dir, "training_dataset_with_pseudo_labels"), all_classes)
            model = YOLO(f"{save_dir}/{model_name}_expanded.pt")
            resume = False
        
        pca_sample_images = cfg["pca_sample_images"]
        pca_sample_labels = cfg["pca_sample_labels"]
        pca_cache_save_path = cfg["pca_cache_save_path"]
        pca_cache_load_path = cfg["pca_cache_load_path"]
        teacher_model = YOLO(cfg["teacher_model"])

        # 如果启用回放，则先基于 VAE 生成回放数据集，并替换 data 路径
        data_yaml_path = OSP.join(save_dir, "training_dataset_with_pseudo_labels", "dataconfig.yaml")
        if args.replay_enable and args.vae_ckpt and args.replay_source_images:
            data_yaml_path = build_vae_replay_dataset(
                base_model=teacher_model,
                base_class_id_map=base_class_id_map,
                new_class_id_map=new_class_id_map,
                cfg_source=data_cfg_path,
                save_dir=save_dir,
                target_dataset_classes=all_classes,
                vae_ckpt=args.vae_ckpt,
                replay_source_images=args.replay_source_images,
                arch=args.vae_arch,
                conf_threshold=args.replay_conf_threshold,
                device=str(cfg.get("device", args.device)) if hasattr(args, "device") else "cuda",
            )

        model.train(data=data_yaml_path, epochs=cfg["epochs"], batch=cfg["batch"], amp=False,
                    workers=cfg["workers"], device=cfg["device"], project=save_dir, freeze=cfg["frozen_layers"],
                    trainer=VSPRegDetectionTrainer,
                    teacher_model=teacher_model,
                    sample_images=pca_sample_images,
                    sample_labels=pca_sample_labels,
                    pca_sample_num=cfg["pca_sample_num"], projection_layers=cfg["projection_layers"],
                    pca_cache_save_path=pca_cache_save_path,
                    pca_cache_load_path=pca_cache_load_path,
                    resume=resume)
    else:
        if checkpoint is not None:
            model = YOLO(checkpoint)
            resume = True
        else:
            model = YOLO(model_cfg)
            resume = False
        model.train(data=data_cfg_path, epochs=cfg["epochs"], batch=cfg["batch"], amp=False,
                    workers=cfg["workers"], device=cfg["device"], project=save_dir,
                    freeze=cfg["frozen_layers"], resume=resume)
    
    model.save(OSP.join(save_dir, "best.pt"))


if __name__ == "__main__":
    main()