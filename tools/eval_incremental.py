import argparse

from ultralytics.utils import YAML
from ultralytics import YOLO

from incremental_utils import create_classes_expanded_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_cfg", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda:0")

    args = parser.parse_args()
    data_cfg = args.data_cfg
    save_dir = args.save_dir
    model_path = args.model_path
    batch = args.batch
    workers = args.workers
    device = args.device

    model = YOLO(model_path)
    all_classes = model.names
    eval_classes = YAML.load(data_cfg)["names"]

    class_id_map = {} # evaluated_class_id -> all_class_id
    for i, cls in eval_classes.items():
        if cls in all_classes.values():
            # 根据value查找all_classes字典中的key
            for key, value in all_classes.items():
                if value == cls:
                    class_id_map[i] = key
                    break
    
    create_classes_expanded_dataset(data_cfg, class_id_map, save_dir, f"eval_dataset_expanded", all_classes)
    
    results = model.val(data=f"{save_dir}/eval_dataset_expanded/dataconfig.yaml", batch=batch,
                        workers=workers, device=device, project=save_dir)
    