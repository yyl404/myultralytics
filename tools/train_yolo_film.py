import warnings
warnings.filterwarnings("ignore")
from ultralytics.models.film import YOLOFiLM
from ultralytics import YOLO

def main():
    # 1. 准备数据配置 (data.yaml)
    # 指定 yaml 配置文件路径
    data_yaml_path = "/root/myultralytics/data/VOC-YOLO-Small-Json/dataset.yaml"

    # 2. 初始化模型-
    # 我们加载标准的 yolov8n.yaml 结构，YOLOFiLM 会自动把它包装成 DetectionModelFiLM
    # model = YOLOFiLM("yolov8l.yaml") 
    model = YOLOFiLM("/root/myultralytics/runs/film_experiment/exp113/weights/best.pt")
    # model_yoloe = YOLO("yoloe-v8l-seg.pt")
    # model.load("yoloe-v8l-seg.pt")

    # 3. 开始训练
    # 使用指定的 yaml 配置文件
    # model.train(
    #     data=data_yaml_path, 
    #     epochs=50, 
    #     imgsz=640,
    #     batch=4,
    #     project="runs/film_experiment",
    #     name="exp1"
    # )
    model.val(
        data=data_yaml_path,
        project="runs/film_experiment"
    )

if __name__ == '__main__':
    main()