from ultralytics import YOLO, YOLOE
from ultralytics.models.yolo.yoloe import YOLOEPETrainer

# Load a pretrained segmentation model
model = YOLOE("yoloe-v8l.yaml")
model.load("yoloe-v8l-seg.pt")
# model = YOLO("yolov8l.yaml")

# Identify the head layer index
head_index = len(model.model.model) - 1

# Freeze all backbone and neck layers (i.e. everything before the head)
freeze = [str(i) for i in range(0, head_index)]

# Freeze parts of the segmentation head, keeping only the classification branch trainable
# for name, child in model.model.model[-1].named_children():
#     if "cv3" not in name:
#         freeze.append(f"{head_index}.{name}")

# Freeze detection branch components
# freeze.extend(
#     [
#         f"{head_index}.cv3.0.0",
#         f"{head_index}.cv3.0.1",
#         f"{head_index}.cv3.1.0",
#         f"{head_index}.cv3.1.1",
#         f"{head_index}.cv3.2.0",
#         f"{head_index}.cv3.2.1",
#     ]
# )

# Train only the classification branch
results = model.train(
    data="data/4-domain/clipart/dataset.yaml",  # Segmentation dataset
    epochs=100,
    # patience=10,
    trainer=YOLOEPETrainer,  # <- Important: use segmentation trainer
    freeze=freeze,
)