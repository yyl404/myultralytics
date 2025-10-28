from ultralytics import YOLO
from ultralytics.nn.modules import DoubleConv
from torch.nn import functional as F

compressed_model = YOLO("runs/yolov8l_voc_inc_15_5_fromscratch/task-1/best_compressed.pt").eval()
ft_model = YOLO("runs/yolov8l_voc_inc_15_5_fromscratch/finetune-task-1-compress/train/weights/best.pt").eval()

print(ft_model.model.model)

for (name, module_base), (name, module_ft) in zip(compressed_model.model.named_modules(), ft_model.model.named_modules()):
    assert type(module_base) == type(module_ft), f"{name} type mismatch, given {type(module_base)} and {type(module_ft)}"
    if isinstance(module_base, DoubleConv):
        weight_compress_base = module_base.conv_1.weight.data
        weight_compress_ft = module_ft.conv_compress.weight.data

        weight_expand_base = module_base.conv_2.weight.data
        weight_expand_ft = module_ft.conv_expand.weight.data

        loss = F.mse_loss(weight_compress_base, weight_compress_ft) / 2.0 + F.mse_loss(weight_expand_base, weight_expand_ft) / 2.0
        print(f"{name} weight loss: {loss.item()}")