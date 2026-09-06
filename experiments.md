# 2026/09/06

截至目前，已经实现并且确认有效的IOD方法包括：伪标签（pseudo_label）, 蒸馏（Distillation），ESPReg 和 样本回放（Replay）。回放样本的挑选方法采用的是随机挑选。

在 VOC TINY 和 VOC 上基于 yolo26m，采用 yoloe-26m-seg.pt 预训练权重的实验结果如下（mAP50，task-agnostic）

VOC-TINY
|Method|old (15)|new (5)|all (20)|
|---|---|---|---|
|joint|-|-|-|
|dist|
|pseudo_label|
|pseudo_label+dist|58.0|65.1|59.8|
|pseudo_label+espreg|58.9|70.6|61.8|
|pseudo_label+espreg+dist|70.9|69.7|70.6|
|pseudo_label+espreg+replay|74.2|76.9|74.9|
|pseudo_label+espreg+dist+replay|74.0|67.6|72.4|

VOC
|Method|old (15)|new (5)|all (20)|
|---|---|---|---|
|joint|1.7|64.9|17.5|
|pseudo_label+espreg|74.7|74.1|74.5|
|pseudo_label+espreg+dist|83.6|74.3|81.2|
|pseudo_label+espreg+replay|83.1|76.8|81.6|
|pseudo_label+espreg+dist+replay|87.0|75.2|84.1|

超参数列表：

训练与解码（VOC / VOC-TINY 共用，来自 `scripts/voc/{tiny,all}/15+5/common.sh` 与各 run 的 `task-*/train/args.yaml`）

| 项 | 值 |
|---|---|
| 模型 / 预训练 | yolo26m / `yoloe-26m-seg.pt` |
| split | 15+5（CIL） |
| epochs | 10 |
| batch / imgsz / device / workers | 16 / 640 / 0 / 8 |
| optimizer / lr0 / lrf | AdamW / 0.001 / 0.01 |
| weight_decay / momentum | 0.0005 / 0.937 |
| warmup_epochs / warmup_bias_lr | 3.0 / 0.0 |
| mosaic / close_mosaic | 0.5 / 10 |
| freeze | 10（冻结前 10 层） |
| seed / deterministic | 0 / true |
| end2end / agnostic_nms / max_det | False / True / 300 |

IOD 组件（仅增量阶段 task-2 生效；未启用的组件为默认关闭）

| 项 | 值 |
|---|---|
| pseudo_label | conf_threshold=0.25，filter_iou_threshold=0.5 |
| dist | dist_loss_weight=100.0，dist_topk=1 |
| espreg | espreg_loss_weight=100.0（PCA cache 来自上一任务） |
| replay | 随机挑选，REPLAY_SAMPLE_NUM=100，replay_loss_weight=1.0，seed=0 |
