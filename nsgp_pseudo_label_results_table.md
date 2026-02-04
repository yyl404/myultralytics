# NSGP+Pseudo Label 评估结果统计表

## 结果汇总

本表格展示了 NSGP+Pseudo Label 方法在三种不同划分（10+10、15+5、19+1）下的评估结果，包括旧类（Old Classes）、新类（New Classes）和总体（All Classes）的 mAP50 和 mAP50-95 指标。

### 评估结果表格

| 划分 | 指标 | 旧类 (Old) | 新类 (New) | 总体 (All) |
|------|------|------------|------------|------------|
| **10+10** | mAP50 | 0.6331 | 0.6699 | 0.6515 |
| | mAP50-95 | 0.4651 | 0.4893 | 0.4772 |
| **15+5** | mAP50 | 0.6458 | 0.4659 | 0.6008 |
| | mAP50-95 | 0.4680 | 0.3502 | 0.4385 |
| **19+1** | mAP50 | 0.7031 | 0.4973 | 0.6929 |
| | mAP50-95 | 0.5283 | 0.3810 | 0.5209 |

### 说明

- **10+10**: Task 1 包含前10个类别（aeroplane 到 cow），Task 2 包含后10个类别（diningtable 到 tvmonitor）
- **15+5**: Task 1 包含前15个类别（aeroplane 到 person），Task 2 包含后5个类别（pottedplant 到 tvmonitor）
- **19+1**: Task 1 包含前19个类别（aeroplane 到 train），Task 2 包含最后1个类别（tvmonitor）

所有结果均来自 Task 2 模型在累积数据集（包含所有已见类别）上的评估结果。

### 数据来源

结果文件位置：
- `runs/yolov8l_voc_10_10_fromscratch_pseudo_label+nsgp/evaluation_results/model_2_eval_cumulative.csv`
- `runs/yolov8l_voc_15_5_fromscratch_pseudo_label+nsgp/evaluation_results/model_2_eval_cumulative.csv`
- `runs/yolov8l_voc_19_1_fromscratch_pseudo_label+nsgp/evaluation_results/model_2_eval_cumulative.csv`

