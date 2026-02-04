# Incremental Learning Script Templates

本目录说明增量学习脚本的模板层级，便于在不同数据集、初始权重和方法上复用。

## 方法名称（统一）

- **EWC**: Elastic Weight Consolidation
- **ESPReg**: Eigen-value Scaled Projection Regularization（原 EWPR/VSPReg，已统一为 ESPReg）
- **Pseudo Label**: 伪标签
- **Prototype Replay**: 原型回放（ProtoRP）

## 模板层级

1. **初级模板**（`_base_incremental.sh`）：通用结构
   - 变量：`MODEL_CFG`, `OUTPUT_DIR`, `EPOCHS`, `BATCH_SIZE`, `IMGSZ`, `WORKERS`, `DEVICE`, `PATIENCE`, `SAVE_PERIOD`
   - `START_TASK`, `TASK_DATASETS`
   - 任务循环：校验 START_TASK、遍历 TASK_DATASETS、task_num 递增
   - 第一任务 vs 后续任务分支（扩展头、转换 ID、训练）

2. **次级模板**（按方法与初始权重）：
   - **Naive**: 无正则，仅扩展头 + 转换 ID + 训练
   - **Pseudo Label**: 使用 `--trainer antiforget --pseudo_label True`
   - **ESPReg**: 首任务后做 PCA，后续任务 `--espreg True --pca_cache_path ... --espreg_loss_weight ...`
   - **ESPReg + Pseudo Label**: 上述两者组合
   - **EWC**: 首任务后算 importance，后续 `--ewc True --importance_path ...`
   - **Pseudo Label + ESPReg + EWC**: 三者组合（如 rsar/3_3/yoloe）
   - **Prototype Replay**: 需先 generate_prototypes，训练时 `--proto_rp True --prototypes ...`
   - 初始权重：from scratch（无 `--weight`）或 pretrained（`--weight <yoloe.pt>` 等）

3. **数据集适配**：在对应 `scripts/<dataset>/<split>/<backbone>/` 下，设置 `TASK_DATASETS` 和 `OUTPUT_DIR` 等。

## 方法名顺序（脚本与输出目录统一）

组合方法时按以下顺序书写：**pseudo_label → espreg → ewc → proto_rp → sample_rp**（未使用的方法不写）。

- 单方法：`train_naive.sh`、`train_pseudo_label.sh`、`train_espreg.sh`、`train_ewc.sh`、`train_proto_rp.sh`
- 组合：`train_pseudo_label+espreg.sh`、`train_pseudo_label+ewc.sh`、`train_pseudo_label+espreg+ewc.sh`、`train_espreg+proto_rp.sh`、`train_pseudo_label+espreg+proto_rp.sh`、`train_pseudo_label+espreg+proto_rp+sample_rp.sh`
- 说明：**ESPReg 与 EWC 同时使用时**，EWC 仅对名称以 `bn` 结尾的模块生效（脚本内 `cal_importance.py` 使用 `--module_pattern "*bn"`）；单独使用 EWC 时无此限制。

输出目录 `OUTPUT_DIR` 与脚本名对应，例如 `runs/..._pseudo_label+espreg`、`runs/..._pseudo_label+espreg+ewc`。

## 使用示例

从初级模板复制后，按数据集修改 `TASK_DATASETS` 和 `OUTPUT_DIR`，再按方法选择是否添加 PCA/importance/原型/伪标签等逻辑。
