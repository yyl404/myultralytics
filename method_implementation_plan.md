# method_plan.md 的实现改动方案（待批准）

本文只描述改动，不含已落地代码。目标方法由三个模块组成，映射为三个 `--method` 组件：

| 方案模块 | 组件名 | 依赖工件 | 依赖教师 |
|---|---|---|---|
| §7 原型可靠性感知 YOLO-LwF 蒸馏 | `prad`（`--lwf True --lwf_mode prad`） | `patch_prototypes.pt` | 上一任务模型（`lwf_teacher_model`） |
| §8 基于局部 patch 原型分布采样的全类别回放蒸馏 | `prd` | `patch_prototypes.pt` | 上一任务模型（复用同一教师） |
| §9 特征值缩放投影正则化 ESPReg | `espreg`（新增权重模式/损失形式开关） | `pca_cache.pkl`（现有） | 上一任务模型（现有） |

推荐方法串：`--method prad+prd+espreg`。§5 固定共享 PCA 与 §6 多中心原型不单独成为组件，它们是 `prad`/`prd` 共用的工件内容。
`dist`（CAD）、`lwf`（yolo_lwf）、`replay`、`repre` 等现有组件保持不变，继续作为方案 §14.2 的对照基线。

---

## 1. 现有代码可复用的部分（先复用，避免新造）

| 需求 | 已有实现 | 复用方式 |
|---|---|---|
| 教师加载、decode 一致性校验、`reg_max` 校验 | `anti_forget.py::_setup_train` 的 LwF 段、`_apply_train_head_args`、`_check_teacher_decode_alignment` | `lwf_mode == "prad"` 直接走同一段教师初始化 |
| 检测头输入特征（`X_s`） | `Detect.forward_head` 返回 `{"boxes","scores","feats"}`，`feats` 即 `X_s` | 不再额外注册 hook，直接从 preds dict 取 |
| 每 batch 教师前向 | `get_model_raw_output` / `_lwf_teacher_output` | 复用 |
| 6 通道/类 DFL 判别、`reg_max==1` 直接回归 | `head.reg_max`、`lwf._decode_boxes` | 复用 |
| 5×5 检测头输入 patch 提取与"边界补零还原"逻辑 | `tools/generate_prototypes.py::extract_pos_patches_from_layer`、`repre._restore_prototype_patches` | 新工件工具沿用同样的 5×5 语义（`r_s` 由头结构推导，见方案 §4.2） |
| 增量 PCA | `tools/pca_on_gpu.py::IncrementalPCAonGPU`（`partial_fit`/`components_`/`explained_variance_`） | 新工件的固定共享 PCA 直接用它拟合，`B_s` 与现有 `pca_cache.pkl` 同构（`mean_` + `components_`） |
| 旧类原型 GT 位置统计的执行框架 | `tools/compute_importance.py`（trainer + callback 的数据管线） | 新工件工具用 `build_yolo_dataset(mode="val")` + `build_dataloader` 取 letterbox 批次与 GT，不自己重写坐标变换 |
| 正样本分配（α）与前景掩码 | `v8DetectionLoss.assigner`（`TaskAlignedAssigner`，返回 `fg_mask/target_gt_idx/target_scores`） | 用**子类化**的 `v8DetectionLoss` 截获分配结果（不改官方文件，见 §2.1/§3.3） |
| ESPReg 主流程 | `espreg.EWPRegLoss` | 只加权重模式与损失形式开关 |
| 方案 §14.1 的失效指标 1/2 | `tools/teacher_error_analysis.py`、`tools/incremental_error_analysis.py` | 已存在，不改 |

---

## 2. 新增文件

### 2.1 `ultralytics/engine/patch_proto.py`（工件侧共享设施）

与 Ultralytics 对象强耦合（直接操作 `Detect` 的 `cv2/cv3` 与特征图），按仓库规则放在 `ultralytics/engine/`。

模块级函数：

- `head_patch_radius(head) -> list[int]`
  逐尺度推导 `r_s = Σ_l d_l(k_l-1)/2`，统计 `cv2[i]` 与 `cv3[i]` 内所有 `stride==1` 的 `Conv2d`，取 `max(r_cls, r_reg)`。yolo26m/v8 头为两层 3×3 → `r_s=2`（5×5）。要求所有尺度 `r_s` 相同，否则报错（跨尺度混合 patch 尺寸会让工件语义混乱）。
- `extract_patches(feats, level, img_idx, rows, cols, radius) -> Tensor`
  用 `F.pad(feats[level], (r,r,r,r)).unfold(2,2r+1,1).unfold(3,2r+1,1)`（视图，不复制）再按选中位置高级索引，返回 `(K, C, 2r+1, 2r+1)`。只对候选位置取 patch，禁止对全图 unfold（8400×C×25 会爆显存）。
- `valid_patch_positions(rows, cols, h, w, radius) -> BoolTensor`
  方案 §4.3：越界位置直接跳过，不做 padding 采样。
- `assigned_targets(preds, batch, criterion) -> (fg_mask, target_gt_idx, target_scores)`
  统一入口，供 `prad`（新类掩码）与工件工具（代表性位置）共用；内部调用 `criterion` 的子类截获结果，不重复跑分配器。

类：

- `RecordingDetectionLoss(v8DetectionLoss)`
  只覆写 `get_assigned_targets_and_loss`：先调父类，再把返回元组里的 `(fg_mask, target_gt_idx, target_scores)` 缓存到 `self.last_assignment` 后原样返回。**不修改** `ultralytics/utils/loss.py`（官方资产），也不额外跑一次分配器。trainer 与工件工具都用它替换默认 criterion。
- `HeadPatchSubspace`
  每个尺度一组固定共享 PCA：`mean (D_s,)`、`components (d_s, D_s)`（非白化，`h = (p-mean) @ B.T`）。
  - `encode(patches) -> (K, d_s)`，`decode(h) -> (K, C_s, 2r+1, 2r+1)`
  - `components is None` 表示方案 §8.5 的无压缩回退（`encode/decode` 为恒等映射，`d_s = D_s`），调用方不需要分支。
  - `save/load`（`torch.save`），并做 `D_s`、`d_s`、`r_s` 与当前模型的一致性校验。
- `MultiCenterBank`
  每尺度保存 `class_ids (N,)`、`means (N,d_s)`、`var (N,d_s)`（对角）、`weight (N,)`（类内 π 归一化）。
  - `fit_class(samples, tau_new, n_min, max_centers, merge_ratio, iters, seed) -> (means, var, weight)`
    带最小簇约束的 DP-Means：单中心起步 → 按最近中心赋值 → 半径超过 `tau_new` 的簇在最远点分裂 → 重算中心 → 合并距离 < `merge_ratio*tau_new` 的中心 → 丢弃样本数 < `n_min` 的簇 → 中心数超过 `max_centers` 时保留样本数最多的 `max_centers` 个 → 重算 `μ/σ²/π`。固定 seed，确定性。
  - `append(new_class_entries)`：只追加新类，旧类原样保留（方案 §11.2 第 13–14 步）。
  - `sample(level, generator, std_scale) -> h~ (K, d_s)`：类均匀 → 类内 `Categorical(π)` → `μ + std_scale*σ⊙ε`。
  - `max_cosine(h, level, class_ids) -> r`：与指定类所有局部原型的最大余弦相似度。

### 2.2 `ultralytics/engine/prad.py`（模块 1）

`class PrototypeReliabilityLwFLoss`：

```python
def __call__(self, student, teacher, new_class_mask, stride) -> (cls_term, reg_term)
# student/teacher: Detect 输出 dict（boxes/scores/feats）
# new_class_mask: (B, A) bool，a_k^new
# cls_term = L_cls^F + η_B * L_cls^B ；reg_term = L_reg^KD
```

内部（全部在 `torch.no_grad()` 之外按需建图，只有涉及学生 logits/回归的项带梯度）：

1. `q_k = max_{c<NC_old} σ(z^T_{k,c})`，`c*_k = argmax`。
2. 候选：`M^F_k = [q_k ≥ τ_F][not a^new_k]`、`M^B_k = [q_k ≤ τ_B][not a^new_k]`；`M^F` 候选数超过 `prad_max_fg` 时按 `q_k` 取 top（安全阀，正常 batch 远小于该值，见 §5 实测 `≈2/图`）。
3. 对 `M^F` 位置取教师 `X_s` patch → `encode` → `r_k = max_m cos(h_k, μ_{c*_k,m})` → `g_k = sigmoid((r_k - τ_proto)/T_proto)`；该类在该尺度无原型时 `g_k = 0` 并只告警一次（不静默）。
4. `A_k = clamp(IoU(b̂^T, b̂^S), 0, 1)`（detach）；`w^F_k = M^F_k q_k^γ g_k A_k`，`w^B_k = M^B_k (1-q_k)^γ`。
5. 逐类 `BKL(p^T||p^S)`（float32 + eps 截断，按方案原式保留 `p log(p/q)` 的常数项，使日志数值与方案一致）。
6. 前景/背景分别归一化（方案 §7.4/§7.5 的分母），回归按 `w^F` 归一化（方案 §7.6）。`Σw` 为 0 时返回同 device 零标量。`prad_split_normalize=False` 时前景/背景共用同一个分母（方案 §14.2 第 2 项消融），其余不变。

### 2.3 `ultralytics/engine/prd.py`（模块 2）

`class PatchReplayDistillation`：

```python
def compute_loss(self) -> (cls_term, reg_term)   # 每步重新采样
```

流程（方案 §8.1–8.7）：

1. 每尺度采样 `prd_samples` 个 `h~`；`decode` 成 `(K, C_s, 5, 5)`。
2. 教师头、学生头**各自**对 patch 前向，取中心位置 `[2,2]` 输出：`z^T_old (K,NC_old)`、`d̂^T (K,4)`、`z^S_all (K,NC_all)`、`d̂^S (K,4)`。
   - 教师全程 `no_grad`。
   - 只对 `cv2[i]/cv3[i]` 前向，不经过 backbone/neck。
   - 两个头在 patch 前向期间切 `eval()`（BN 用运行统计，避免用合成 patch 污染 BN 统计），`finally` 中恢复原 `training` 状态（沿 `repre.py` 的既有做法）。
3. 教师全类别目标：`c < NC_old` 用 `σ(z^T_old,c)`，`c ≥ NC_old` 用概率 0（**不写 logit 0**，方案 §13.6）；`BKL` 逐类求和，分母 `|V_all|·|R|`。
4. `reg_term = SmoothL1(d̂^S, d̂^T)`，分母 `|R|`。
5. 某尺度无原型（空类空间）时跳过该尺度；全部为空时返回零标量并告警一次。

### 2.4 `tools/generate_patch_prototypes.py`（工件生成 + 方案 §8.5/§14.1 报告）

一次调用完成"采集 → （基础阶段）拟合共享 PCA → 投影 → 多中心原型 → 落盘"，并可选输出重建偏差与相似度报告。

CLI：

```
--model <task-k best.pt> --data <本任务 dataset.yaml> --output <task-k/patch_prototypes.pt>
--device --imgsz 640 --batch_size 8 --workers 8 --seed 0
--pca_dims 128            # 可写成 128,128,256 逐尺度覆盖
--pca_energy_ratio 0.0    # >0 时按累计解释方差自动定 d_s（上限 pca_dims）
--no_compression          # 方案 §8.5 回退：不保存 B_s，h=p
--patch_radius -1         # -1=按头结构自动；0=单向量原型（消融 6）
--covariance diag         # diag | lowrank(预留，本版不实现)
--var_shrinkage 0.1       # 对角方差向类内池化方差收缩
--tau_new_ratio 0.5       # τ_new = ratio × 该类到类均值的距离中位数
--min_centers 1 --max_centers 8 --min_cluster 5 --merge_ratio 0.5 --iters 20
--max_samples_per_class 256
--center_ratio 0.5        # 候选位置限制在 GT 中心区域
--load_hist <上一任务 patch_prototypes.pt>
--report_dims 32 64 128 256   # 输出重建偏差 CSV
--report_similarity           # 输出 r_k 统计（用于定 τ_proto）
```

采集规则：

- `build_yolo_dataset(mode="val")`（letterbox、无增强）→ 同时拿到 `img` 与 `bboxes/cls/batch_idx`，坐标已是模型输入空间。
- 用 `TaskAlignedAssigner` 的分配结果，对每个 GT 在每个尺度取其被分配位置中 `target_scores`（即归一化 α）最大者作为代表性位置（方案 §4.3），并要求该位置落在 GT 中心 `center_ratio` 区域内。
- 每个 (类, 尺度) 最多保留 `max_samples_per_class` 个 patch 参与聚类；PCA 拟合用 `IncrementalPCAonGPU(n_components=d_s)` 流式累积（避免把 ~4.5 万×12800 的矩阵读进显存）。
- 类别 id 按**类名**映射到当前模型的类别空间（沿 `generate_prototypes.py` 的做法），映射失败直接报错而不是跳过。

工件格式（`patch_prototypes.pt`）：

```python
{
  "version": 1,
  "patch_radius": [2, 2, 2],
  "pca": [{"mean": (D_s,), "components": (d_s, D_s) or None}, ...],
  "levels": [{"class": (N,), "means": (N,d_s), "var": (N,d_s), "weight": (N,)}, ...],
  "nc": int, "task": int, "covariance": "diag", "class_names": {id: name},
  "stats": {"num_centers": {...}, "artifact_bytes": int, "patch_count": {...}},
}
```

输出：逐尺度/逐类的中心数、工件体积（方案 §12 的存储开销）、`--report_dims` 的 `E_feat/E_cls/E_reg` CSV、`--report_similarity` 的 `r_k` 分位数。

---

## 3. 修改文件

### 3.1 `ultralytics/cfg/default.yaml`

在现有 anti-forget 段追加（全部为新增键，不删除任何现有键）：

```yaml
# patch-prototype artifact shared by prad / prd
patch_prototypes: None          # (str) 上一任务导出的固定共享 PCA + 多中心原型工件
# prad (prototype-reliability-aware YOLO-LwF distillation, lwf_mode=prad)
prad_fg_threshold: 0.5          # τ_F
prad_bg_threshold: 0.1          # τ_B
prad_conf_power: 1.0            # γ
prad_proto_threshold: 0.5       # τ_proto
prad_proto_temperature: 0.1     # T_proto
prad_bg_weight: 1.0             # η_B
prad_cls_loss_weight: 4.0       # λ_KD
prad_reg_loss_weight: 0.5       # λ_KD·λ_reg
prad_max_fg: 512                # 单步前景候选上限（安全阀）
prad_split_normalize: True      # 前景/背景分别归一化（方案 §7.5）；False 时共用分母（消融 2）
# prd (patch replay distillation)
prd: False
prd_samples: 64                 # 每尺度每步采样 patch 数
prd_std_scale: 1.0
prd_loss_weight: 1.0            # λ_PRD
prd_reg_loss_weight: 0.5        # λ_PRD·λ_PRD-reg
# espreg additions
espreg_weight_mode: nt          # raw | nt
espreg_form: mean_norm          # mean_norm(现状) | frobenius(方案 §9.3)
espreg_scope: all               # all | backbone（方案 §14.2 第 7 项消融）
```

### 3.2 `ultralytics/engine/espreg.py`

- `EWPRegLoss.__init__(..., weight_mode="nt", form="mean_norm", scope_names=None)`：
  - `weight_mode=="raw"` 时用 `self.eigen_values` 原值、跳过 `adjust_eigen_values`（省一次肘点计算）；`"nt"` 保持现状。两者使用**同一份**中心化 PCA 缓存，不做非中心化处理（D2 已定）。
  - `form=="frobenius"` 时按方案 §9.3 的严格二次型：`α_l * ‖(ΔW U) diag(√ω)‖_F²`（即 `Tr(ΔW U diag(ω) U^T ΔW^T)`）；`"mean_norm"` 保持现有 `norm(dim=2).mean()`。
  - `scope_names` 由 trainer 传入（`backbone` 时用现有 `_get_nsgp_backbone_names` 过滤 `module_names`）。
  - 两种模式共用同一接口，只改变 ω 与归约方式。

### 3.3 `ultralytics/engine/anti_forget.py`

不新增文件级重复逻辑，只做接线：

1. 顶部增加 `from ultralytics.engine.patch_proto import ...`、`prad`、`prd`。
2. `_anti_forget_loss_names()`：`lwf_mode=="prad"` 时追加 `("prad_cls_loss","prad_bg_loss","prad_reg_loss")`；`prd` 时追加 `("prd_cls_loss","prd_reg_loss")`。顺序与 `_do_train` 追加顺序一致。
3. `_setup_train`（LwF 段内）：
   - 教师加载条件由 `self.args.lwf` 放宽为 `self.args.lwf or self.args.prd`（`prad`/`prd` 都必须以 `lwf_teacher_model` 指向上一个任务模型），加载、`_apply_train_head_args`、`_check_teacher_decode_alignment`、`reg_max` 一致性、`nc` 递增校验全部沿用；`YOLOLwFLoss` 只在 `lwf_mode in {"yolo_lwf","lwf"}` 且启用 `lwf` 时构造。
   - `lwf_mode` 合法值增加 `"prad"`；`prad`/`prd` 任一开启时必须提供 `patch_prototypes`，且 `reg_max == 1`、`Detect` 头、`end2end == False`，否则 `ValueError/TypeError` 明确报错（OBB 直接拒绝）。
   - 加载工件 → 构造 `HeadPatchSubspace` + `MultiCenterBank`；`prad` 再构造 `PrototypeReliabilityLwFLoss`，`prd` 再构造 `PatchReplayDistillation(student_head, teacher_head, subspace, bank, nc_old, ...)`（两者共用同一份工件对象，只读一次文件）。校验：`工件 nc == teacher nc`、`工件 r_s == head_patch_radius`、`d_s`/`D_s` 与当前模型通道一致、工件类 id 全在 `student nc` 内。
   - `prad` 时 `unwrap_model(self.model).criterion = _RecordingDetectionLoss(unwrap_model(self.model))`（见下），用于取分配结果；`model.end2end` 为真直接拒绝。
4. `RecordingDetectionLoss`（定义在 `patch_proto.py`，见 §2.1）替换默认 criterion 以缓存分配结果。
5. `_do_train`：
   - `lwf_mode=="prad"` 时走独立块（不经过 `_lwf_terms`）：计算 `a^new_k` → `prad` 三项 → `self.loss += w_cls*(L^F+η_B L^B) + w_reg*L_reg`，**不乘 batch size**（与 `dist/espreg/repre` 一致；现有 `lwf` 分支的 `×batch` 保持不动，因此 `prad_*` 权重与 `lwf_*` 权重不可互换，故 prad 使用独立权重键）。
   - `prd` 独立块：`self.loss += λ_PRD*L_PRD_cls + λ_PRD-reg*L_PRD_reg`。
   - `lwf_mode=="prad"` 与 `replay_mode in {distill,both}` 互斥（`ValueError`），因为 prad 只在当前 batch 上定义；图像回放不属于本方案。`prd` 与 `replay_mode=="supervised"` 可以并存（两套回放独立加权），实现上不需要额外分支。
   - 现有 `lwf` 分支的进入条件加 `and self.args.lwf_mode != "prad"`。

### 3.4 `scripts/model_adapters/ultralytics.sh`

- `model_adapter_check_method_components`：允许 `prad`、`prd`。
- `method_needs_patch_prototypes() { method_has prad || method_has prd; }`；`PREVIOUS_PATCH_PROTOTYPES="${previous_task_dir}/patch_prototypes.pt"`。
- `model_adapter_initialize`：新增环境变量（默认值取 §5 推荐值）
  `PRAD_FG_THRESHOLD / PRAD_BG_THRESHOLD / PRAD_CONF_POWER / PRAD_PROTO_THRESHOLD / PRAD_PROTO_TEMPERATURE / PRAD_BG_WEIGHT / PRAD_CLS_LOSS_WEIGHT / PRAD_REG_LOSS_WEIGHT /`
  `PRD_SAMPLES / PRD_STD_SCALE / PRD_LOSS_WEIGHT / PRD_REG_LOSS_WEIGHT /`
  `ESPREG_WEIGHT_MODE / ESPREG_FORM / ESPREG_SCOPE /`
  `PATCH_PCA_DIMS / PATCH_PROTO_MAX_CENTERS / PATCH_PROTO_TAU_RATIO / PATCH_PROTO_MAX_SAMPLES`。
- task>1 的 `TRAINER_ARGS` 增加：
  `--lwf True --lwf_mode prad --lwf_teacher_model $PREVIOUS_MODEL --patch_prototypes $PREVIOUS_PATCH_PROTOTYPES --prad_* ...`
  `--prd True --patch_prototypes $PREVIOUS_PATCH_PROTOTYPES --prd_* ...`
  `--espreg True ... --espreg_weight_mode/--espreg_form/--espreg_scope`（`espreg` 组件已接线，只加三个新 flag）。
- `model_adapter_finalize_task`：`method_needs_patch_prototypes` 时
  `python tools/generate_patch_prototypes.py --model $PREVIOUS_MODEL --data $TRAIN_DATA --output ${TASK_DIR}/patch_prototypes.pt ... --load_hist $PREVIOUS_PATCH_PROTOTYPES`
  （task-1 无 `--load_hist` 且拟合 PCA；task>1 复用固定 PCA，只追加新类）。
- **不**为 `patch_prototypes.pt` 提供"缺失即重建"：固定共享 PCA 只能在基础阶段产生，缺失时直接报错并提示重跑该 run（与 importance/pca 的自动重建策略不同，理由写进注释）。
- `espreg` 工件生成保持现状（中心化、`--exclude_head`），不新增 `uncentered` 分支：`raw` 与 `nt` 共用同一份缓存，只差 ω（D2）。`raw` 的 ω 直接取 `explained_variance_`（中心化协方差特征值），与方案 §9.1 的非中心二阶矩相差一个秩 1 项（`μμᵀ`），属已接受的近似，在代码注释里写清。

### 3.5 `scripts/libexec/experiment.sh`

`experiment_known_method_components` 追加 `prad prd`（一行）。

### 3.6 不改动的部分（明确边界）

`scripts/train.sh`、`eval.sh`、`predict.sh`、`create.sh`、`run_incremental.sh`、`tools/train.py`、`tools/eval.py`、`tools/predict.py`、`tools/expand_model_head.py`、`ultralytics/utils/loss.py`、validator/predictor、推理路径均不改。方案 §12.4 要求推理只保留学生模型，本方案天然满足（工件只被 trainer 读取）。

---

## 4. 关键实现要点（易错处）

1. **patch 半径**由头结构推导，不写死 5×5；`r_s` 与工件不一致时 trainer 直接报错。
2. **只对候选位置取 patch**：`feats[l].unfold(...)` 是视图，索引后才复制；禁止全图 unfold。
3. **BN 处理**：`prd` 的 patch 前向必须让 `cv2[i]/cv3[i]` 处于 `eval()` 并用 `finally` 恢复；否则合成 patch 的统计会污染检测头 BN，且训练/推理不一致。
4. **`reg_max > 1` 直接拒绝**（方案 §13.12，直接回归头才用 SmoothL1），v8/v11 头（`reg_max=16`）不支持。
5. **新类目标在概率域为 0**：`BKL(0‖q) = -log(1-q)`，禁止在 logit 域写 0（方案 §13.6）。
6. **DDP**：新增项不含跨 rank 集合通信；采样用各 rank 独立 RNG（不同 rank 采不同 patch，天然互补）；patch 前向只覆盖 `cv2/cv3`，这些参数在主前向的 det loss 中始终参与构图，与现有"同一步多次前向"（replay/BPF）模式一致，`find_unused_parameters=True` 已开启。新增 loss 一律 float32 计算，`eps` 截断。
7. **空集是一等公民**：`Σw=0`、某尺度无原型、某类无原型都返回带正确 device/dtype 的零标量并（只）告警一次，不返回 `None`。
8. **工件只读**：trainer 不写工件；工件在 `finalize_task` 生成，与 `importance/pca/repre` 的时序一致。
9. **`espreg_weight_mode=raw`**：ω 取中心化 PCA 缓存的 `explained_variance_` 原值（不做非中心化、不做肘点归一化），因此与方案 §9.1 的 `E[xxᵀ]` 谱相差一个秩 1 项 `μμᵀ`；`λ_ESP` 必须单独校准。

---

## 5. 超参数、推荐取值与确定方法

### 5.1 方案公式已固定的量（不开放）

| 量 | 取值 | 来源 |
|---|---|---|
| 前景分类/回归权重 | `w^F = M^F q^γ g · IoU` | §7.3/§7.4 |
| 背景权重 | `w^B = M^B (1-q)^γ` | §7.3 |
| 前景/背景归一化 | 各自按 `|V_old|·Σw` 归一 | §7.4/§7.5 |
| 新类教师目标 | 概率域 0 | §8.4 |
| 回归蒸馏 | SmoothL1（直接回归头） | §7.6/§8.7 |
| 原型位置 | 检测头输入 `X_s` 的 `(2r+1)²` patch | §4.2 |
| PCA | 每尺度一组、基础阶段固定、非白化 | §5 |
| 类内分布 | 多中心 + 对角协方差 + π | §6.2 |

### 5.2 需要选值的超参数（含推荐值与确定方法）

**模块 1（prad）**

| 参数 | 符号 | 推荐默认 | 确定方法 |
|---|---|---|---|
| `prad_fg_threshold` | τ_F | **0.5** | 在任务 t 训练图像上用教师前向统计 `q_k`：旧类 GT 命中位置的分位数（实测中位数 0.78–0.82）与背景位置的分位数（实测 p99≈0）；τ_F 取两者之间的量级，默认 0.5（约保留教师最可靠的那部分前景）。也可直接用 `tools/teacher_error_analysis.py` 的旧类置信度统计确定 |
| `prad_bg_threshold` | τ_B | **0.1** | 统计非 GT 位置的 `q_k` 的 p99（实测 ≈0，即教师对背景几乎不给旧类概率）；必须 `< τ_F`，代码校验 |
| `prad_conf_power` | γ | **1.0** | 线性即方案默认；消融 0.5 / 2.0 |
| `prad_proto_threshold` | τ_proto | **0.5** | 由 `tools/generate_patch_prototypes.py --report_similarity` 给出两簇 `r_k`：基础类 GT 位置（应 ≈0.8–1.0）与"教师把新类误判为旧类"的位置（应 ≈0.2–0.5），取分界中点 |
| `prad_proto_temperature` | T_proto | **0.1** | 取 `r_k` 过渡带宽度量级（两簇间标准差的 1/2）；固定后不再调 |
| `prad_bg_weight` | η_B | **1.0** | 前景/背景已各自归一，量级可比即可；实测背景项 `L^B≈6e-5`（远小于 `L^F`），该项是"防止学生在新域背景上激活旧类"的预防项，若日志显示 `prad_cls_bg` 长期 ≈0 而 Old mAP 掉，再上调到 10 |
| `prad_cls_loss_weight` | λ_KD | **4.0** | 见 §5.4 实测：`L^F` 末态 0.12–0.68；×4 ≈ 0.5–2.7，与现有 yolo_lwf 的加权量级（0.47）以及 `L_det≈1.2` 匹配 |
| `prad_reg_loss_weight` | λ_KD·λ_reg | **0.5** | 实测 `L_reg^KD` 末态 0.14；×0.5≈0.07，约为现有 lwf 回归项（0.033）的两倍 |
| `lwf_mask_old` | — | **True** | 新类监督只落在本任务类通道（现有语义） |
| `prad_max_fg` | — | **512** | 安全阀；实测前景候选 ≈2/图（bs=8 时约 18 个），正常远不触发 |

**模块 2（prd）**

| 参数 | 符号 | 推荐默认 | 确定方法 |
|---|---|---|---|
| `prd_samples` | \|R\|/尺度 | **64** | 每步 3×64=192 个 5×5 patch，头前向开销 <5% 步时；显存与批次无关 |
| `prd_std_scale` | — | **1.0** | 严格按 §8.1 采样；>1 增大覆盖但偏离原型 |
| `prd_loss_weight` | λ_PRD | **1.0** | 实测全类别 BKL 项 0.39（旧类 patch 上）；×1≈0.39≈0.3×`L_det`，作为主模块合理；任务开始时新类通道随机初始化会把它顶到 ~0.5–0.7，随训练衰减 |
| `prd_reg_loss_weight` | λ_PRD·λ_PRD-reg | **0.5** | 实测 SmoothL1 末态 0.41；×0.5≈0.2（≈15% `L_det`），保证定位不被回放项淹没 |
| 采样分配 | — | 尺度均分、类均匀、类内按 π | 类均匀可避免长尾类被淹没；π 保留类内多模态比例 |

**模块 3（espreg，新增开关）**

| 参数 | 推荐 | 说明 |
|---|---|---|
| `espreg_loss_weight` | **100**（现状） | `form=frobenius` 后量级变化，需按日志 `espreg_loss` 重新校准：令加权后 ≈0.05–0.1×`L_det` |
| `espreg_weight_mode` | **nt**（本研究建议同时跑 raw 对照） | §9.3 开关；同一份工件、只改 ω；两种模式分别校准 λ_ESP（raw 的 λ 未归一化，λ_ESP 可能需小 1–3 个数量级） |
| `espreg_form` | **frobenius**（新方法），`mean_norm` 保留为历史基线 | 见决策 D1 |
| `espreg_scope` | **all**（方案 §14.2 第 7 项消融：backbone） | 用 `_get_nsgp_backbone_names` 过滤 |

**工件构建（不影响训练步时，只影响工件质量/体积）**

| 参数 | 推荐默认 | 确定方法 |
|---|---|---|
| `--pca_dims` d_s | **128** | 见 §5.3 实测：d=128 时 `E_cls ≤ 2.8e-2`（每类平均 ≤2e-3）、`E_reg ≤ 0.025`，而需要纠正的学生-教师回归漂移实测 ≈0.41，PCA 引入的偏差约 6%。d=64 时 `E_reg≈0.055`（13%），d=256 时 ≈0.01。逐尺度可用 `--pca_dims 128,128,256` 覆盖（P5 最敏感） |
| `--pca_energy_ratio` | 0（不用） | 备选：按累计解释方差（0.95）自动定 d，上限 `pca_dims`。128 维在实测中仅覆盖 0.75–0.85 能量，故"按能量定维"容易给出过小/不稳定的 d，降级为可选项 |
| `--patch_radius` | -1（自动，=2） | 消融 6 用 0（单向量原型） |
| `--covariance` | **diag** | 方案 §6.2 基础实现 |
| `--var_shrinkage` | **0.1** | 小簇（`min_cluster=5`）下对角方差按 `(1-ρ)C_m + ρ·C_pooled` 收缩 |
| `--tau_new_ratio` | **0.5** | `τ_new = 0.5 × median_b‖h_b − μ_c‖`，自动适配每个 (类,尺度) 的尺度；工具打印每类中心数，期望 1–4（>6 需下调 ratio 或收紧 `max_centers`） |
| `--min_cluster` / `--min_centers` / `--max_centers` | 5 / 1 / 8 | 噪声簇剔除、至少 1 中心、存储预算上限（8 中心 × 20 类 × 3 尺度 × (2·128+1) float ≈ 0.5 MB） |
| `--merge_ratio` | 0.5 | 合并阈值 = `0.5·τ_new` |
| `--max_samples_per_class` | **256** | PCA 用流式 `IncrementalPCA`，该上限只影响聚类与采样精度 |
| `--center_ratio` | 0.5 | 方案 §4.3"限制在 GT 中心区域"，抑制边缘/背景 patch |
| `--seed` | 0 | 工件确定性；PRD 采样在训练时由各 rank RNG 产生 |

存储估算（yolo26m，d=128）：PCA 工件 ≈16.4 MB（`D_s=6400/12800/12800`），原型 ≈0.5 MB；d=64 时 ≈8.2 MB。每个任务目录各存一份（累计 ×任务数），大文件仍放 `/hy-tmp`。

### 5.3 PCA 可行性实测（yolo26m，VOC 15+5 基础模型、任务 2 图像，每尺度 ≈1000 个实例 patch）

| 尺度 | d=64 `E_cls` / `E_reg` | d=128 | d=256 | d=384 |
|---|---|---|---|---|
| P3 (D=6400) | 2.1e-3 / 0.055 | 1.3e-3 / 0.025 | 7.4e-4 / 0.010 | 3.7e-4 / 0.005 |
| P4 (D=12800) | 1.3e-2 / 0.051 | 8.3e-3 / 0.025 | 3.3e-3 / 0.009 | 1.6e-3 / 0.004 |
| P5 (D=12800) | 4.2e-2 / 0.060 | 2.8e-2 / 0.025 | 1.7e-2 / 0.007 | 9.1e-3 / 0.003 |

（`E_feat` 在 d=128 时 0.29–0.40，只作解释性指标。对照量级：需要纠正的师生回归漂移 SmoothL1≈0.41、旧类 BKL 每类 ≈0.1–1。）结论：d=128 的 PCA 近似可行，偏差比回放要纠正的漂移小一个量级；正式实验仍按方案 §8.5 用 `--report_dims` 在**基础阶段**输出逐尺度/逐类报告后再定稿 d_s。

### 5.4 权重标定的实测依据（yolo26m，VOC，bs=8，任务 2 训练图像）

| 量 | 5+5+5+5（LwF 学生） | 15+5（naive 学生） | 现有 lwf 基线 |
|---|---|---|---|
| `L^F`（按前景/类数归一） | 0.123 | 0.682 | — |
| `L^B` | 6e-5 | 9e-5 | — |
| `L_reg^KD` | 0.143 | 1.034 | — |
| 现有 lwf cls（全元素均值） | 1.2e-4 | 9.2e-4 | ×4000 → **0.47** |
| 现有 lwf reg | 7e-5 | 1.3e-3 | ×500 → **0.033** |
| `L_PRD^cls`（全类别 BKL 均值，真实旧类 patch） | 0.393 | 0.123 | — |
| `L_PRD^reg`（SmoothL1，真实旧类 patch） | 0.410 | 0.165 | — |
| `L_det`（`results.csv`：box+cls+l1） | ≈1.2 | ≈1.2 | ≈1.2 |

要点：新归一化（只在前景/背景内部求加权均值）使同名的分类项比现有 lwf 的"全元素均值"大约 3 个数量级，**所以 prad 必须使用独立权重键**（不能沿用 4000/500）。上表 → §5.2 的 λ 推荐值。

标定流程（每次结构/数据集变化后执行一次）：
1. 用推荐值跑 1–2 个 epoch，读 `results.csv` 的 `train/prad_cls_loss`、`train/prad_bg_loss`、`train/prad_reg_loss`、`train/prd_cls_loss`、`train/prd_reg_loss`、`train/espreg_loss` 与 `train/box_loss+cls_loss+l1_loss`。
2. 目标区间：KD/PRD 分类项加权后 ∈[0.2,0.5]×`L_det`；KD/PRD 回归项 ∈[0.05,0.15]×`L_det`；ESP ∈[0.05,0.1]×`L_det`。按比例线性缩放对应权重（各权重之间近似独立，因为每项都自带归一化）。
3. raw 与 NT 两种 ESPReg 模式分别校准 λ_ESP。

---

## 6. 与方案 §14.2 消融的对应（全部靠配置，不复制代码）

| 消融 | 配置 |
|---|---|
| 1 无原型匹配的 YOLO-LwF | `--method lwf`（现有 yolo_lwf） |
| 2 有原型匹配但不分前景/背景 | `prad_split_normalize False`（前景/背景共用分母）；去掉原型匹配则再加 `prad_proto_threshold -1`（g≡1） |
| 3 仅分类回放 | `prd_reg_loss_weight 0` |
| 4 分类+回归回放 | 默认 |
| 5 单中心 vs 多中心 | 工件 `--max_centers 1` vs 默认 |
| 6 单向量 vs patch 原型 | 工件 `--patch_radius 0` vs 默认 |
| 7 无ESPReg / 仅骨干 / 骨干+颈部 | `--method` 不带 `espreg` / `ESPREG_SCOPE=backbone` / `all` |
| 8 raw vs NT | `ESPREG_WEIGHT_MODE=raw|nt`（各自校准 λ_ESP） |
| 9 完整 vs 逐模块移除 | `--method prd+prad+espreg` 及去掉任一项 |

---

## 7. 验证计划

1. **工件工具**：VOC-TINY 15+5 task-1 上跑一次，检查中心数分布（1–4/类/尺度）、工件体积、`--report_dims` 报告；task-2 追加后检查旧类条目字节级不变（只追加）。
2. **单任务 smoke**：`EPOCHS=1 bash scripts/voc/tiny/15+5/pipeline.sh`（METHOD=prad+prd+espreg）跑通 task-1→task-2，检查 loss 曲线量级、`loss_items` 键齐全、日志无 NaN。
3. **数值一致性**：`prad` 在 `g≡1,η_B=0` 且 `τ_F=0,τ_B=-1` 时，其分类项应等于"全锚点全旧类通道 BKL 加权均值"，与手写参考实现对比（一次性断言脚本，不进仓库）。
4. **DDP**：本机只有 1 张 GPU，无法实跑 2 rank。代码层面保证：新增项无集合通信、参数总是同时出现在主路径图中、patch 前向沿用 `repre` 的 BN 切换模式；实施时补一条 2-GPU smoke 命令留在文档里，由有多卡的环境执行。
5. **推理无回归**：`bash scripts/eval.sh`/`predict.sh` 在带新方法的 run 上结果与 `naive` run 同流程（只读工件不参与推理）。

---

## 8. 决策记录

| # | 决策 | 状态 |
|---|---|---|
| D1 | ESPReg 损失形式 | **待定**（分析见 §8.1，建议采纳 `espreg_form` 开关，新方法用 `frobenius`） |
| D2 | 中心化 / 非中心化二阶矩 | **已定：不做区分**。`raw` 直接使用现有中心化 PCA 缓存的 `explained_variance_` 作为 ω，不新增 `uncentered` 分支 |
| D3 | `prad` 与图像回放 distill 组合 | **已定：互斥并报错** |
| D4 | 新类掩码 `a^new_k` 用 TAL 严格分配 | **已定**（TAL = Task-Aligned Assigner，见 §8.2） |
| D5 | `espreg_loss_weight` 重校准 | 随 D1 决定；选 `frobenius` 则按 §5.4 重校，选 `mean_norm` 则沿用 100 |
| D6 | 更新 `skills/scripts_structure_skill.md` | **已授权**，改动清单见 §8.3 |

### 8.1 D1：两种 ESPReg 形式的差异（逐项）

现有实现（`espreg.py::get_loss`，逐模块）：

```python
loss += 100 * ((delta_w @ proj.transpose(1, 2)) * scale.unsqueeze(1)).norm(dim=2).mean()
loss = loss / len(self.module_names)
# proj[k] = u_k，scale[k] = ω_k，delta_w[o] = ΔW_o（第 o 个输出通道，含 bias 列）
# ⇒ 每个模块 (100/|L|) · mean_{g,o} sqrt( Σ_k ω_k² (ΔW_{g,o}·u_{g,k})² )
```

方案 §9.3（逐模块）：

```
Σ_o Σ_k ω_k (ΔW_o·u_k)²  （= ‖ΔW U diag(√ω)‖_F² = Tr(ΔW U diag(ω) Uᵀ ΔWᵀ)）
```

四处不等价，按重要性排序：

1. **ω 与 ω²**：现有形式的 `scale` 在 `sqrt` **内部**，所以单个方向对损失的"能量权重"是 `ω_k²`（对振幅的权重才是 `ω_k`）；方案要求能量权重就是 `ω_k`。`raw` 模式下 `ω=λ` 而 λ 跨好几个数量级，`λ²` 会把谱的头部/尾部差异平方放大，于是"raw = 局部响应扰动恒等式"不再成立（这是审稿意见 2.1 的核心）。`nt` 模式下 `ω=min(λ/λ_r,1)∈(0,1]`，`ω²` 更狠地压低尾部，但仍在同一单调族内，所以只是量变。
2. **开方（√·）与平方（·²）**：现有形式是"先对每个输出通道求 L2 范数、再对通道取均值"，属于 group-lasso 式的**范数**惩罚（在 0 处不可导、梯度模长恒定）；方案是 Frobenius **平方**（在 0 处光滑、梯度 ∝ 偏差）。即使令所有 `ω=1`，两者也只是最优点相同、优化轨迹不同：前者像恒定拉力的硬约束，后者像胡克弹簧。
3. **归约方式**：现有形式对 `(g, C_out/g)` 取**均值**（组和通道一起平均）再乘 `100/|L|`；方案对输出通道与组求和，只有 `α_l` 作为层间平衡。方案 §9.2 的恒等式在 `α_l=1` 时才严格是 `Σ_l E‖ΔW_l x_l‖²`，所以两者相差约 `C_out·|L|/100` 的常数因子 —— 这一项只影响 λ_ESP 标定（D5），不构成数学上的不等价。
4. **层间平衡的实际含义**：现有 `α_l ≡ 100/|L|` 是均匀的；方案允许逐层 `α_l`。两者都只影响层间相对权重，本版保持均匀（`100/|L|`，可整体被 λ_ESP 吸收），不引入逐层参数。

结论：`raw` 模式要能对应方案 §9.2 的推导，必须同时满足"平方归约 + 能量权重取 λ"。`mean_norm` 与推导之间不是等价变换，继续用它会重现审稿人 2.1 的质疑。

### 8.2 D4：TAL 是什么、用在哪两处

TAL = **Task-Aligned Assigner**（TOOD 提出的对齐式标签分配，本仓库实现为 `ultralytics/utils/tal.py::TaskAlignedAssigner`，训练时由 `v8DetectionLoss` 以 `topk=10, alpha=0.5, beta=6.0` 实例化）：

- 对每个 GT，用对齐分数 `α_{b,k} = s_{k,c_b}^{α} · IoU(b̂_k, b)^{β}` 给所有候选锚点打分（`s` 为学生/教师在该 GT 类别上的预测分数），再在候选集中按 `α` 取 top-k 并做中心先验/重叠过滤，得到该 GT 的正样本锚点集合 `Ω_s(b)`，并归一化成 `target_scores`（每个 GT 的 α 和为 1）；同时给出 `fg_mask`（哪些锚点被判为正）与 `target_gt_idx`（正样本属于哪个 GT）。
- 它是方案 §4.3 里 `α_{b,k}`（"标签分配器给出的对齐分数"）与 §7.3 里 `a_k^new`（"被当前新类 GT 明确分配"）的**同一来源**，所以两处共用一个机制：
  - 工件工具：对每个 GT、每个尺度，在被分配的锚点里取 `target_scores`（∝ α）最大者作为代表性位置 `k_{b,s}`（方案 §4.3）。
  - trainer：把 `fg_mask & (target_gt_idx 对应的 GT 类 ≥ NC_old)` 作为 `a^new_k`，用于把新类 GT 位置从旧类蒸馏里排除（方案 §7.3）。
- 复用方式：`RecordingDetectionLoss(v8DetectionLoss)` 子类化后缓存父类已经算好的 `(fg_mask, target_gt_idx, target_scores)`，**不重跑**分配器、不改官方文件（`ultralytics/utils/loss.py` 一行不动）。
- 备选方案（锚点落在新类 GT 框内）只能用近似 `a^new_k`，且拿不到 α，工件侧仍需另写一套 IoU 选点，因此不采用。

### 8.3 D6：`skills/scripts_structure_skill.md` 待改清单（已授权）

1. `## Method string` 表：新增 `prad`、`prd` 两行（Effect 列写明"原型可靠性感知 YOLO-LwF 蒸馏（需 patch_prototypes）"、"全类别 patch 回放蒸馏（需 patch_prototypes）"）。
2. `## Env knobs` 表：新增 `PRAD_*`（8 个）、`PRD_*`（4 个）、`ESPREG_WEIGHT_MODE / ESPREG_FORM / ESPREG_SCOPE`、`PATCH_*`（4 个）的默认值行；`ESPREG_LOSS_WEIGHT` 行补充"`espreg_form=frobenius` 时需重新校准"。
3. `## File roles` 表：`model_adapters/ultralytics.sh` 行补充 `patch_prototypes.pt` 工件；`scripts/` 目录树无需变化（无新脚本）。
4. `## Checklist: adding a new method` 无变化（本次正是按该清单落地）。

---

## 9. 实施顺序与工作量

| 阶段 | 内容 | 预估 |
|---|---|---|
| 1 | `patch_proto.py` + `tools/generate_patch_prototypes.py`（采集/PCA/DP-Means/落盘/报告） | 1.5 天 |
| 2 | `prad.py` + `prd.py` + `anti_forget.py` 接线 + `default.yaml` | 1 天 |
| 3 | `espreg.py` 三个开关（form/weight_mode/scope） | 0.5 天 |
| 4 | 适配器与 `experiment.sh` 接线、工件时序 | 0.5 天 |
| 5 | smoke 验证（工件 → 单任务 → 量级标定 → 双任务跑通） | 1 天 |

合计约 4.5 个工作日（不含多轮调参与正式实验）。

---

## 10. 实施记录（代码已落地）

### 10.1 实际改动

| 文件 | 内容 |
|---|---|
| `ultralytics/engine/patch_proto.py`（新） | patch 半径推导、按位置取 patch、`RecordingDetectionLoss`（子类化 `v8DetectionLoss`，用一个 assigner 代理记录 `fg_mask/target_gt_idx/target_labels/alpha`）、`HeadPatchSubspace`（固定共享 PCA，`components=None` 为无压缩回退）、`MultiCenterBank`（多中心 + 对角方差 + π + 采样 + 余弦匹配）、工件读写 |
| `ultralytics/engine/prad.py`（新） | `PrototypeReliabilityLwFLoss`（§7 全部公式，含 `prad_split_normalize` 消融开关） |
| `ultralytics/engine/prd.py`（新） | `PatchReplayDistillation`（§8，头分支临时 `eval()` 后恢复，概率域新类 0 目标） |
| `ultralytics/engine/espreg.py` | `EWPRegLoss` → **`ESPRegLoss`**；新增 `weight_mode∈{raw,nt}`、`form∈{original,frobenius}`；`frobenius` 为 `Σ_o Σ_k ω_k (ΔW_o·u_k)²`（即 `Tr(ΔW U diag(ω) Uᵀ ΔWᵀ)`），`original` 保留 `mean_o sqrt(Σ_k ω_k²(ΔW_o·u_k)²)`；模块缺失改为立即报错 |
| `ultralytics/engine/anti_forget.py` | 教师加载条件放宽到 `lwf or prd`；`_setup_patch_prototypes`；`prad`/`prd` 两条 loss 块；当前 batch 分配结果快照；`espreg_scope` 过滤；`lwf_mask_old` 对所有 lwf 模式生效 |
| `ultralytics/cfg/default.yaml` | 新增 `patch_prototypes`、`prad_*`(10)、`prd_*`(5)、`espreg_weight_mode/espreg_form/espreg_scope` |
| `tools/generate_patch_prototypes.py`（新） | 采集（TAL 代表性位置 + GT 中心区域）→ 增量 PCA → 多中心聚类 → 落盘；`--report_dims`（E_feat/E_cls/E_reg，按尺度/维度/类别）与 `--report_similarity`（r_k 分位数） |
| `scripts/model_adapters/ultralytics.sh` | `prad`/`prd` 组件、`method_needs_patch_prototypes`、工件时序（finalize 生成、缺失即报错）、19 个环境变量 |
| `scripts/libexec/experiment.sh` | 组件白名单 |
| `skills/scripts_structure_skill.md` | 组件表 +2 行、环境变量表 +10 行、file roles 补工件（D6 授权） |

### 10.2 与方案/初版计划的偏差（均已确认或实测）

1. **DP-Means 用 SSE 归约判定新中心**（而非 `tau_new` 距离阈值）：`λ = split_ratio × 类别总平方误差`，仅在二分带来的 SSE 下降超过 λ 时分裂。原因：按 `tau_new` 的距离阈值在真实 patch 分布上会过度分裂（11 个 patch 裂成 8 个中心）。默认 `split_ratio=0.2`（单峰→1 中心，双峰→2，三峰→3）。
2. **`prad`/`prd` 不乘 batch size**（与 `dist/espreg/repre` 一致），因此 `prad_*` 权重与 `lwf_*` 权重不可互换，故使用独立权重键。
3. **`_AssignmentRecorder` 必须屏蔽 dunder 属性代理**，否则 `copy.deepcopy`（EMA）会把 assigner 的 `__getstate__` 当成 recorder 的状态，导致验证阶段 criterion 损坏。

### 10.3 实测标定

| 量 | 实测 | 结论 |
|---|---|---|
| 原型余弦相似度（VOC 10+10 基础模型，d=128，1200 图） | L0 p25=0.409/中位 0.558；L1 p25=0.217/中位 0.395；L2 p25=0.364/中位 0.540 | `prad_proto_threshold` 由 0.5 下调为 **0.3**（0.5 会门控掉 40–75% 的分布内 patch） |
| ESPReg 四种组合在同一 ΔW（naive 任务末态）下的损失 | original/nt 43.7；original/raw 710；frobenius/nt 10566；frobenius/raw 138244 | 与现有 `espreg_loss_weight=100` 等价换算：frobenius/raw 需 ≈ **0.03**（按"同一 ΔW 下加权损失相同"标定） |
| **域外 patch 的原型相似度**（基础模型 + 基础工件，任务 2 图像，按教师置信度采样） | L0 中位 0.332 / p25 0.234；L1 中位 0.161 / p25 0.002；L2 中位 0.349 / p25 0.023 | 与分布内（上行）相比 L1 明显下移、L2 次之：τ_proto=0.3 正好落在两组之间，原型匹配确实能区分"教师在新类图像上高置信旧类响应" |
| **任务 2 真实数据探针**（VOC 10+10，基线 task-1 模型 + d=128 工件，3 epoch，与主训练并行） | ep1/ep2：`espreg` 64.5/140.8（加权 1.94/4.22）、`prad_cls` 0.078/0.069（×4 = 0.31/0.27）、`prad_reg` 0.057/0.096（×0.5）、`prd_cls` 0.008/0.017、`prd_reg` 0.003/0.006；L_det 4.34/2.96；mAP50 0.401→0.566 | 各分项量级正常；`espreg` 加权后与 L_det 同量级且与"历史 100 + original/nt"在同一 ΔW 下的压力等价（实测比值 3163×，0.03×140.8 ≈ 100×0.0445），新类学习正常上升，故保留 `λ_ESP=0.03` |
| `prad`/`prd` 首 epoch 量级（VOC-TINY 烟测） | prad_cls 0.36、prad_bg 2e-4、prad_reg 6.1、prd_cls 1.5e-3、prd_reg 0.084，L_det≈7.5 | KD/PRD 项在任务开始时接近 0，随学生漂移增长；权重按任务末态标定（§5.4），首 epoch 数值不作校准依据 |

### 10.4 主实验命令（原始特征值加权 + PCA 降维原型）

```bash
ESPREG_WEIGHT_MODE=raw ESPREG_FORM=frobenius ESPREG_SCOPE=all \
ESPREG_LOSS_WEIGHT=0.03 PRAD_PROTO_THRESHOLD=0.3 PATCH_PCA_DIMS=128 \
METHOD=prad+prd+espreg \
bash scripts/train.sh --tasks <task yamls> --model yolo26m --weights yolo26m-cls.pt \
    --method "$METHOD" --output runs/<run> -- \
    --optimizer AdamW --lr0 0.001 --warmup_bias_lr 0.0 --mosaic 0.5
```

### 10.4b 训练开销实测（VOC 10+10，task-2，bs16，328 iter/epoch）

| 方法 | 平均每 epoch | 相对 naive |
|---|---|---|
| naive | 94.1 s | 1.00× |
| YOLO-LwF | 109.4 s | 1.16× |
| 本方法 prad+prd+espreg（raw+frobenius） | 233.4 s | **2.48×** |

额外开销来自：每个 step 多一次教师前向、`prad` 的全锚点逐类 BKL、`prd` 的 3×64 patch 头前向、
以及 ESPReg 对 88 个模块的全谱（`n_components = D`）投影与反传。后续若要压开销，优先做
ESPReg 的特征方向子采样或按步累积（不改本版协议）。

### 10.4c 原始特征值与 NT 权重的谱差异（VOC 10+10 基础阶段 pca 缓存）

88 个模块、70491 个主方向：λ 跨 **3.2e-6 ~ 299（约 8 个数量级）**，中位数 0.063；每模块 **top-10 方向平均占谱的 44.6%**
（最深模块 `model.20.conv`：λ₁=299、中位 1.26e-3）。因此：

- `nt` 把 λ/λ_肘点 截断到 1，最靠前的上百个方向被压成**同一权重**，谱内相对量级丢失；
- `raw` 保留原始特征值，最主导方向与中位方向相差约 1e5 量级——这正是方案 §9.3 要求的、与
  `Tr(ΔWΣΔWᵀ)` 严格对应的加权，也是审稿意见 2.1 针对"截断后不再是推导出的二次型"的回应依据。

### 10.5 ESPReg 改名的等价性回归检查

`git show HEAD:ultralytics/engine/espreg.py` 的旧 `EWPRegLoss` 与新 `ESPRegLoss(weight_mode="nt", form="original")`
在同一 ΔW、同一 PCA 缓存上输出 **43.6557312012 vs 43.6557312012，相对差 0.0** —— 改名与开关化没有改变既有 ESPReg 行为，
历史 `--method espreg` 实验可直接复现。`frobenius` 形式与 `raw` 权重模式为新方法路径，需单独校 λ（见 §10.3）。

### 10.6 基础阶段复用与离线实验包

**基础阶段复用**：task-1 与增量方法无关，因此所有方法与所有数据集都复用同一个已训练好的基础模型，
不再为每个 run 重新训练 task-1（VOC 10+10 复用 naive run 的 `task-1/best.pt`，VOC 5+5+5+5 同理），
以 `START_TASK=2` 起跑。适配器在 `START_TASK=2` 时会用该基础模型 + 基础任务数据自动补齐
`patch_prototypes.pt` 与 `pca_cache.pkl`（`START_TASK>2` 时固定共享 PCA 不可重建，直接报错）。
效果：10+10 省约 3 h，5+5+5+5 省约 2.5 h，且各方法共享同一基础模型，比较更干净。

**离线实验包**（把一部分实验搬到别的机器跑，再回传结果）：

- `scripts/offline_experiment.sh`：一条命令完成"（可选）从 `data/VOC-YOLO` 自动切分数据集 → 复用基础模型
  （或 `--train-base` 先训基础阶段）→ 补齐基础工件 → 训练增量阶段 →
  评估（per-task + cumulative，class-aware + class-agnostic）→ 分阶段 PCA 重建报告 → 文本汇总"。
  参数为 `--tasks` 或 `--dataset/--split`（后者在数据缺失时调用 `scripts/create.sh` 现场切分，并自行推导
  任务序列）、`--base-model`（缺省按 `runs/<model>_<tag>_pretrained-from-<weights>_naive/task-1/best.pt` 查找，
  找不到可用 `--train-base` 现训）、`--output`、`--summary`；累计评估序列由任务 yaml 所在目录的 `task_1-*`
  自动发现；训练超参默认取本项目协议（AdamW / lr0 0.001 / warmup_bias_lr 0 / mosaic 0.5）；
  `--prepare-only` 只做切分与基础模型检查后退出，适合长跑前预检。不依赖网络与绝对路径。
- `tools/summarize_run.py`：把一个 run 目录汇总成单个 txt（训练协议、per-task/cumulative 矩阵、
  最终模型 Old/New/All、分阶段 mAP、PCA 重建偏差、原型相似度、CSV 清单）。

分工：本地跑 VOC 10+10；VOC 5+5+5+5 交给离线机器，回传 `summary.txt`（及 `evaluation_results/`、
`pca_reconstruction/` 的 CSV）后与本地结果合并出总表。
