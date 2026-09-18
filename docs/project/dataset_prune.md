# 数据集剪枝（coreset）算法说明

实现：`tools/prune_dataset.py`，入口：`scripts/prune.sh`。  
对比实验模板：`scripts/prune_compare/voc/`（迁到其它数据集只改该目录 `common.sh` 的第一块）。

本文描述**当前选择器**的逻辑。约束：只对原有样本做筛选与保留（图像/标签符号链接），不合成新样本。

---

## 1. 问题与接口

给定 YOLO 检测数据集 yaml，把指定 split **独立**压缩到给定规模。规模用两种方式之一指定：

- `--ratio`：keep 数为 \(\mathrm{round}(N\cdot r)\)，再夹到 \([1, N]\)
- `--num_out`：keep 数为 \(\min(\texttt{num\_out}, N)\)；一旦指定则忽略 `--ratio`

| 参数 | 含义 | 默认 |
|---|---|---|
| `--src_data` | 源数据集 yaml | 必填 |
| `--out_dir` | 输出目录 | 必填 |
| `--splits` | 写入输出的 split | `train val test` |
| `--random_splits` | 上述 split 中走均匀随机、不走 coreset 的集合 | `val test` |
| `--seed` | 随机 split 的 RNG | `0` |
| `--backbone` | 视觉特征骨干；选择器当前不使用 | `none` |

未列入 `--splits` 的 split **不会**出现在输出目录中。yaml 中的路径相对该 yaml，形如 `images/<split>`。

`--random_splits` 默认包含 val/test：验证集与测试集应与源数据同分布，才能用来评估模型。对比实验里通常只压缩 train，再用 `tools/attach_source_splits.py` 把完整源 val/test 以相对路径挂回输出 yaml。

```bash
bash scripts/prune.sh --src_data data/VOC-YOLO/VOC.yaml --out_dir data/VOC-pruned --ratio 0.2
bash scripts/prune.sh --src_data data/VOC-YOLO/VOC.yaml --out_dir data/VOC-pruned-500 --num_out 500
```

---

## 2. 流水线

对每个请求且源 yaml 中存在的 split：

1. 收集该 split 全部图像，按文件名排序（保证可复现）。
2. 解析对应 YOLO 标签，得到每张图的统计量。
3. 计算 keep 数（见上）。
4. 若该 split ∈ `--random_splits`：无放回均匀随机。
5. 否则：`select_stratified_coreset`（最稀有类分层 + 系统抽样 + 轻度稀有上采）。
6. 把选中图像与标签符号链接到 `images/<split>`、`labels/<split>`。
7. 写出 `dataset.yaml`（`names` + 各 split 的相对路径）。

源 yaml 没有的 split 记 warning 并跳过。若一个 split 都没写出则报错。

---

## 3. 每张图的标注统计

每张图解析 YOLO 行 `class_id x_center y_center width height`，得到：

- \(\mathbf{c}_i\in\mathbb{R}^{C}\)：各类实例计数
- \(b_i=\sum_c c_{i,c}\)：框总数
- 面积直方图：归一化框面积 \(w\cdot h\) 分到 4 档  
  \([0,0.02),\ [0.02,0.05),\ [0.05,0.15),\ [0.15,1]\)

无标注图（背景图）\(\mathbf{c}_i=\mathbf{0}\)，后面单独成一层。面积直方图当前**不参与**打分，解析时仍计算。无效行、越界 class id 记 warning 后跳过该行。

---

## 4. 选择器：最稀有类分层系统抽样

检测的 mAP50 是各类 AP 的**不加权平均**。均匀随机在 keep 较小时，稀有类覆盖方差大，某一类抽空或抽得极少会直接拉低 mAP。选择器把“主类”变成抽样单位，使每一层都有名额，并给稀有层略多一点位子。

### 4.1 给每张图指定主层

先统计源 split 中「含有类 \(c\) 的图像数」\(f_c\)。

对第 \(i\) 张图：

- 若没有任何框：主层 \(s_i=-1\)（背景层）
- 否则：主层为它包含的类里 \(f_c\) **最小**的那个（最稀有类）

\[
s_i=\arg\min_{c:\,c_{i,c}>0} f_c
\]

检测图常是多标签。若按“所有出现过的类都计数”，一张 `person+bottle` 会被两个类重复占用，配额会偏。指定唯一主层后每张图只属于一层。选最稀有类当主层，是为了把稀有类图像从被 person/car 淹没的池子里拆出来。

例：VOC 上大量图主层是 person；只含 diningtable、或 diningtable 相对更稀有的图，进入 diningtable 层。

### 4.2 层配额：\(n^{0.85}\) + 最大余数法

记层 \(s\) 的图像数为 \(n_s\)，总 keep 为 \(k\)。

若按 \(k\cdot n_s/N\) 分配，期望上与均匀随机相同，只是方差更小。当前实现再给稀有层轻度上采：

\[
w_s=\max\bigl(1,\; n_s^{0.85}\bigr)
\]

指数 \(0.85\) 介于正比（\(1.0\)，与随机同期望）与 \(\sqrt{n}\)（\(0.5\)，稀有上采更强）之间。

整数名额用最大余数法（Hamilton）把 \(k\) 分到各层：

1. 每层先取 \(\lfloor k\cdot w_s/\sum_t w_t\rfloor\)
2. 剩余名额按小数部分从大到小各加 1（同分则层更大者优先）
3. 每层名额不超过该层真实张数 \(n_s\)

若 cap 之后总数小于 \(k\)，用尚未选中的图像按框数排序补齐。

### 4.3 层内系统抽样

层内图像按 \((b_i,\;\text{图像下标})\) 排序，再等间隔取 \(\mathrm{quota}\) 张：

\[
\text{下标}=\bigl\lfloor q\cdot n_s/\mathrm{quota}\bigr\rfloor,\quad q=0,1,\ldots,\mathrm{quota}-1
\]

这是确定性 systematic sample：同一层里既覆盖框少的图也覆盖框多的图，避免全集中在“特别空”或“特别挤”的一端。排序含图像下标，结果与文件名排序一致，可复现。

### 4.4 类别覆盖修补

若源 split 出现过的某个类在子集中一张都没有（keep 很小或某层配额被 cap 时可能发生）：

1. 从未选集合中取一张含该类的图（按下标第一张）
2. 丢掉一张**删了也不会让任何仍应覆盖的类从子集消失**的已选图
3. 找不到可换出的图则停止（keep 小于类别数时无法保证全覆盖）

修补保持 keep 不变，只做原样本之间的替换。

---

## 5. 随机分支

`--random_splits` 中的 split 使用 `random.Random(seed).sample`，无放回均匀抽取 `keep` 张。对比脚本中 coreset 与随机是两次独立调用：后者把 train 也标进 `--random_splits`，从而得到同规模的均匀子集。

---

## 6. 设计动机

- **为何分层**：mAP50 对“某类完全没抽到 / 抽得极少”敏感。把主类当作抽样单位后，每层都有名额。
- **为何主层取最稀有类**：多标签图若算给所有类，常见类会吞掉稀有类图像。
- **为何 \(n^{0.85}\)**：严格按层大小正比时，子集与均匀随机过近，短训下难以稳定超过随机；过强的稀有上采（例如典型样本 / 实例直方图匹配）会挤占常见类、并偏向多框图，短训 mAP 反而更差。
- **为何层内按框数系统抽样**：控制每层内部的目标密度，避免整层抽成同一拥挤程度。
- **为何 val/test 默认可随机、对比实验挂回全集**：最终评估应对源测试分布；训练时的 val 若也随 train 一起被压成另一种分布，会把“选子集”和“选 checkpoint”缠在一起。

在相同超参、相同轮数、原始完整 test、指标 mAP50 的协议下：

| 数据集 | 源 train | 通过的 keep 区间 | 对应张数 |
|---|---|---|---|
| VOC | 8218 | 0.20–0.25 | 1644 / 2054 |
| ODinW-13 VehiclesOpenImages | 878 | 0.15–0.20 | 132 / 176 |

更低 keep（例如 VOC 5%–10%）上均匀随机更好：短训更吃“实例分布接近全集”，稀有上采会占掉常见类数据。

对比命令：

```bash
bash scripts/prune_compare/voc/run.sh
bash scripts/prune_compare/odinw13/run.sh
```

---

## 7. 当前未参与选择的接口

`--backbone` 仍可提取视觉特征（YOLO / torchvision / timm，可 PCA 到 64 维并缓存），但 `select_stratified_coreset` **丢弃** `visual`，默认 `none`。面积直方图同样不进入配额与抽样。

---

## 8. 相关文件

| 路径 | 角色 |
|---|---|
| `tools/prune_dataset.py` | 选择与物化 |
| `scripts/prune.sh` | 仓库根目录下的 shell 入口 |
| `tools/attach_source_splits.py` | 把源 split 以相对路径挂进已剪枝 yaml |
| `tools/summarize_prune_compare.py` | 判断是否存在连续比例区间上 coreset mAP50 严格高于随机 |
| `tools/convert_odinw13_to_yolo.py` | ODinW-13 COCO json → YOLO |
| `scripts/prune_compare/voc/` | 对比模板（只改 `common.sh` 第一块即可迁移） |
| `scripts/prune_compare/odinw13/` | 上述模板迁到 VehiclesOpenImages 的副本 |
