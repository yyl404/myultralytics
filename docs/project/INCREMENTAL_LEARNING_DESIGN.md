# 增量学习系统设计文档

> **存档说明（2026-08 重构）**：本文档描述的是早期设计（VSPReg/ProtoRP/SampleRP），其中部分机制与工具已被取代或移除：ProtoRP 由 RePRE（`ultralytics/engine/repre.py`）取代；`train_incremental.py`、`eval_incremental.py`、`incremental_utils.py`、`generate_pseudo_label.py`、`generate_replay_dataset.py`、`convert_prototype_classes.py`、`train_head_proto.py` 已删除。现行管线以 `scripts/<dataset>/<split>/<baseline>/train_*.sh` + `tools/train.py` 为准，可用方法为 naive / pseudo_label / ewc / l2 / espreg / nsgp(+repre) / distillation / bpf 及组合。文档其余内容保留作为设计历史参考。

## 概述

本系统实现了一个基于 YOLO 的多任务增量学习框架，通过结合 **VSPReg (Variance-Scaled Projection Regularization)**、**ProtoRP (Prototype Replay)** 和 **SampleRP (Sample Replay)** 三种机制，有效缓解增量学习中的灾难性遗忘问题。该系统支持在多个任务序列上持续学习，每个新任务可以引入新的类别，同时保持对历史任务类别的检测能力。

## 核心设计理念

### 1. 增量学习挑战

在增量学习场景中，模型需要：
- **学习新任务**：在新任务数据上学习新类别
- **保持旧知识**：不遗忘历史任务中已学习的类别
- **模型扩展**：动态扩展检测头以支持新类别

### 2. 解决方案架构

系统采用三种互补的机制来防止遗忘：

1. **VSPReg**: 在权重更新的主成分子空间中进行正则化，限制对重要方向的更新
2. **ProtoRP**: 通过重放代表性原型特征来保持历史知识
3. **SampleRP**: 通过重放历史样本（使用伪标签）来增强记忆

## 关键技术详解

### 1. VSPReg (Variance-Scaled Projection Regularization)

VSPReg 是一种基于主成分分析（PCA）的正则化方法，通过限制权重更新在重要方向上的投影来防止遗忘。

#### 核心思想

- 对历史任务训练后的模型权重进行 PCA 分析，提取主成分
- 在训练新任务时，限制权重更新在主成分子空间上的投影长度
- 通过方差缩放，对高方差（重要）方向施加更强的约束

#### 实现细节

**PCA 分析阶段**：

```python
# tools/pca.py
class PCAHooker:
    def __init__(self, model, layers, modules=None, device="cuda"):
        # 为指定层注册前向钩子，收集输入特征
        # 对卷积层的输入特征进行展开（unfold）处理
        # 使用增量PCA计算主成分和方差
```

**VSPReg 损失计算**：

```python
# ultralytics/engine/vspreg.py
class VSPRegLoss:
    def __init__(self, model_update, model_base, module_names, 
                 components, variances, keep_ratio=0.9, 
                 center_ratio=0.9, steepness=100):
        """
        Args:
            model_update: 当前更新的模型
            model_base: 历史任务的基模型（冻结）
            module_names: 需要应用VSPReg的模块名称列表
            components: PCA主成分 [num_groups, num_components, c_in//g*k*k]
            variances: PCA方差 [num_groups, num_components]
            keep_ratio: 保留的主成分比例（基于累积方差）
        """
        # 根据keep_ratio保留主要成分
        # 初始化更新模型和基模型的权重字典
        # 注册前向钩子以捕获权重
    
    def get_loss(self):
        """计算VSPReg损失"""
        loss = 0
        for n in self.module_names:
            proj = self.components[n]  # [g, r, c_in//g*k*k]
            scale = torch.sqrt(self.variances[n])  # [g, r]
            
            update_w = self.update_weights[n]  # [g, c_out//g, c_in//g*k*k]
            base_w = self.base_weights[n]  # [g, c_out//g, c_in//g*k*k]
            
            # 计算归一化的权重更新
            delta_w = F.normalize(update_w - base_w, p=2, dim=2)
            
            # 计算在主成分子空间上的投影
            # ([g, c_out//g, c_in//g*k*k] @ [g, c_in//g*k*k, r]) 
            # = [g, c_out//g, r]
            proj_length = (delta_w @ proj.transpose(1, 2)).norm(dim=2).mean()
            loss += proj_length
        
        return loss / len(self.module_names)
```

**关键特性**：
- **分组处理**：对分组卷积（grouped convolution）的每个组分别进行PCA
- **成分选择**：根据累积方差比例（`keep_ratio`）保留主要成分
- **归一化投影**：对权重更新进行L2归一化后再投影，避免尺度影响

### 2. ProtoRP (Prototype Replay)

ProtoRP 通过重放从历史任务中提取的代表性原型特征来保持历史知识。

#### 核心思想

- 从历史任务的训练数据中提取代表性特征原型
- 每个原型包含：5×5 特征块、回归输出、分类输出、位置信息
- 在新任务训练时，定期重放这些原型，计算损失以保持历史知识

#### 原型生成流程

```python
# tools/generate_prototypes.py
def generate_prototypes(model, dataset, num_protos=10):
    """
    1. 对每个训练图像进行前向传播
    2. 对每个GT bbox，收集所有检测层的特征：
       - IOU > 0.5 且分类正确的检测向量
       - 提取 5×5 特征块和对应的回归/分类输出
    3. 按类别和层组织原型
    4. 使用K-means聚类选择代表性原型（每类num_protos个）
    5. 保存为: [num_prototypes, C*5*5 + reg_out + cls_out + pad_mask]
    """
```

**原型格式**：

```python
prototypes[layer_id] = torch.Tensor([
    # [num_prototypes, feature_dim + reg_dim + cls_dim + pad_dim]
    # feature_dim = in_channels * 5 * 5  (5×5特征块展平)
    # reg_dim = 4 * reg_max  (4个回归值，每个reg_max维)
    # cls_dim = num_classes  (分类logits)
    # pad_dim = 5 * 5  (填充掩码，标记有效区域)
])
```

#### 原型重放损失

```python
# ultralytics/engine/antiforget.py
def compute_proto_replay_loss(self, batch_idx: int):
    """
    计算原型重放损失
    
    流程：
    1. 从原型中提取特征块、回归输出、分类输出
    2. 恢复5×5特征图（处理填充）
    3. 通过当前模型的检测头前向传播
    4. 计算与监督信号的损失
    """
    detect = self.model.model[-1]
    detect.eval()
    
    for lid in range(detect.nl):  # 对每个检测层
        # 提取原型特征
        prototypes = self.prototypes[lid][:, :in_channels*5*5]
        prototypes = prototypes.reshape(-1, in_channels, 5, 5)
        
        # 恢复特征图（处理填充）
        prototypes, offset_y, offset_x = self.restore_prototypes(
            prototypes, pad_mask
        )
        
        # 通过检测头
        reg_out = reg[lid](prototypes)
        cls_out = cls[lid](prototypes)
        
        # 提取对应位置的输出
        y_pos = offset_y + 2
        x_pos = offset_x + 2
        reg_out = reg_out[:, :, y_pos, x_pos]  # [N, reg_dim]
        cls_out = cls_out[:, :, y_pos, x_pos]  # [N, cls_dim]
        
        # 获取监督信号
        if self.proto_rp_use_base_model:
            # 使用基模型的输出作为监督（知识蒸馏）
            with torch.no_grad():
                base_reg_out = base_reg[lid](prototypes)
                base_cls_out = base_cls[lid](prototypes)
                reg_supervision = base_reg_out[:, :, y_pos, x_pos]
                cls_supervision = base_cls_out[:, :, y_pos, x_pos]
        else:
            # 使用原型中存储的原始输出作为监督
            reg_supervision = self.prototypes[lid][:, in_channels*5*5:...]
            cls_supervision = ...
        
        # 计算损失
        cls_loss_proto += F.binary_cross_entropy_with_logits(
            cls_out, cls_supervision.sigmoid()
        ) - F.binary_cross_entropy_with_logits(
            cls_supervision, cls_supervision.sigmoid()  # 最小化目标
        )
        
        reg_loss_proto += F.cross_entropy(
            reg_out.reshape(-1, reg_max),
            F.softmax(reg_supervision.reshape(-1, reg_max), dim=1)
        ) - F.cross_entropy(
            reg_supervision.reshape(-1, reg_max),
            F.softmax(reg_supervision.reshape(-1, reg_max), dim=1)
        )
    
    detect.train()
    return cls_loss_proto, reg_loss_proto
```

**关键特性**：
- **特征级重放**：重放特征而非原始图像，节省存储和计算
- **位置感知**：保留特征在原图中的位置信息（offset）
- **两种监督模式**：
  - `proto_rp_use_base_model=True`: 使用基模型输出作为软标签（知识蒸馏）
  - `proto_rp_use_base_model=False`: 使用原型中存储的原始输出

### 3. SampleRP (Sample Replay)

SampleRP 通过重放历史任务的真实样本（使用伪标签）来增强记忆。

#### 核心思想

- 从原型中提取对应的原始图像和标注信息
- 使用历史任务模型生成伪标签（用于数据增强）
- 将重放样本与当前任务数据合并训练

#### 实现流程

```python
# tools/generate_replay_dataset.py
def generate_replay_dataset(prototypes, output_dir):
    """
    1. 从原型meta_info中提取图像路径和标注
    2. 复制图像到输出目录
    3. 创建YOLO格式的标签文件
    4. 生成dataset.yaml配置文件
    """
```

**伪标签生成**：

```python
# tools/generate_pseudo_label.py
def generate_pseudo_labels(model, dataset, output_dir, conf_threshold=0.25):
    """
    使用历史任务模型对重放样本生成伪标签
    - 只保留置信度 > conf_threshold 的检测
    - 保存为YOLO格式标签文件
    """
```

**数据集合并**：

```python
# tools/merge_datasets.py
def merge_datasets(dataset1_yaml, dataset2_yaml, output_dir):
    """
    合并重放数据集和当前任务数据集
    - 合并图像路径
    - 合并类别名称（去重）
    - 生成新的dataset.yaml
    """
```

## 增量学习流程

### 任务 1：初始任务训练

```bash
# 1. 融合YOLOE模型到YOLO架构
python tools/fuse_zero-shot_yoloe.py \
    --input yoloe-v8l-seg.pt \
    --output task-1/yoloe-v8l-fused.pt \
    --model_cfg yolov8l.yaml \
    --data data/4-domain/voc/dataset.yaml

# 2. 训练第一个任务
python tools/train.py \
    --model task-1/yoloe-v8l-fused.pt \
    --data data/4-domain/voc/dataset.yaml \
    --save_path task-1/best.pt \
    --epochs 100 \
    --freeze [0,1,2,...,21]  # 冻结backbone

# 3. 执行PCA分析
python tools/pca.py \
    --model task-1/best.pt \
    --dataset data/4-domain/voc/dataset.yaml \
    --save_path task-1/pca_cache.pkl

# 4. 生成原型
python tools/generate_prototypes.py \
    --model task-1/best.pt \
    --data data/4-domain/voc/dataset.yaml \
    --output task-1/prototypes.pt \
    --num_protos 100
```

**关键步骤说明**：

1. **YOLOE融合**：将零样本YOLOE模型的文本编码能力融合到标准YOLO架构中
2. **冻结训练**：冻结backbone，只训练检测头，快速适应新任务
3. **PCA分析**：对卷积层输入特征进行PCA，提取主成分用于VSPReg
4. **原型生成**：从训练数据中提取代表性特征原型

### 任务 2+：增量任务训练

```bash
# 1. 扩展模型头以支持新类别
python tools/expand_model_head.py \
    --model task-1/best.pt \
    --model_cfg yolov8l.yaml \
    --dataset data/4-domain/clipart/dataset.yaml \
    --save_path task-2/task-1-best-expanded.pt \
    --class_embedding_init \
    --yoloe_model yoloe-v8l-seg.pt

# 2. 转换原型类别ID（匹配扩展后的模型）
python tools/convert_prototype_classes.py \
    --prototypes task-1/prototypes.pt \
    --original_model task-1/best.pt \
    --expanded_model task-2/task-1-best-expanded.pt \
    --output task-2/task-1-prototypes-converted.pt

# 3. 生成重放数据集
python tools/generate_replay_dataset.py \
    --prototypes task-2/task-1-prototypes-converted.pt \
    --output task-2/task-1-replay-samples \
    --copy_images \
    --use_all_annotations

# 4. 生成伪标签
python tools/generate_pseudo_label.py \
    --model task-1/best.pt \
    --dataset task-2/task-1-replay-samples/dataset.yaml \
    --output_dir task-2/task-1-replay-samples-pseudo-labels \
    --conf_threshold 0.25

# 5. 合并数据集
python tools/merge_datasets.py \
    --datasets task-2/task-1-replay-samples-pseudo-labels/dataset.yaml \
               data/4-domain/clipart/dataset.yaml \
    --output_dir task-2/task-1-dataset_merged

# 6. 转换类别ID（匹配扩展模型）
python tools/convert_dataset_class_ids.py \
    --model task-2/task-1-best-expanded.pt \
    --dataset task-2/task-1-dataset_merged/dataset.yaml \
    --output_dir task-2/task-1-dataset_converted

# 7. 训练（使用VSPReg和ProtoRP）
python tools/train.py \
    --model task-2/task-1-best-expanded.pt \
    --data task-2/task-1-dataset_converted/dataset.yaml \
    --save_path task-2/best.pt \
    --trainer antiforget \
    --vspreg True \
    --pca_cache_path task-1/pca_cache.pkl \
    --proto_rp True \
    --prototypes task-2/task-1-prototypes-converted.pt \
    --proto_rp_use_base_model True \
    --freeze [0,1,2,...,21]
```

**关键步骤说明**：

1. **模型头扩展**：
   - 为新类别分配新的检测头通道
   - 使用YOLOE文本嵌入初始化新类别权重（`class_embedding_init`）
   - 保持历史类别权重不变

2. **原型转换**：
   - 将历史原型的类别ID映射到扩展模型的类别空间
   - 确保原型重放时使用正确的类别索引

3. **重放数据集生成**：
   - 从原型meta_info中提取原始图像和标注
   - 创建YOLO格式的数据集

4. **伪标签生成**：
   - 使用历史任务模型对重放样本进行推理
   - 生成高质量的伪标签用于训练

5. **数据集合并与转换**：
   - 合并重放样本和当前任务数据
   - 转换类别ID以匹配扩展模型的类别顺序

6. **增量训练**：
   - 使用 `AntiForgetTrainer` 进行训练
   - 同时应用VSPReg和ProtoRP损失

## 核心组件实现

### 1. AntiForgetTrainer

`AntiForgetTrainer` 是增量学习的核心训练器，集成了VSPReg和ProtoRP机制。

```python
# ultralytics/engine/antiforget.py
class AntiForgetTrainer(BaseTrainer):
    def _setup_train(self, world_size):
        """设置训练环境"""
        # 1. 设置基模型（历史任务模型，冻结）
        self.base_model = deepcopy(self.model).eval()
        for p in self.base_model.parameters():
            p.requires_grad_(False)
        
        # 2. 设置VSPReg损失
        if self.args.vspreg:
            self.pca_cache = joblib.load(self.args.pca_cache_path)
            components, variances = self._extract_pca_components()
            self.vspreg_loss = VSPRegLoss(
                self.model, self.base_model,
                module_names=self.pca_cache.keys(),
                components=components,
                variances=variances,
                keep_ratio=self.args.vspreg_keep_ratio,
                center_ratio=self.args.vspreg_center_ratio,
                steepness=self.args.vspreg_steepness
            )
        
        # 3. 设置ProtoRP
        if self.args.proto_rp:
            prototypes_dict = torch.load(self.args.prototypes)
            self.prototypes = prototypes_dict["prototypes"]
            # 移动到设备并冻结梯度
            for lid, x in enumerate(self.prototypes):
                self.prototypes[lid] = x.to(self.device).requires_grad_(False)
            
            self.proto_rp_use_base_model = self.args.proto_rp_use_base_model
    
    def train_step(self, batch):
        """训练步骤"""
        # 前向传播
        preds = self.model(batch["img"])
        loss, loss_items = self.model.loss(preds, batch)
        
        # 添加VSPReg损失
        if self.args.vspreg:
            self.vspreg_loss.register_hook()  # 注册钩子捕获权重
            vspreg_loss = self.vspreg_loss.get_loss()
            loss += vspreg_loss * self.vspreg_loss_weight
            loss_items = torch.cat([loss_items, vspreg_loss])
        
        # 添加ProtoRP损失
        if self.args.proto_rp:
            proto_losses = self.compute_proto_replay_loss(batch_idx)
            cls_loss_proto, reg_loss_proto = proto_losses[:2]
            loss += (cls_loss_proto + reg_loss_proto) * self.proto_rp_loss_weight
            loss_items = torch.cat([loss_items, cls_loss_proto, reg_loss_proto])
        
        # 反向传播
        loss.backward()
        self.optimizer.step()
        
        return loss, loss_items
```

### 2. 模型头扩展

模型头扩展是增量学习的关键步骤，需要为新类别分配通道并初始化权重。

```python
# tools/expand_model_head.py
def expand_detection_head(ckpt_path, model_cfg, channel_map, 
                          classes_names, save_dir, output_name,
                          class_embedding_init=False, yoloe_model_path=None):
    """
    扩展检测头以支持新类别
    
    Args:
        channel_map: 旧类别ID到新类别ID的映射 {old_idx: new_idx}
        class_embedding_init: 是否使用文本嵌入初始化新类别权重
    """
    # 1. 加载旧模型
    model = YOLO(ckpt_path)
    old_weight = model.model.state_dict()
    
    # 2. 创建新模型（更多类别）
    model_cfg["nc"] = len(classes_names)
    new_model = YOLO(new_model_cfg)
    
    # 3. 迁移旧类别权重
    for name, param in new_model.model.named_parameters():
        if 'cv3' in name and name.endswith('.2.weight'):  # 分类头权重
            # 迁移旧类别权重到对应位置
            for old_idx, new_idx in channel_map.items():
                param.data[new_idx] = old_weight[name][old_idx]
            
            # 初始化新类别权重
            if class_embedding_init:
                # 使用YOLOE文本嵌入初始化
                yoloe_model = YOLOE(yoloe_model_path)
                new_class_names = [classes_names[i] for i in new_class_indices]
                text_embeddings = yoloe_model.get_text_pe(new_class_names)
                # 将文本嵌入投影到分类头权重空间
                param.data[new_class_indices] = project_embedding(text_embeddings)
            else:
                # 零初始化
                param.data[new_class_indices].zero_()
    
    # 4. 保存扩展后的模型
    torch.save({
        'model': new_model.model.state_dict(),
        'names': classes_names,
        # ... 其他元数据
    }, save_path)
```

**关键设计**：
- **权重迁移**：旧类别权重按映射关系迁移到新模型
- **智能初始化**：新类别可以使用YOLOE文本嵌入初始化，利用零样本知识
- **通道映射**：维护旧类别到新类别的映射关系

### 3. 原型转换

当模型头扩展后，需要将历史原型的类别ID转换到新的类别空间。

```python
# tools/convert_prototype_classes.py
def convert_prototype_classes(prototypes, original_model, 
                              expanded_model, output_path):
    """
    转换原型的类别ID以匹配扩展后的模型
    
    流程：
    1. 加载原始模型和扩展模型，获取类别映射关系
    2. 对原型中的类别相关部分进行转换：
       - 分类输出：重新排列类别维度
       - 类别掩码：更新有效类别标记
    3. 保存转换后的原型
    """
```

## 训练流程详解

### 完整训练循环

```python
# ultralytics/engine/antiforget.py
for epoch in range(self.epochs):
    for batch_idx, batch in enumerate(dataloader):
        # 1. 标准检测损失
        preds = self.model(batch["img"])
        loss, loss_items = self.model.loss(preds, batch)
        
        # 2. VSPReg损失（如果启用）
        if self.args.vspreg:
            # 注册钩子捕获当前batch的权重
            self.vspreg_loss.register_hook()
            
            # 计算权重更新在主成分子空间上的投影
            vspreg_loss = self.vspreg_loss.get_loss()
            loss += vspreg_loss * self.vspreg_loss_weight
            
            # 移除钩子
            self.vspreg_loss.remove_handle_()
        
        # 3. ProtoRP损失（如果启用）
        if self.args.proto_rp:
            # 从原型中采样一批原型特征
            # 通过当前模型前向传播
            # 计算与监督信号的损失
            proto_losses = self.compute_proto_replay_loss(batch_idx)
            cls_loss_proto, reg_loss_proto = proto_losses[:2]
            loss += (cls_loss_proto + reg_loss_proto) * self.proto_rp_loss_weight
        
        # 4. 反向传播和优化
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
```

### 损失函数组合

总损失函数为：

```
L_total = L_detection + λ_vspreg * L_vspreg + λ_proto * L_proto
```

其中：
- `L_detection`: 标准YOLO检测损失（box loss + cls loss + dfl loss）
- `L_vspreg`: VSPReg正则化损失
- `L_proto`: 原型重放损失（分类 + 回归）
- `λ_vspreg`, `λ_proto`: 损失权重（可配置）

## 工具脚本说明

### 核心工具脚本

1. **`tools/fuse_zero-shot_yoloe.py`**
   - 将YOLOE模型的零样本能力融合到标准YOLO架构
   - 提取文本编码器权重并初始化YOLO分类头

2. **`tools/expand_model_head.py`**
   - 扩展检测头以支持新类别
   - 支持零初始化和文本嵌入初始化两种模式

3. **`tools/pca.py`**
   - 对模型中间层输入特征进行PCA分析
   - 提取主成分和方差，用于VSPReg

4. **`tools/generate_prototypes.py`**
   - 从训练数据中提取代表性特征原型
   - 使用K-means聚类选择每类的代表性原型

5. **`tools/convert_prototype_classes.py`**
   - 转换原型的类别ID以匹配扩展后的模型

6. **`tools/generate_replay_dataset.py`**
   - 从原型meta_info中提取图像和标注
   - 生成YOLO格式的重放数据集

7. **`tools/generate_pseudo_label.py`**
   - 使用历史模型对重放样本生成伪标签
   - 支持置信度阈值过滤

8. **`tools/merge_datasets.py`**
   - 合并多个数据集（重放数据集 + 当前任务数据集）
   - 处理类别名称去重和映射

9. **`tools/convert_dataset_class_ids.py`**
   - 转换数据集的类别ID以匹配模型的类别顺序

## 训练脚本流程

### 脚本结构

```bash
# scripts/4-domain/yoloev8/train_pseudo_label+espreg+proto_rp+sample_rp.sh

# 配置参数
MODEL_CFG="yolov8l.yaml"
YOLOE_MODEL_WEIGHT="yoloe-v8l-seg.pt"
TASK_DATASETS=(
    "data/4-domain/voc/dataset.yaml"
    "data/4-domain/clipart/dataset.yaml"
    "data/4-domain/watercolor/dataset.yaml"
    "data/4-domain/comic/dataset.yaml"
)

# 任务1：初始训练
if [ $task_num -eq 1 ]; then
    # 融合YOLOE → 训练 → PCA → 生成原型
fi

# 任务2+：增量训练
else
    # 扩展模型头 → 转换原型 → 生成重放数据集 → 
    # 生成伪标签 → 合并数据集 → 转换类别ID → 训练
fi
```

### 关键配置参数

```bash
# VSPReg配置
vspreg=True
vspreg_loss_weight=1.0
vspreg_keep_ratio=1.0      # 保留的主成分比例
vspreg_center_ratio=0.9   # 中心成分比例
vspreg_steepness=100       # 陡度参数

# ProtoRP配置
proto_rp=True
proto_rp_loss_weight=10000  # 原型重放损失权重（通常较大）
proto_rp_use_base_model=True  # 使用基模型输出作为监督

# 训练配置
freeze=[0,1,2,...,21]      # 冻结backbone层
epochs=100
batch_size=16
patience=15                 # 早停耐心值
```

## 技术亮点

### 1. 多层次防遗忘机制

- **权重级别**：VSPReg限制权重更新方向
- **特征级别**：ProtoRP重放代表性特征
- **样本级别**：SampleRP重放真实样本

### 2. 高效的存储和计算

- **原型压缩**：只存储5×5特征块而非完整图像
- **按需重放**：每个batch只重放一批原型，而非全部
- **PCA降维**：只保留主要成分，减少存储和计算

### 3. 灵活的类别扩展

- **动态扩展**：支持任意数量的新类别
- **智能初始化**：利用YOLOE文本嵌入初始化新类别
- **类别映射**：自动处理类别ID的转换

### 4. 可恢复的训练

- **检查点支持**：可以从任意任务恢复训练
- **状态保存**：保存PCA缓存和原型，支持断点续训

## 使用示例

### 完整训练流程

```bash
# 运行增量学习训练脚本
bash scripts/4-domain/yoloev8/train_pseudo_label+espreg+proto_rp+sample_rp.sh
```

### 从特定任务恢复

```bash
# 从任务3开始训练
START_TASK=3 bash scripts/4-domain/yoloev8/train_pseudo_label+espreg+proto_rp+sample_rp.sh
```

### 自定义配置

```bash
# 修改脚本中的配置参数
MODEL_CFG="yolov8s.yaml"        # 使用更小的模型
NUM_PROTOS=50                    # 减少原型数量
PROTO_RP_LOSS_WEIGHT=5000        # 调整损失权重
```

## 实验结果与评估

### 评估指标

系统支持标准的增量学习评估指标：

- **平均准确率（Average Accuracy）**：所有任务的平均mAP
- **遗忘度（Forgetting）**：历史任务性能下降程度
- **最终准确率（Final Accuracy）**：所有任务训练后的最终性能

### 评估脚本

```bash
# tools/eval_incremental.py
python tools/eval_incremental.py \
    --model task-4/best.pt \
    --datasets data/4-domain/voc/dataset.yaml \
              data/4-domain/clipart/dataset.yaml \
              data/4-domain/watercolor/dataset.yaml \
              data/4-domain/comic/dataset.yaml
```

## 文件结构

```
ultralytics/
├── engine/
│   ├── antiforget.py          # AntiForgetTrainer
│   └── vspreg.py              # VSPRegLoss
├── models/yolo/detect/
│   └── train.py               # AntiForgetDetectionTrainer
tools/
├── fuse_zero-shot_yoloe.py    # YOLOE融合
├── expand_model_head.py       # 模型头扩展
├── pca.py                     # PCA分析
├── generate_prototypes.py     # 原型生成
├── convert_prototype_classes.py  # 原型转换
├── generate_replay_dataset.py # 重放数据集生成
├── generate_pseudo_label.py   # 伪标签生成
├── merge_datasets.py          # 数据集合并
├── convert_dataset_class_ids.py  # 类别ID转换
└── train.py                   # 训练入口
scripts/
└── 4-domain/yoloev8/
    └── train_pseudo_label+espreg+proto_rp+sample_rp.sh  # 训练脚本
```

## 总结

本增量学习系统通过结合VSPReg、ProtoRP和SampleRP三种机制，实现了高效的多任务增量学习。系统具有以下优势：

1. **全面的防遗忘机制**：从权重、特征、样本三个层面防止遗忘
2. **高效的存储和计算**：原型压缩和PCA降维减少资源消耗
3. **灵活的扩展能力**：支持动态类别扩展和智能初始化
4. **完整的工具链**：提供从数据准备到模型训练的完整工具
5. **可恢复的训练**：支持断点续训和任务恢复

该实现为增量目标检测提供了一个完整的解决方案，可以应用于需要持续学习新类别的各种场景。

