# YOLOFiLM 设计文档

## 概述

YOLOFiLM 是一个基于 Ultralytics YOLO 框架的对象检测模型，集成了 **FiLM (Feature-wise Linear Modulation)** 机制，能够根据自然语言属性描述对视觉特征进行条件调制。该设计允许模型在检测过程中考虑额外的属性信息（如物体状态、可见性等），从而提升检测性能。

## 核心设计理念

### 1. FiLM 机制

FiLM (Feature-wise Linear Modulation) 是一种条件特征调制技术，通过仿射变换（缩放和偏移）对特征图进行调制：

```
output = input * gamma + beta
```

其中 `gamma` 和 `beta` 是根据条件信息（属性文本）动态生成的调制参数。

### 2. 架构设计

YOLOFiLM 在标准 YOLO 检测架构基础上增加了两个关键组件：

1. **AttributeEncoder**: 将自然语言属性文本编码为特征向量
2. **FiLM 模块**: 在检测头之前对多尺度特征图进行条件调制

## 文件结构

```
ultralytics/
├── models/film/
│   ├── __init__.py          # 导出 YOLOFiLM
│   ├── model.py             # YOLOFiLM 主接口类
│   ├── train.py             # FiLMTrainer 训练器
│   └── val.py               # FiLMValidator 验证器
├── nn/
│   ├── modules/film.py      # AttributeEncoder 和 FiLM 模块
│   └── tasks_film.py        # DetectionModelFiLM 模型定义
└── data/
    └── dataset_json.py      # JSONAttributeDataset 数据集类
```

## 核心组件详解

### 1. AttributeEncoder - 属性编码器

`AttributeEncoder` 使用 CLIP 模型的预训练权重将自然语言属性文本编码为特征向量，类似于 YOLOE 的文本分支设计。

**关键特性：**
- 使用 CLIP ViT-B/32 的预训练权重
- 冻结 CLIP 参数，只训练投影层
- 自动处理设备迁移和边界情况

**代码实现：**

```python
class AttributeEncoder(nn.Module):
    """
    将结构化自然语言属性文本编码为连续的Embedding向量。
    使用CLIP模型的预训练权重进行编码，类似于YOLOE的文本分支。
    """
    def __init__(self, embed_dim=256, clip_variant="clip:ViT-B/32", device=None):
        super().__init__()
        self.embed_dim = embed_dim
        
        # 创建CLIP文本编码器（使用预训练权重）
        self.clip_model = build_text_model(clip_variant, device=device)
        self.clip_model.eval()
        
        # 冻结CLIP模型参数，只训练投影层
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # 获取CLIP的输出维度（通常是512维）
        with torch.no_grad():
            dummy_text = ["test"]
            dummy_tokens = self.clip_model.tokenize(dummy_text)
            dummy_features = self.clip_model.encode_text(dummy_tokens)
            clip_dim = dummy_features.shape[-1]
        
        # 投影层：将CLIP输出维度映射到目标embedding维度
        self.projector = nn.Sequential(
            nn.Linear(clip_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )
    
    def forward(self, attr_texts):
        # 处理文本并编码
        processed_texts = []
        for text in attr_texts:
            if text and isinstance(text, str) and text.strip():
                processed_texts.append(text.strip())
            else:
                processed_texts.append("a normal object")
        
        # 使用CLIP编码文本
        with torch.no_grad():
            tokens = self.clip_model.tokenize(processed_texts)
            tokens = tokens.to(current_device)
            clip_features = self.clip_model.encode_text(tokens)
            clip_features = clip_features.to(current_device)
        
        # 通过投影层映射到目标维度
        x = self.projector(clip_features)
        return x
```

**设计亮点：**
- 利用 CLIP 的预训练知识，无需从零训练文本编码器
- 投影层可训练，允许模型学习任务特定的文本-视觉对齐
- 自动处理设备迁移，确保在训练和推理时都能正确工作

### 2. FiLM 模块

`FiLM` 模块实现特征级别的线性调制，根据属性 embedding 生成调制参数。

**代码实现：**

```python
class FiLM(nn.Module):
    """
    FiLM (Feature-wise Linear Modulation) 层。
    作用：根据属性特征，对视觉特征进行仿射变换 (Scale & Shift)。
    """
    def __init__(self, feat_channels, attr_dim):
        super().__init__()
        self.scale_gen = nn.Linear(attr_dim, feat_channels)
        self.shift_gen = nn.Linear(attr_dim, feat_channels)
        
        # 初始化：scale 从 1.0 开始，shift 从 0 开始
        nn.init.zeros_(self.scale_gen.weight)
        nn.init.constant_(self.scale_gen.bias, 1.0)
        nn.init.zeros_(self.shift_gen.weight)
        nn.init.zeros_(self.shift_gen.bias)

    def forward(self, x, attr_emb):
        """
        x: [B, C, H, W] - 视觉特征图
        attr_emb: [B, attr_dim] - 属性embedding
        """
        if attr_emb is None:
            return x
        
        batch_size, channels, _, _ = x.shape
        
        # 生成调制参数 [B, C, 1, 1]
        gamma = self.scale_gen(attr_emb).view(batch_size, channels, 1, 1)
        beta = self.shift_gen(attr_emb).view(batch_size, channels, 1, 1)
        
        # 应用调制
        return x * gamma + beta
```

**设计亮点：**
- 初始化策略：scale 从 1.0 开始，确保训练初期不破坏预训练特征
- 空间不变性：调制参数在所有空间位置共享，减少参数量
- 通道特定调制：每个通道有独立的调制参数，允许细粒度控制

### 3. DetectionModelFiLM - 主模型

`DetectionModelFiLM` 继承自 `DetectionModel`，集成 FiLM 机制到 YOLO 检测流程中。

**架构集成点：**

FiLM 模块被插入到检测头（Detect Head）之前，对多尺度特征图（P3, P4, P5）分别进行调制：

```python
class DetectionModelFiLM(DetectionModel):
    def __init__(self, cfg='yolov8n.yaml', ch=3, nc=None, verbose=True):
        # 初始化属性编码器
        device = next(self.model.parameters()).device
        self.attr_encoder = AttributeEncoder(
            embed_dim=self.attr_dim, 
            clip_variant="clip:ViT-B/32", 
            device=device
        )
        
        # 为每个检测尺度创建 FiLM 模块
        m = self.model[-1]  # Detect Head
        if isinstance(m, (Detect, Segment, Pose, OBB)):
            self.film_modules = nn.ModuleList()
            for i in range(m.nl):  # m.nl 通常是 3 (P3, P4, P5)
                c_in = m.cv2[i][0].conv.in_channels
                self.film_modules.append(FiLM(c_in, self.attr_dim))
```

**前向传播流程：**

```python
def _predict_once(self, x, profile=False, visualize=False, attr_emb=None):
    """在特征进入 Head 之前应用 FiLM"""
    y, dt = [], []
    for m in self.model:
        if m.f != -1:
            x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
        
        # 关键修改: 在 Head 之前拦截 x
        if isinstance(m, (Detect, Segment, Pose, OBB)) and \
           self.film_modules is not None and attr_emb is not None:
            # x 是一个列表 [P3, P4, P5]
            x_modified = []
            for i, feat in enumerate(x):
                # 应用对应的 FiLM 模块
                feat_out = self.film_modules[i](feat, attr_emb)
                x_modified.append(feat_out)
            x = x_modified
        
        x = m(x)  # run
        y.append(x if m.i in self.save else None)
    return x
```

**设计亮点：**
- 多尺度调制：每个检测尺度（P3, P4, P5）都有独立的 FiLM 模块
- 无缝集成：不影响原有的 YOLO 架构，只在检测头前添加调制
- 条件推理：支持有/无属性文本的推理模式

### 4. JSONAttributeDataset - 数据集类

`JSONAttributeDataset` 扩展了 `YOLODataset`，支持从 JSON 文件加载标注和属性信息。

**数据格式：**

```json
[
    {
        "image_path": "/path/to/image.jpg",
        "class_id": 14,
        "xywh": [0.5, 0.5, 0.3, 0.4],
        "img_size": [640, 480],
        "attributes": {
            "state": "occluded",
            "visibility": "raining"
        }
    }
]
```

**属性文本转换：**

```python
def attr_to_text(attr_dict):
    """
    将属性字典转换为自然语言文本。
    例如: {"state": "occluded", "visibility": "raining"} 
    -> "state: occluded, visibility: raining"
    """
    if not isinstance(attr_dict, dict):
        return ""
    parts = [f"{k}: {v}" for k, v in sorted(attr_dict.items())]
    return ", ".join(parts)
```

**关键实现：**

```python
class JSONAttributeDataset(YOLODataset):
    def __init__(self, *args, json_path=None, **kwargs):
        self.json_path = json_path
        self.json_data = []
        if json_path:
            with open(json_path, 'r') as f:
                self.json_data = json.load(f)
        
        # 建立图像路径到标注的映射
        self.img_map = {item['image_path']: item for item in self.json_data}
        
        # 属性转文本函数
        self.attr_to_text = lambda attr: ", ".join(
            [f"{k}: {v}" for k, v in sorted(attr.items())]
        )
        
        super().__init__(*args, **kwargs)
    
    def get_labels(self):
        """从 JSON 数据构建标签"""
        img_to_labels = {}
        
        for item in self.json_data:
            img_path = item['image_path']
            if img_path not in img_to_labels:
                img_to_labels[img_path] = {
                    'im_file': img_path,
                    'shape': (item['img_size'][1], item['img_size'][0]),
                    'cls': [],
                    'bboxes': [],
                    'attr_texts': [],  # 存储属性文本
                    'segments': [],
                    'keypoints': None
                }
            
            # 添加标注信息
            img_to_labels[img_path]['cls'].append([item['class_id']])
            img_to_labels[img_path]['bboxes'].append(item['xywh'])
            img_to_labels[img_path]['attr_texts'].append(
                self.attr_to_text(item['attributes'])
            )
        
        # 转换为列表格式
        labels = []
        for img_path in sorted(img_to_labels.keys()):
            label = img_to_labels[img_path]
            label['cls'] = np.array(label['cls'], dtype=np.float32)
            label['bboxes'] = np.array(label['bboxes'], dtype=np.float32)
            labels.append(label)
        
        return labels
    
    def __getitem__(self, index):
        data = super().__getitem__(index)
        label = self.labels[index]
        
        # 添加属性文本到 batch
        if label.get('attr_texts') and len(label['attr_texts']) > 0:
            data['attr_text'] = label['attr_texts'][0]  # 使用第一个目标的属性
        else:
            data['attr_text'] = ""
        
        return data
    
    @staticmethod
    def collate_fn(batch):
        new_batch = YOLODataset.collate_fn(batch)
        new_batch['attr_text'] = [item['attr_text'] for item in batch]
        return new_batch
```

**设计亮点：**
- 兼容 YOLO 数据格式：继承自 `YOLODataset`，保持接口一致性
- 灵活的属性格式：支持任意结构的属性字典
- 自动文本转换：将结构化属性转换为自然语言描述

### 5. FiLMTrainer - 训练器

`FiLMTrainer` 扩展了 `DetectionTrainer`，处理属性文本的传递。

**关键方法：**

```python
class FiLMTrainer(DetectionTrainer):
    def build_dataset(self, img_path, mode="train", batch=None):
        """构建 JSONAttributeDataset"""
        return JSONAttributeDataset(
            img_path=None,
            json_path=img_path,  # 注意：img_path 实际是 JSON 路径
            data=self.data,
            task="detect",
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            # ... 其他参数
        )
    
    def train_step(self, batch):
        """训练步骤：传递 attr_texts 给模型"""
        self.optimizer.zero_grad()
        
        # 提取属性文本
        attr_texts = batch.get('attr_text', [])
        if isinstance(attr_texts, str):
            attr_texts = [attr_texts]
        elif not isinstance(attr_texts, list):
            attr_texts = []
        
        # 前向传播（传入 attr_texts）
        preds = self.model(batch['img'], attr_texts=attr_texts)
        
        # 计算损失
        loss, loss_items = self.model.loss(preds, batch)
        
        # 反向传播
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        
        return loss, loss_items
```

### 6. FiLMValidator - 验证器

`FiLMValidator` 扩展了 `DetectionValidator`，确保验证时也传递属性文本。

**关键实现：**

```python
class FiLMValidator(DetectionValidator):
    def build_dataset(self, img_path, mode="val", batch=None):
        """构建验证数据集"""
        return JSONAttributeDataset(
            img_path=None,
            json_path=img_path,
            data=self.data,
            task="detect",
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=False,  # 验证时无数据增强
            # ... 其他参数
        )
    
    def __call__(self, trainer=None, model=None):
        """验证流程：在推理时传递 attr_texts"""
        # ... 设置代码 ...
        
        for batch_i, batch in enumerate(bar):
            batch = self.preprocess(batch)
            
            # 推理时传递 attr_texts
            attr_texts = batch.get('attr_text', [])
            if not isinstance(attr_texts, list):
                attr_texts = [attr_texts] if isinstance(attr_texts, str) else []
            
            preds = model(batch["img"], augment=augment, attr_texts=attr_texts)
            
            # ... 后续处理 ...
```

## 数据流程

### 训练流程

1. **数据加载**：
   - `JSONAttributeDataset` 从 JSON 文件加载图像路径、标注和属性
   - 将属性字典转换为自然语言文本（如 "state: occluded, visibility: raining"）

2. **Batch 构建**：
   - `collate_fn` 将属性文本列表添加到 batch 中：`batch['attr_text'] = [text1, text2, ...]`

3. **前向传播**：
   - `AttributeEncoder` 将属性文本编码为 embedding：`attr_emb = encoder(attr_texts)`
   - 在检测头前，`FiLM` 模块对多尺度特征进行调制：`feat_modulated = FiLM(feat, attr_emb)`
   - 检测头处理调制后的特征，输出检测结果

4. **损失计算**：
   - 使用标准 YOLO 损失函数（box loss, cls loss, dfl loss）

### 推理流程

1. **输入**：图像 + 属性文本（可选）
2. **编码**：属性文本 → CLIP 编码 → 投影 → `attr_emb`
3. **特征提取**：图像 → Backbone → 多尺度特征 [P3, P4, P5]
4. **FiLM 调制**：每个尺度特征 + `attr_emb` → 调制后的特征
5. **检测**：调制后的特征 → 检测头 → 检测结果

## 技术亮点

### 1. CLIP 预训练权重的利用

- **优势**：无需从零训练文本编码器，利用 CLIP 在大规模数据上学习的文本-视觉对齐知识
- **实现**：冻结 CLIP 参数，只训练轻量级投影层
- **效率**：减少训练时间和显存占用

### 2. 多尺度特征调制

- **设计**：每个检测尺度（P3, P4, P5）都有独立的 FiLM 模块
- **原因**：不同尺度捕获不同层次的语义信息，需要独立的调制参数
- **效果**：允许模型在不同尺度上根据属性进行差异化处理

### 3. 灵活的属性格式

- **输入**：支持任意结构的属性字典
- **转换**：自动转换为自然语言文本
- **扩展性**：易于添加新的属性类型

### 4. 设备兼容性

- **自动检测**：自动检测并使用正确的设备（CPU/GPU）
- **设备迁移**：确保 CLIP 模型和投影层始终在同一设备上
- **显式处理**：在 forward 中显式处理设备迁移，避免运行时错误

## 使用示例

### 训练

```python
from ultralytics.models.film import YOLOFiLM

# 初始化模型
model = YOLOFiLM("yolov8n.yaml")

# 训练（需要提供包含属性信息的 JSON 数据集）
model.train(
    data="/path/to/dataset.yaml",  # YAML 中指定 JSON 文件路径
    epochs=50,
    imgsz=640,
    batch=4
)
```

### 验证

```python
# 验证
results = model.val(data="/path/to/dataset.yaml")
```

### 推理

```python
# 推理（带属性文本）
results = model.predict(
    source="image.jpg",
    attr_texts=["state: occluded, visibility: raining"]
)
```

## 修改文件清单

### 新增文件

1. `ultralytics/models/film/__init__.py` - 模块导出
2. `ultralytics/models/film/model.py` - YOLOFiLM 主接口
3. `ultralytics/models/film/train.py` - FiLMTrainer
4. `ultralytics/models/film/val.py` - FiLMValidator
5. `ultralytics/nn/modules/film.py` - AttributeEncoder 和 FiLM 模块
6. `ultralytics/nn/tasks_film.py` - DetectionModelFiLM
7. `ultralytics/data/dataset_json.py` - JSONAttributeDataset
8. `tools/convert_yolo_to_json.py` - YOLO 格式转 JSON 工具
9. `tools/train_yolo_film.py` - 训练脚本示例

### 修改文件

1. `ultralytics/data/__init__.py` - 导出 JSONAttributeDataset
2. `ultralytics/nn/modules/__init__.py` - 导出 FiLM 相关模块

## 总结

YOLOFiLM 通过集成 FiLM 机制和 CLIP 文本编码器，实现了基于自然语言属性的条件检测。该设计具有以下优势：

1. **模块化设计**：各组件职责清晰，易于维护和扩展
2. **预训练知识利用**：充分利用 CLIP 的预训练权重
3. **无缝集成**：与 Ultralytics YOLO 框架完美集成
4. **灵活扩展**：支持任意结构的属性信息
5. **高效训练**：冻结 CLIP 参数，只训练少量参数

该实现为条件目标检测提供了一个完整的解决方案，可以应用于需要根据额外属性信息进行检测的各种场景。

