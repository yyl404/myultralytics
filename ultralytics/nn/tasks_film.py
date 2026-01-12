import torch
import torch.nn as nn
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules import Detect, Segment, Pose, OBB
from ultralytics.nn.modules.film import AttributeEncoder, FiLM
from ultralytics.utils.plotting import feature_visualization
from ultralytics.utils import LOGGER

class DetectionModelFiLM(DetectionModel):
    """
    继承自 DetectionModel (YOLOE/YOLOv8 的基类)。
    增加了 FiLM 分支用于处理额外属性。
    """
    def __init__(self, cfg='yolov8n.yaml', ch=3, nc=None, verbose=True):
        # 初始化 film_modules 为 None，避免在父类 __init__ 调用 forward 时出错
        self.film_modules = None
        self.attr_dim = 256
        self.attr_encoder = None
        
        super().__init__(cfg, ch, nc, verbose)
        
        # 1. 定义属性编码器
        # 使用CLIP模型的预训练权重进行文本编码，类似于YOLOE的文本分支
        # 自动检测设备（在super().__init__之后，self.model已经存在）
        device = next(self.model.parameters()).device
        self.attr_encoder = AttributeEncoder(embed_dim=self.attr_dim, clip_variant="clip:ViT-B/32", device=device)
        
        # 2. 为 Detect Head 的每个输入特征层通过挂载 FiLM 模块
        # 我们寻找最后一层 (Head)
        m = self.model[-1]
        if isinstance(m, (Detect, Segment, Pose, OBB)):
            self.film_modules = nn.ModuleList()
            # m.nl 是 Head 的层数 (通常是3: P3, P4, P5)
            for i in range(m.nl):
                # 获取 Head 第 i 个输入的通道数
                # m.cv2[i] 是 nn.Sequential，第一个元素是 Conv 模块
                c_in = m.cv2[i][0].conv.in_channels 
                self.film_modules.append(FiLM(c_in, self.attr_dim))
            
            LOGGER.info(f"Initialized FiLM modules for {m.nl} detection scales.")
        else:
            LOGGER.warning("Could not find Detect Head to attach FiLM modules!")
            self.film_modules = None

    def forward(self, x, profile=False, visualize=False, augment=False, embed=None, attr_texts=None):
        """
        重写 forward 以接收 attr_texts（属性文本列表）。
        支持字典输入（训练时）和张量输入（推理时）。
        """
        # 如果 x 是字典（训练模式），需要先进行预测，然后计算损失
        if isinstance(x, dict):
            # 从字典中提取 attr_texts
            if attr_texts is None:
                attr_texts = x.get('attr_text')
                # 如果是单个字符串，转换为列表
                if isinstance(attr_texts, str):
                    attr_texts = [attr_texts]
            # 获取图像张量
            img = x['img']
            # 进行预测（使用 attr_texts）
            preds = self.predict(img, profile=profile, visualize=visualize, augment=augment, embed=embed, attr_texts=attr_texts)
            # 计算损失
            return self.loss(x, preds)
        
        # 推理模式：x 是张量
        return self.predict(x, profile=profile, visualize=visualize, augment=augment, embed=embed, attr_texts=attr_texts)
    
    def predict(self, x, profile=False, visualize=False, augment=False, embed=None, attr_texts=None):
        """
        重写 predict 以支持 attr_texts（属性文本列表）。
        """
        if augment:
            return self._forward_augment(x) # 不支持 TTA + FiLM
            
        # 1. 编码属性文本
        attr_emb = None
        if attr_texts is not None:
            attr_emb = self.attr_encoder(attr_texts)
            
        return self._predict_once(x, profile, visualize, attr_emb)

    def _predict_once(self, x, profile=False, visualize=False, attr_emb=None):
        """
        重写推理循环。在特征进入 Head 之前应用 FiLM。
        """
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            
            if profile:
                self._profile_one_layer(m, x, dt)
            
            # --- 关键修改: 在 Head 之前拦截 x ---
            if isinstance(m, (Detect, Segment, Pose, OBB)) and self.film_modules is not None and attr_emb is not None:
                # x 是一个列表 [P3, P4, P5]
                # 我们依次对每个尺度的特征图应用 FiLM
                x_modified = []
                for i, feat in enumerate(x):
                    # 应用对应的 FiLM 模块
                    feat_out = self.film_modules[i](feat, attr_emb)
                    x_modified.append(feat_out)
                x = x_modified
            # -----------------------------------

            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output

            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x