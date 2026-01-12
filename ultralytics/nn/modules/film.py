import torch
import torch.nn as nn
from ultralytics.nn.text_model import build_text_model

class AttributeEncoder(nn.Module):
    """
    将结构化自然语言属性文本编码为连续的Embedding向量。
    使用CLIP模型的预训练权重进行编码，类似于YOLOE的文本分支。
    """
    def __init__(self, embed_dim=256, clip_variant="clip:ViT-B/32", device=None):
        """
        Args:
            embed_dim: 输出embedding维度
            clip_variant: CLIP模型变体，例如 "clip:ViT-B/32" 或 "mobileclip:blt"
            device: 设备，如果为None则自动检测
        """
        super().__init__()
        self.embed_dim = embed_dim
        
        # 自动检测设备
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        
        # 创建CLIP文本编码器（使用预训练权重）
        # 注意：CLIP模型会被注册为子模块，这样它会自动跟随主模型的设备移动
        self.clip_model = build_text_model(clip_variant, device=device)
        self.clip_model.eval()  # 设置为评估模式
        # 冻结CLIP模型参数，只训练投影层
        for param in self.clip_model.parameters():
            param.requires_grad = False
        # 确保CLIP模型被注册为子模块（虽然build_text_model返回的已经是nn.Module）
        # 这样它会自动跟随主模型的设备移动
        
        # 获取CLIP的输出维度
        # CLIP ViT-B/32 输出 512 维，MobileCLIP 可能不同
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
        
        # 将投影层移动到相同设备
        self.projector.to(device)

    def forward(self, attr_texts):
        """
        Args:
            attr_texts: list[str] 或 None，属性文本列表（每个元素对应一个batch item）
        Returns:
            x: [batch_size, embed_dim] 属性embedding
        """
        # 动态获取当前设备（确保与主模型在同一设备上）
        current_device = next(self.projector.parameters()).device
        
        # 处理 None 或空列表
        if not attr_texts or len(attr_texts) == 0:
            # 如果没有属性文本，返回零向量（batch_size=1）
            batch_size = 1
            return torch.zeros((batch_size, self.embed_dim), device=current_device)
        
        # 过滤空字符串，使用默认文本
        processed_texts = []
        for text in attr_texts:
            if text and isinstance(text, str) and text.strip():
                processed_texts.append(text.strip())
            else:
                # 使用默认文本
                processed_texts.append("a normal object")
        
        # 确保CLIP模型在正确的设备上（作为子模块应该自动跟随，但显式确保）
        self.clip_model.to(current_device)
        self.clip_model.eval()  # 确保CLIP模型处于评估模式
        
        # 使用CLIP编码文本（CLIP模型参数已冻结，使用no_grad提高效率）
        with torch.no_grad():
            # Tokenize文本（tokenize方法应该已经将tokens移到设备上，但显式确保）
            tokens = self.clip_model.tokenize(processed_texts)
            # 确保tokens在正确的设备上（双重保险）
            tokens = tokens.to(current_device)
            # 编码为特征向量（已经L2归一化）
            clip_features = self.clip_model.encode_text(tokens)
            # 确保clip_features在正确的设备上（双重保险）
            clip_features = clip_features.to(current_device)
        
        # 通过投影层映射到目标维度（投影层可训练，梯度可以流动）
        x = self.projector(clip_features)
        return x

class FiLM(nn.Module):
    """
    FiLM (Feature-wise Linear Modulation) 层。
    作用：根据属性特征，对视觉特征进行仿射变换 (Scale & Shift)。
    """
    def __init__(self, feat_channels, attr_dim):
        super().__init__()
        self.scale_gen = nn.Linear(attr_dim, feat_channels)
        self.shift_gen = nn.Linear(attr_dim, feat_channels)
        
        nn.init.zeros_(self.scale_gen.weight)
        nn.init.constant_(self.scale_gen.bias, 1.0)
        nn.init.zeros_(self.shift_gen.weight)
        nn.init.zeros_(self.shift_gen.bias)

    def forward(self, x, attr_emb):
        """
        x: [B, C, H, W]
        attr_emb: [B, attr_dim]
        """
        if attr_emb is None:
            return x
            
        batch_size, channels, _, _ = x.shape
        
        # 生成调制参数 [B, C, 1, 1]
        gamma = self.scale_gen(attr_emb).view(batch_size, channels, 1, 1)
        beta = self.shift_gen(attr_emb).view(batch_size, channels, 1, 1)
        
        return x * gamma + beta