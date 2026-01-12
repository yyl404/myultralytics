from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.data import build_dataloader
from ultralytics.data.dataset_json import JSONAttributeDataset
from ultralytics.nn.tasks_film import DetectionModelFiLM
from ultralytics.utils import RANK, colorstr

class FiLMTrainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        """
        返回我们需要使用的 DetectionModelFiLM 实例。
        """
        # 创建我们的自定义模型
        model = DetectionModelFiLM(cfg, nc=self.data['nc'], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def build_dataset(self, img_path, mode="train", batch=None):
        """
        构建 JSONAttributeDataset。
        """
        # 注意：这里的 img_path 实际上是我们在 yaml 中配置的 json 路径
        return JSONAttributeDataset(
            img_path=None, # 不再通过文件夹扫描
            json_path=img_path, 
            data=self.data,  # 传递 data 参数
            task="detect",
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=self.args.rect,
            cache=self.args.cache,
            single_cls=self.args.single_cls,
            stride=int(self.stride),
            pad=0.0 if mode == "train" else 0.5,
            prefix=colorstr(f"{mode}: "),
            classes=None  # 不过滤类别，使用所有类别
        )

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """覆盖 get_dataloader 以便传入 json 路径"""
        dataset = self.build_dataset(dataset_path, mode, batch_size)
        loader = build_dataloader(dataset, batch_size, self.args.workers, shuffle=(mode == "train"), rank=rank)
        return loader

    def preprocess_batch(self, batch):
        """将 attr_text 移动到 GPU（文本不需要移动，但保留方法以兼容）"""
        batch = super().preprocess_batch(batch)
        # attr_text 是字符串列表，不需要移动到 GPU
        # 但我们可以在这里进行验证
        if 'attr_text' in batch:
            # 确保 attr_text 是列表
            if not isinstance(batch['attr_text'], list):
                batch['attr_text'] = [batch['attr_text']] if isinstance(batch['attr_text'], str) else []
        return batch

    def train_step(self, batch):
        """调用 model.forward 时传入 attr_texts"""
        self.optimizer.zero_grad()
        
        # 核心改动：传入 attr_texts（属性文本列表）
        attr_texts = batch.get('attr_text', [])
        # 确保是列表格式
        if isinstance(attr_texts, str):
            attr_texts = [attr_texts]
        elif not isinstance(attr_texts, list):
            attr_texts = []
        
        preds = self.model(batch['img'], attr_texts=attr_texts)
        
        loss, loss_items = self.model.loss(preds, batch)
        
        # Backward ...
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        
        return loss, loss_items