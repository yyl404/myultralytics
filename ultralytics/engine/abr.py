# ultralytics/engine/abr.py
import json, random
from pathlib import Path
import cv2
import numpy as np
import torch

class ABRReplay:
    def __init__(
        self,
        memory_json,
        ratio=(1,1,2),
        iou_thr=0.05,
        max_mix_boxes=2,
        mix_beta=32.0,
        mosaic_scale=(0.4, 0.6),
        seed=0,
    ):
        self.rng = random.Random(seed)
        self.ratio = ratio
        self.iou_thr = iou_thr
        self.max_mix_boxes = max_mix_boxes
        self.mix_beta = mix_beta
        self.mosaic_scale = mosaic_scale

        with open(memory_json, "r") as f:
            self.memory = json.load(f)

        self.by_cls = {}
        self.cache = []
        for item in self.memory:
            img = cv2.imread(item["crop_path"])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            item["crop"] = img
            self.cache.append(item)
            self.by_cls.setdefault(item["cls"], []).append(item)

    def _sample_mode(self):
        return self.rng.choices(["mix", "mosaic", "new"], weights=self.ratio, k=1)[0]

    def _xywhn_to_xyxy(self, box, W, H):
        xc, yc, w, h = box
        x1 = (xc - w/2) * W
        y1 = (yc - h/2) * H
        x2 = (xc + w/2) * W
        y2 = (yc + h/2) * H
        return np.array([x1, y1, x2, y2], dtype=np.float32)

    def _xyxy_to_xywhn(self, box, W, H):
        x1, y1, x2, y2 = box
        xc = ((x1 + x2) / 2.0) / W
        yc = ((y1 + y2) / 2.0) / H
        w  = (x2 - x1) / W
        h  = (y2 - y1) / H
        return np.array([xc, yc, w, h], dtype=np.float32)

    def _iou_xyxy(self, a, b):
        xx1 = max(a[0], b[0]); yy1 = max(a[1], b[1])
        xx2 = min(a[2], b[2]); yy2 = min(a[3], b[3])
        inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
        area_a = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
        area_b = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
        union = area_a + area_b - inter + 1e-6
        return inter / union

    def _paste_alpha(self, img, crop, x1, y1, x2, y2, lam):
        crop_rs = cv2.resize(crop, (x2-x1, y2-y1))
        crop_rs = torch.from_numpy(crop_rs).permute(2,0,1).float().to(img.device) / 255.0
        img[:, y1:y2, x1:x2] = lam * img[:, y1:y2, x1:x2] + (1 - lam) * crop_rs
        return img

    def _mix_one(self, img, boxes, cls):
        C, H, W = img.shape
        old_boxes = [self._xywhn_to_xyxy(b, W, H) for b in boxes.cpu().numpy()] if len(boxes) else []
        num_to_mix = self.rng.randint(1, self.max_mix_boxes)

        for _ in range(num_to_mix):
            item = self.rng.choice(self.cache)
            crop = item["crop"]
            ch, cw = crop.shape[:2]

            # 可加一个轻微随机缩放
            scale = self.rng.uniform(0.8, 1.2)
            nh = max(8, int(ch * scale))
            nw = max(8, int(cw * scale))
            nh = min(nh, H // 2)
            nw = min(nw, W // 2)

            placed = False
            for _try in range(50):
                x1 = self.rng.randint(0, W - nw)
                y1 = self.rng.randint(0, H - nh)
                x2, y2 = x1 + nw, y1 + nh
                new_box = np.array([x1, y1, x2, y2], dtype=np.float32)

                bad = False
                for b in old_boxes:
                    if self._iou_xyxy(new_box, b) > self.iou_thr:
                        bad = True
                        break
                if bad:
                    continue

                lam = np.random.beta(self.mix_beta, self.mix_beta)
                img = self._paste_alpha(img, crop, x1, y1, x2, y2, lam)

                box_n = self._xyxy_to_xywhn(new_box, W, H)
                box_t = torch.tensor(box_n, device=img.device).unsqueeze(0)
                cls_t = torch.tensor([[item["cls"]]], device=img.device, dtype=cls.dtype)

                boxes = torch.cat([boxes, box_t], dim=0)
                cls = torch.cat([cls, cls_t], dim=0)
                old_boxes.append(new_box)
                placed = True
                break

            if not placed:
                continue

        return img, boxes, cls

    def _mosaic_one(self, img, boxes, cls):
        C, H, W = img.shape
        quad = [
            (0,      0,      W//2, H//2),
            (W//2,   0,      W,    H//2),
            (0,      H//2,   W//2, H),
            (W//2,   H//2,   W,    H),
        ]

        self.rng.shuffle(quad)
        for q in quad:
            item = self.rng.choice(self.cache)
            crop = item["crop"]
            x1, y1, x2, y2 = q
            qw, qh = x2 - x1, y2 - y1

            mu = self.rng.uniform(self.mosaic_scale[0], self.mosaic_scale[1])
            nw = max(8, int(qw * mu))
            nh = max(8, int(qh * mu))

            px1 = x1 + (qw - nw) // 2
            py1 = y1 + (qh - nh) // 2
            px2, py2 = px1 + nw, py1 + nh

            img = self._paste_alpha(img, crop, px1, py1, px2, py2, lam=0.0)

            box_n = self._xyxy_to_xywhn(np.array([px1, py1, px2, py2], dtype=np.float32), W, H)
            box_t = torch.tensor(box_n, device=img.device).unsqueeze(0)
            cls_t = torch.tensor([[item["cls"]]], device=img.device, dtype=cls.dtype)

            boxes = torch.cat([boxes, box_t], dim=0)
            cls = torch.cat([cls, cls_t], dim=0)

        return img, boxes, cls

    def __call__(self, batch):
        imgs = batch["img"]
        B, _, H, W = imgs.shape

        new_boxes, new_cls, new_batch_idx = [], [], []

        for i in range(B):
            mask = batch["batch_idx"] == i
            boxes_i = batch["bboxes"][mask].clone()
            cls_i = batch["cls"][mask].clone()
            img_i = imgs[i]

            mode = self._sample_mode()
            if mode == "mix":
                img_i, boxes_i, cls_i = self._mix_one(img_i, boxes_i, cls_i)
            elif mode == "mosaic":
                img_i, boxes_i, cls_i = self._mosaic_one(img_i, boxes_i, cls_i)

            imgs[i] = img_i
            if len(boxes_i):
                new_boxes.append(boxes_i)
                new_cls.append(cls_i)
                new_batch_idx.append(torch.full((len(boxes_i),), i, device=imgs.device, dtype=torch.long))

        batch["img"] = imgs
        batch["bboxes"] = torch.cat(new_boxes, dim=0) if new_boxes else torch.empty((0,4), device=imgs.device)
        batch["cls"] = torch.cat(new_cls, dim=0) if new_cls else torch.empty((0,1), device=imgs.device, dtype=torch.long)
        batch["batch_idx"] = torch.cat(new_batch_idx, dim=0) if new_batch_idx else torch.empty((0,), device=imgs.device, dtype=torch.long)
        return batch