import argparse
import os
import types
import shutil
from tqdm import tqdm

import cv2
import torch

from ultralytics import YOLO
from ultralytics.utils import YAML

from osr_utils import get_directory_size_system


def _predict_once_custom(self, x, profile=False, visualize=False, embed=None):
    """
    A customized version of _predict_once in DetectionModel.
    Change list:
    - When use embed in _predict_once, the embeddings are no longer pooled and flattened, 
      instead, the embeddings remain the feature maps generated from embedding layers.
    """
    y, dt, embeddings = [], [], []  # outputs
    for m in self.model:
        if m.f != -1:  # if not from previous layer
            x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
        if profile:
            self._profile_one_layer(m, x, dt)
        x = m(x)  # run
        y.append(x if m.i in self.save else None)  # save output
        if embed and m.i in embed:
            # embeddings.append(nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
            # if m.i == max(embed):
                # return torch.unbind(torch.cat(embeddings, 1), dim=0)
            embeddings.append(x)
            if m.i == max(embed):
                return embeddings
    return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_cfg", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    data_cfg = YAML.load(args.data_cfg)
    classes = [data_cfg["names"][k] for k in sorted(data_cfg["names"].keys())]
    train_images_dir = os.path.join(os.path.dirname(args.data_cfg), data_cfg["train"])
    train_labels_dir = os.path.join(os.path.dirname(args.data_cfg), data_cfg["train"].replace("images", "labels"))
    model = YOLO(args.model_path)
    model.model._predict_once = types.MethodType(_predict_once_custom, model.model)

    if os.path.exists(args.save_dir):
        shutil.rmtree(args.save_dir)
    os.makedirs(args.save_dir)
    os.makedirs(os.path.join(args.save_dir, "all_samples"))

    instance_features = {name: 0. for name in classes}
    instance_features_count = {name: 0 for name in classes}
    samples = {name: [] for name in classes}
    images = os.listdir(train_images_dir)
    labels = os.listdir(train_labels_dir)
    for i in tqdm(range(len(images)), desc="Inferring through all samples"):
        image_path = os.path.join(train_images_dir, images[i])
        image_array = cv2.imread(image_path)
        image_size = image_array.shape[:2]
        label_path = os.path.join(train_labels_dir, labels[i])
        with open(label_path, "r") as f:
            label = [line.strip() for line in f.readlines()]
        for box in label:
            cls, cx, cy, w, h = box.split(" ")
            cls, cx, cy, w, h = int(cls), float(cx), float(cy), float(w), float(h)
            # crop the image
            x1, y1, x2, y2 = int((cx - w/2) * image_size[1]), int((cy - h/2) * image_size[0]),\
                int((cx + w/2) * image_size[1]), int((cy + h/2) * image_size[0])
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(image_size[1]-1, x2), min(image_size[0]-1, y2)
            cropped_image = image_array[y1:y2+1, x1:x2+1]
            n = instance_features_count[classes[cls]]
            cv2.imwrite(os.path.join(args.save_dir, "all_samples", f"{classes[cls]}_sample_{n}.jpg"), cropped_image)
            samples[classes[cls]].append(os.path.join(args.save_dir, "all_samples", f"{classes[cls]}_sample_{n}.jpg"))
            embedding = model.embed(os.path.join(args.save_dir, "all_samples", f"{classes[cls]}_sample_{n}.jpg"), verbose=False)[-1]
            feature = torch.mean(embedding, dim=(2, 3)).squeeze(0)
            instance_features[classes[cls]] = instance_features[classes[cls]] * (n/(n+1)) + feature/(n+1)
            instance_features_count[classes[cls]] += 1
    
    for name, samples_list in samples.items():
        similarity = []
        for sample in tqdm(samples_list, desc=f"Picking {name}'s best samples"):
            embedding = model.embed(sample, verbose=False)[-1]
            feature = torch.mean(embedding, dim=(2, 3)).squeeze(0)
            sim = torch.cosine_similarity(feature, instance_features[name], dim=0)
            similarity.append({"sample_path": sample, "sim": sim})
        top_k_samples = [sample["sample_path"] for sample in sorted(similarity, key=lambda x: x["sim"], reverse=True)[:args.k]]
        for k, sample_path in enumerate(top_k_samples):
            shutil.copy(sample_path, os.path.join(args.save_dir, f"{name}_best_sample_{k}.jpg"))
    shutil.rmtree(os.path.join(args.save_dir, "all_samples"))

    system_size = get_directory_size_system(args.save_dir)
    if system_size >= 0:
        print(f"\033[94mINFO:\033[0m Memory bank occupies {system_size/1024:.2f} KB")
    else:
        print(f"\033[94mINFO:\033[0m Memory bank occupies ? KB")