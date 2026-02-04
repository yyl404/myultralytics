"""
可视化不同训练方法在特定卷积层上的权重更新轨迹（按epoch），
并将权重差值投影到对应层PCA缓存的前两个主成分子空间中。

当前脚本针对实验：
- NSGP:  /root/myultralytics/runs/yolov8l_voc_15_5_fromscratch_nsgp+pseudo_label/task-2/train3
- ESPReg:  /root/myultralytics/runs/yolov8l_voc_15_5_fromscratch_ewpr+pseudo_label/task-2/train

步骤：
1. 读取基准模型（task-1-best-expanded.pt）和每个epoch的权重（epoch0.pt ~ epoch14.pt）
2. 提取指定层（默认：model.9.cv1.conv）的卷积核权重
3. 计算相对于基准模型的权重差值
4. 使用对应方法 task-1 的 PCA cache，将差值在前两个主成分方向上投影，得到2D坐标序列
5. 保存两种方法的坐标序列，并在同一张图中画出带箭头的轨迹
"""

import os
import sys
from typing import Dict, List, Tuple

import joblib
import numpy as np
import torch
import matplotlib.pyplot as plt


# 确保能反序列化 pca_cache 中的 IncrementalPCAonGPU 对象，
# 同时让 Python 能找到本地的 ultralytics 包用于反序列化模型。
TOOLS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(TOOLS_DIR)
for p in (TOOLS_DIR, PROJECT_ROOT):
    if p not in sys.path:
        sys.path.append(p)


LAYER_NAME = "model.9.cv1.conv"
# 使用从第 1 个 epoch 到第 15 个 epoch 的权重：
# 这里文件名是 epoch0.pt ~ epoch14.pt，因此我们把 epoch0 视作“第 1 个 epoch”。
EPOCH_INDICES = list(range(15))  # 0~14 -> 第 1~15 个 epoch


class MethodConfig:
    def __init__(self, name: str, root_dir: str, train_subdir: str):
        self.name = name
        self.root_dir = root_dir
        self.task2_dir = os.path.join(root_dir, "task-2")
        self.task1_pca_cache = os.path.join(root_dir, "task-1", "pca_cache.pkl")
        self.base_ckpt = os.path.join(self.task2_dir, "task-1-best-expanded.pt")
        self.train_dir = os.path.join(self.task2_dir, train_subdir)
        self.weights_dir = os.path.join(self.train_dir, "weights")


METHODS: Dict[str, MethodConfig] = {
    "NSGP": MethodConfig(
        name="NSGP",
        root_dir="/root/myultralytics/runs/yolov8l_voc_15_5_fromscratch_nsgp+pseudo_label",
        train_subdir="train6",
    ),
    "ESPReg": MethodConfig(
        name="ESPReg",
        root_dir="/root/myultralytics/runs/yolov8l_voc_15_5_fromscratch_ewpr+pseudo_label",
        train_subdir="train",
    ),
}


def load_pca_components(pca_path: str, layer_name: str) -> torch.Tensor:
    """从 pca_cache 中读取指定层的 PCA 主成分（返回 shape: [n_components, feat_dim] 的 Tensor）。"""
    if not os.path.isfile(pca_path):
        raise FileNotFoundError(f"PCA cache not found: {pca_path}")

    cache = joblib.load(pca_path)
    if layer_name not in cache:
        raise KeyError(f"Layer {layer_name} not found in PCA cache: {pca_path}")

    ops = cache[layer_name]
    if len(ops) != 1:
        # 当前层 groups=1，一般只会有一个 operator；若不为1则给出提示
        raise ValueError(f"Expected 1 group PCA operator for {layer_name}, got {len(ops)}")

    components = ops[0].components_
    if not isinstance(components, torch.Tensor):
        components = torch.from_numpy(np.asarray(components)).float()
    else:
        components = components.float()
    return components.cpu()  # [n_components, feat_dim] on CPU


def load_conv_weight_from_ckpt(ckpt_path: str, layer_name: str) -> torch.Tensor:
    """
    从 Ultralytics 训练产生的 checkpoint 中读取指定卷积层的权重。

    对于 task-2 中每个 epochX.pt：
        state['ema'] 是 DetectionModel，卷积权重在其 state_dict 中。
    对于 task-1-best-expanded.pt：
        state['model'] 是 DetectionModel。
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    model = None
    if isinstance(state, dict):
        if "ema" in state and state["ema"] is not None:
            model = state["ema"]
        elif "model" in state and state["model"] is not None:
            model = state["model"]

    if model is None:
        raise RuntimeError(f"Cannot find model/ema in checkpoint: {ckpt_path}")

    sd = model.state_dict()
    if layer_name + ".weight" not in sd:
        raise KeyError(f"Weight {layer_name + '.weight'} not found in checkpoint: {ckpt_path}")

    w = sd[layer_name + ".weight"].float()  # [c_out, c_in, kH, kW]
    return w


def compute_epoch_coords_for_method(
    cfg: MethodConfig,
    layer_name: str,
    epoch_indices: List[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    对某个方法（NSGP / ESPReg）计算指定层在各个 epoch 的投影坐标序列。

    返回:
        epochs: shape [T]，从1开始计数（1~15）
        coords: shape [T, 2]，对应 (PC1, PC2)
    """
    print(f"=== Processing method: {cfg.name} ===")
    print(f"Weights dir:    {cfg.weights_dir}")
    print(f"PCA cache:      {cfg.task1_pca_cache}")

    # 1. 加载 PCA 主成分
    components = load_pca_components(cfg.task1_pca_cache, layer_name)  # [n_comp, feat_dim]
    pc2 = components[:2]  # [2, feat_dim]

    # 2. 加载“第 1 个 epoch”的权重作为基准（权重差值的参考点）
    base_epoch_idx = epoch_indices[0]
    base_ckpt = os.path.join(cfg.weights_dir, f"epoch{base_epoch_idx}.pt")
    print(f"Base epoch for diffs: epoch{base_epoch_idx} ({base_ckpt})")
    w_base = load_conv_weight_from_ckpt(base_ckpt, layer_name)  # [c_out, c_in, kH, kW]
    c_out = w_base.shape[0]
    w_base_flat = w_base.view(c_out, -1)  # [c_out, feat_dim]

    feat_dim = w_base_flat.shape[1]
    if pc2.shape[1] != feat_dim:
        raise ValueError(
            f"PCA feature dim mismatch for {layer_name}: "
            f"components feat_dim={pc2.shape[1]}, weight feat_dim={feat_dim}"
        )

    coords: List[List[float]] = []
    epoch_ids: List[int] = []

    for ei in epoch_indices:
        ckpt = os.path.join(cfg.weights_dir, f"epoch{ei}.pt")
        print(f"  Loading {cfg.name} epoch{ei} from {ckpt}")
        w_epoch = load_conv_weight_from_ckpt(ckpt, layer_name)  # [c_out, c_in, kH, kW]
        if w_epoch.shape != w_base.shape:
            raise ValueError(
                f"Weight shape mismatch at epoch{ei}: base {w_base.shape}, epoch {w_epoch.shape}"
            )

        w_epoch_flat = w_epoch.view(c_out, -1)  # [c_out, feat_dim]
        diff = w_epoch_flat - w_base_flat       # [c_out, feat_dim]

        # [c_out, feat_dim] @ [feat_dim, 2] -> [c_out, 2]
        proj = diff @ pc2.T  # [c_out, 2]
        coord = proj.mean(dim=0)  # [2]，对所有输出通道取平均

        coords.append(coord.cpu().numpy().tolist())
        # 题目中说“第1个epoch到第15个epoch”，这里把 epoch0 记为 1，epoch14 记为 15
        epoch_ids.append(ei + 1)

    coords_arr = np.asarray(coords, dtype=np.float32)  # [T, 2]
    epochs_arr = np.asarray(epoch_ids, dtype=np.int32)  # [T]
    return epochs_arr, coords_arr


def save_coords(save_dir: str, method_name: str, epochs: np.ndarray, coords: np.ndarray) -> None:
    os.makedirs(save_dir, exist_ok=True)
    npy_path = os.path.join(save_dir, f"{method_name}_model9_cv1_conv_epoch_coords.npy")
    csv_path = os.path.join(save_dir, f"{method_name}_model9_cv1_conv_epoch_coords.csv")

    data = np.concatenate(
        [epochs.reshape(-1, 1).astype(np.float32), coords.astype(np.float32)], axis=1
    )  # [T, 3] -> epoch, pc1, pc2

    np.save(npy_path, data)
    np.savetxt(
        csv_path,
        data,
        delimiter=",",
        header="epoch,pc1,pc2",
        comments="",
    )
    print(f"Saved {method_name} coords to:\n  {npy_path}\n  {csv_path}")


def plot_trajectories(
    out_path: str,
    epochs_nsgp: np.ndarray,
    coords_nsgp: np.ndarray,
    epochs_ewpr: np.ndarray,
    coords_ewpr: np.ndarray,
) -> None:
    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    # 计算范围以便设置箭头尺寸（已注释，但保留用于可能的未来使用）
    all_x = np.concatenate([coords_nsgp[:, 0], coords_ewpr[:, 0]])
    all_y = np.concatenate([coords_nsgp[:, 1], coords_ewpr[:, 1]])
    x_range = all_x.max() - all_x.min() if all_x.size > 0 else 1.0
    y_range = all_y.max() - all_y.min() if all_y.size > 0 else 1.0
    scale = max(x_range, y_range)
    head_width = 0.02 * scale
    head_length = 0.03 * scale

    def _plot_single(epochs: np.ndarray, coords: np.ndarray, color: str, label: str):
        x = coords[:, 0]
        y = coords[:, 1]
        # 画连续轨迹：使用线条连接各个点
        ax.plot(x, y, color=color, alpha=0.8, linewidth=2, label=label)
        
        # 暂时注释掉箭头绘制逻辑
        # # 画连续轨迹：前一个 epoch 的终点指向后一个 epoch
        # for i in range(len(x) - 1):
        #     dx = x[i + 1] - x[i]
        #     dy = y[i + 1] - y[i]
        #     ax.arrow(
        #         x[i],
        #         y[i],
        #         dx,
        #         dy,
        #         length_includes_head=True,
        #         head_width=head_width,
        #         head_length=head_length,
        #         color=color,
        #         alpha=0.8,
        #     )
        
        # 在起点和终点画标记，避免文字过度拥挤
        ax.scatter(x[0], y[0], color=color, marker="o", label=f"{label} start", s=100)
        ax.scatter(x[-1], y[-1], color=color, marker="s", label=f"{label} end", s=100)

    _plot_single(epochs_nsgp, coords_nsgp, color="tab:blue", label="NSGP")
    _plot_single(epochs_ewpr, coords_ewpr, color="tab:orange", label="ESPReg")

    # 设置坐标轴范围，使其以0点为中心
    all_x = np.concatenate([coords_nsgp[:, 0], coords_ewpr[:, 0]])
    all_y = np.concatenate([coords_nsgp[:, 1], coords_ewpr[:, 1]])
    
    if all_x.size > 0 and all_y.size > 0:
        x_max = max(abs(all_x.max()), abs(all_x.min()))
        y_max = max(abs(all_y.max()), abs(all_y.min()))
        # 添加一些边距（10%）
        x_max = x_max * 1.1 if x_max > 0 else 1.0
        y_max = y_max * 1.1 if y_max > 0 else 1.0
        ax.set_xlim(-x_max, x_max)
        ax.set_ylim(-y_max, y_max)
    
    # 添加0点参考线
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"Epoch Trajectories on PCA Subspace ({LAYER_NAME})")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved trajectory figure to: {out_path}")


def main():
    # 计算两种方法的坐标序列
    epochs_nsgp, coords_nsgp = compute_epoch_coords_for_method(
        METHODS["NSGP"], LAYER_NAME, EPOCH_INDICES
    )
    epochs_ewpr, coords_ewpr = compute_epoch_coords_for_method(
        METHODS["ESPReg"], LAYER_NAME, EPOCH_INDICES
    )

    # 保存坐标（分别保存在各自方法的 task-2 目录下）
    save_coords(METHODS["NSGP"].task2_dir, "NSGP", epochs_nsgp, coords_nsgp)
    save_coords(METHODS["ESPReg"].task2_dir, "ESPReg", epochs_ewpr, coords_ewpr)

    # 绘制联合轨迹图，放在一个公共位置（VOC_15_5 对应的根目录下）
    out_fig = "/root/myultralytics/runs/yolov8l_voc_15_5_model9_cv1_conv_epoch_trajectories.png"
    plot_trajectories(out_fig, epochs_nsgp, coords_nsgp, epochs_ewpr, coords_ewpr)


if __name__ == "__main__":
    main()


