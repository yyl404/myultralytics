"""
对比 NSGP 与 EWPR 两个实验在 VOC 15-5 设置下的 mAP50 曲线（前 15 个 epoch）。

数据来源：
- NSGP: /root/myultralytics/runs/yolov8l_voc_15_5_fromscratch_nsgp+pseudo_label/task-2/train3/results.csv
- EWPR: /root/myultralytics/runs/yolov8l_voc_15_5_fromscratch_ewpr+pseudo_label/task-2/train-w1/results.csv

输出：
- 一张包含两个方法 mAP50 折线的图，保存在：
  /root/myultralytics/runs/yolov8l_voc_15_5_map50_curve.png
"""

import os

import matplotlib.pyplot as plt
import pandas as pd


NSGP_CSV = "/root/myultralytics/runs/yolov8l_voc_15_5_fromscratch_nsgp+pseudo_label/task-2/train3/results.csv"
EWPR_CSV = "/root/myultralytics/runs/yolov8l_voc_15_5_fromscratch_ewpr+pseudo_label/task-2/train-w1/results.csv"
OUT_FIG = "/root/myultralytics/runs/yolov8l_voc_15_5_map50_curve.png"


def load_map50(csv_path: str, max_epochs: int = 15):
    df = pd.read_csv(csv_path)
    # Ultralytics 默认 epoch 从 1 开始，这里只取前 max_epochs 行
    df = df.head(max_epochs)
    if "metrics/mAP50(B)" not in df.columns:
        raise KeyError(f"'metrics/mAP50(B)' not found in {csv_path}")
    epochs = df["epoch"].to_numpy()
    map50 = df["metrics/mAP50(B)"].to_numpy()
    return epochs, map50


def main():
    epochs_nsgp, map50_nsgp = load_map50(NSGP_CSV, max_epochs=15)
    epochs_ewpr, map50_ewpr = load_map50(EWPR_CSV, max_epochs=15)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_nsgp, map50_nsgp, marker="o", color="tab:blue", label="NSGP mAP50")
    plt.plot(epochs_ewpr, map50_ewpr, marker="s", color="tab:orange", label="EWPR mAP50")

    plt.xlabel("Epoch")
    plt.ylabel("mAP50")
    plt.title("VOC 15-5: mAP50 vs Epoch (NSGP vs EWPR)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
    plt.savefig(OUT_FIG, dpi=300)
    plt.close()
    print(f"Saved mAP50 curve figure to: {OUT_FIG}")


if __name__ == "__main__":
    main()


