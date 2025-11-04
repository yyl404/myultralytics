"""
聚合逐类别的confusion martix，保留三个不同的行和列：old classes, new classes, background

使用方式：
    python tools/parse_confusion_matrix.py \
    --confusion_matrix_path runs/yolov8l_voc_inc_15_5_fromscratch/eval_model-1_task-2-full-labels/confusion_matrix.csv \
    --old_classes aeroplane bicycle bird boat bottle bus car cat chair cow diningtable dog horse motorbike person \
    --new_classes pottedplant sheep sofa train tvmonitor \
    --background_name background \
    --save_dir runs/yolov8l_voc_inc_15_5_fromscratch/eval_model-1_task-2-full-labels/confusion_matrix_analysis

输出三个可视化矩阵：
    1. 原始聚合混淆矩阵
    2. 横向归一化矩阵
    3. 纵向归一化矩阵

输入的csv格式可以参考tools/eval.py的输出格式
"""
import argparse
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_confusion_matrix(csv_path):
    """加载混淆矩阵CSV文件
    CSV格式：第一列是预测类别，第一行（除了第一个单元格）是实际类别
    返回格式：行=预测类别（Predicted Label），列=实际类别（True Label）
    """
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        true_classes = header[1:]  # 第一行（跳过'Predicted'）是实际类别（列）
        
        matrix = []
        predicted_classes = []
        for row in reader:
            predicted_classes.append(row[0])  # 第一列是预测类别（行）
            matrix.append([float(x) for x in row[1:]])  # 数值部分
        
        # CSV格式：matrix[i,j] 表示预测为predicted_classes[i]，实际为true_classes[j]
        # 保持这个格式：行=预测类别，列=实际类别
        matrix = np.array(matrix)
    
    return matrix, true_classes, predicted_classes


def classify_classes(class_names, old_classes, new_classes, background_name):
    """将类别分类到 old, new, background 三个组"""
    class_to_group = {}
    
    # 确保 old_classes 和 new_classes 是列表
    old_classes_set = set(old_classes) if isinstance(old_classes, list) else {old_classes}
    new_classes_set = set(new_classes) if isinstance(new_classes, list) else {new_classes}
    
    for cls in class_names:
        if cls == background_name:
            class_to_group[cls] = 'background'
        elif cls in old_classes_set:
            class_to_group[cls] = 'old_classes'
        elif cls in new_classes_set:
            class_to_group[cls] = 'new_classes'
        else:
            # 如果类别不在任何组中，默认归为 old_classes
            class_to_group[cls] = 'old_classes'
    
    return class_to_group


def aggregate_matrix(matrix, true_classes, predicted_classes, class_to_group):
    """聚合混淆矩阵到三个组
    矩阵格式：行=预测类别（Predicted Label），列=实际类别（True Label）
    输出格式：行=预测类别组，列=实际类别组
    """
    groups = ['old_classes', 'new_classes', 'background']
    
    # 聚合矩阵
    aggregated = np.zeros((len(groups), len(groups)), dtype=float)
    
    # 聚合预测类别（行）
    for i, pred_cls in enumerate(predicted_classes):
        pred_group = class_to_group[pred_cls]
        pred_group_idx = groups.index(pred_group)
        
        # 聚合实际类别（列）
        for j, true_cls in enumerate(true_classes):
            true_group = class_to_group[true_cls]
            true_group_idx = groups.index(true_group)
            
            aggregated[pred_group_idx, true_group_idx] += matrix[i, j]
    
    return aggregated, groups


def normalize_row(matrix):
    """按行归一化（横向归一化）"""
    row_sums = matrix.sum(axis=1, keepdims=True)
    # 避免除零
    row_sums[row_sums == 0] = 1
    return matrix / row_sums


def normalize_col(matrix):
    """按列归一化（纵向归一化）"""
    col_sums = matrix.sum(axis=0, keepdims=True)
    # 避免除零
    col_sums[col_sums == 0] = 1
    return matrix / col_sums


def plot_confusion_matrix(matrix, group_names, title, save_path):
    """绘制混淆矩阵
    矩阵格式：行=预测类别，列=实际类别
    """
    plt.figure(figsize=(10, 8))
    
    # 使用seaborn绘制热力图
    sns.heatmap(matrix, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=group_names, yticklabels=group_names,
                cbar_kws={'label': 'Count' if '归一化' not in title else 'Proportion'})
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('Predicted Label', fontsize=12)
    plt.xlabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='聚合混淆矩阵并进行可视化')
    parser.add_argument('--confusion_matrix_path', type=str, required=True,
                        help='混淆矩阵CSV文件路径')
    parser.add_argument('--old_classes', nargs='+', required=True,
                        help='旧类别列表')
    parser.add_argument('--new_classes', nargs='+', required=True,
                        help='新类别列表')
    parser.add_argument('--background_name', type=str, default='background',
                        help='背景类别名称')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='保存输出图片的目录')
    
    args = parser.parse_args()
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 加载混淆矩阵
    print(f"Loading confusion matrix from {args.confusion_matrix_path}")
    matrix, true_classes, predicted_classes = load_confusion_matrix(args.confusion_matrix_path)
    
    # 分类类别（对实际类别和预测类别都需要分类）
    # 合并所有类别以确保都能正确分类
    all_classes = list(set(true_classes + predicted_classes))
    class_to_group = classify_classes(all_classes, args.old_classes, 
                                      args.new_classes, args.background_name)
    
    # 聚合矩阵
    print("Aggregating confusion matrix...")
    aggregated, group_names = aggregate_matrix(matrix, true_classes, predicted_classes, class_to_group)
    
    # 生成三个可视化
    # 1. 原始聚合混淆矩阵
    save_path1 = os.path.join(args.save_dir, 'aggregated_confusion_matrix.png')
    plot_confusion_matrix(aggregated, group_names, 
                         'Aggregated Confusion Matrix', save_path1)
    
    # 2. 横向归一化矩阵（按行归一化，行是预测类别）
    row_norm_matrix = normalize_row(aggregated)
    save_path2 = os.path.join(args.save_dir, 'row_normalized_confusion_matrix.png')
    plot_confusion_matrix(row_norm_matrix, group_names,
                         'Row Normalized Confusion Matrix (Normalized by Predicted Label)', save_path2)
    
    # 3. 纵向归一化矩阵（按列归一化，列是实际类别）
    col_norm_matrix = normalize_col(aggregated)
    save_path3 = os.path.join(args.save_dir, 'col_normalized_confusion_matrix.png')
    plot_confusion_matrix(col_norm_matrix, group_names,
                         'Column Normalized Confusion Matrix (Normalized by True Label)', save_path3)
    
    # 打印聚合后的矩阵值（用于验证）
    # 格式：行=预测类别，列=实际类别
    print("\nAggregated Confusion Matrix (Row=Predicted, Col=True):")
    print(f"{'Predicted/True':<15}", end='')
    for name in group_names:
        print(f"{name:<15}", end='')
    print()
    for i, name in enumerate(group_names):
        print(f"{name:<15}", end='')
        for j in range(len(group_names)):
            print(f"{aggregated[i, j]:<15.2f}", end='')
        print()
    
    print(f"\nAll visualizations saved to {args.save_dir}")


if __name__ == "__main__":
    main()
