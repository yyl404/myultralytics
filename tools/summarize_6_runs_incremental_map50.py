import argparse
import os
import pandas as pd

RUNS = [
    "yolov8x_voc_15_5_fromcls_abr_pseudo_label",
    "yolov8x_voc_15_5_fromcls_osr_pseudo_label",
    "yolov8x_voc_19_1_fromcls_abr_pseudo_label",
    "yolov8x_voc_19_1_fromcls_osr_pseudo_label",
    "yolov8x_voc_10_10_fromcls_abr_pseudo_label",
    "yolov8x_voc_10_10_fromcls_osr_pseudo_label",
]

VOC_CLASS_NAMES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

SPLIT_INFO = {
    "15_5": {
        "old": VOC_CLASS_NAMES[:15],
        "new": VOC_CLASS_NAMES[15:],
    },
    "19_1": {
        "old": VOC_CLASS_NAMES[:19],
        "new": VOC_CLASS_NAMES[19:],
    },
    "10_10": {
        "old": VOC_CLASS_NAMES[:10],
        "new": VOC_CLASS_NAMES[10:],
    },
}


def infer_split(run_name: str):
    if "voc_15_5" in run_name:
        return "15_5"
    if "voc_19_1" in run_name:
        return "19_1"
    if "voc_10_10" in run_name:
        return "10_10"
    raise ValueError(f"Cannot infer split from run name: {run_name}")


def infer_method(run_name: str):
    if "_abr_" in run_name:
        return "ABR+PseudoLabel"
    if "_osr_" in run_name:
        return "OSR+PseudoLabel"
    return "Unknown"


def find_metric_columns(df: pd.DataFrame):
    cols = [c.lower() for c in df.columns]

    class_col = None
    ap50_col = None

    for c in df.columns:
        cl = c.lower()
        if class_col is None and cl in ["class", "cls", "name", "category"]:
            class_col = c
        if ap50_col is None and cl in ["ap50", "map50", "ap_50", "map_50"]:
            ap50_col = c

    if class_col is None:
        # fallback: find the first text-like column
        for c in df.columns:
            if df[c].dtype == object:
                class_col = c
                break

    if ap50_col is None:
        # fallback: prefer columns containing ap50/map50
        for c in df.columns:
            cl = c.lower()
            if "ap50" in cl or "map50" in cl:
                ap50_col = c
                break

    if class_col is None or ap50_col is None:
        raise RuntimeError(
            f"Cannot find class/ap50 columns. Columns are: {list(df.columns)}"
        )

    return class_col, ap50_col


def normalize_class_name(x):
    x = str(x).strip()
    x = x.replace("tvmonitor", "tvmonitor")
    return x


def summarize_one(csv_path: str, split_name: str):
    df = pd.read_csv(csv_path)
    class_col, ap50_col = find_metric_columns(df)

    df = df[[class_col, ap50_col]].copy()
    df[class_col] = df[class_col].apply(normalize_class_name)

    # filter only VOC classes
    df = df[df[class_col].isin(VOC_CLASS_NAMES)]

    old_classes = SPLIT_INFO[split_name]["old"]
    new_classes = SPLIT_INFO[split_name]["new"]

    old_df = df[df[class_col].isin(old_classes)]
    new_df = df[df[class_col].isin(new_classes)]
    all_df = df[df[class_col].isin(old_classes + new_classes)]

    old_map50 = old_df[ap50_col].mean() if len(old_df) > 0 else float("nan")
    new_map50 = new_df[ap50_col].mean() if len(new_df) > 0 else float("nan")
    all_map50 = all_df[ap50_col].mean() if len(all_df) > 0 else float("nan")

    return old_map50, new_map50, all_map50


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_root", type=str, default="runs")
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--output_md", type=str, required=True)
    args = parser.parse_args()

    rows = []

    for run_name in RUNS:
        split_name = infer_split(run_name)
        method = infer_method(run_name)
        eval_csv = os.path.join(
            args.runs_root,
            run_name,
            "evaluation_results",
            "final_cumulative_eval.csv"
        )

        if not os.path.exists(eval_csv):
            print(f"Warning: missing eval csv, skip: {eval_csv}")
            continue

        old_map50, new_map50, all_map50 = summarize_one(eval_csv, split_name)

        rows.append({
            "run_name": run_name,
            "split": split_name,
            "method": method,
            "old_mAP50": round(old_map50, 4) if pd.notna(old_map50) else None,
            "new_mAP50": round(new_map50, 4) if pd.notna(new_map50) else None,
            "all_mAP50": round(all_map50, 4) if pd.notna(all_map50) else None,
        })

    out_df = pd.DataFrame(rows)
    if len(out_df) == 0:
        raise RuntimeError("No valid evaluation results found.")

    split_order = {"19_1": 0, "15_5": 1, "10_10": 2}
    method_order = {"ABR+PseudoLabel": 0, "OSR+PseudoLabel": 1}

    out_df["split_order"] = out_df["split"].map(split_order)
    out_df["method_order"] = out_df["method"].map(method_order)
    out_df = out_df.sort_values(["split_order", "method_order"]).drop(columns=["split_order", "method_order"])

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)

    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write(out_df.to_markdown(index=False))

    print(f"Saved CSV: {args.output_csv}")
    print(f"Saved MD : {args.output_md}")


if __name__ == "__main__":
    main()