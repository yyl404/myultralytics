#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
import yaml


def load_names(dataset_yaml: str):
    cfg = yaml.safe_load(open(dataset_yaml, 'r', encoding='utf-8'))
    names = cfg['names']
    if isinstance(names, dict):
        return [names[i] if i in names else names[str(i)] for i in range(len(names))]
    return list(names)


def parse_eval_csv(csv_path: str):
    rows = []
    all_row = None
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cls = str(row.get('Class', '')).strip()
            if cls.lower() == 'all':
                all_row = row
            else:
                rows.append(row)
    return rows, all_row


def find_row_for_class(rows, class_name: str):
    for r in rows:
        if str(r.get('Class', '')).strip() == class_name:
            return r
    return None


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def fmt(x):
    return '' if x is None else f'{x:.2f}'


def method_from_name(name: str):
    lname = name.lower()
    if 'abr' in lname:
        return 'ABR'
    if 'osr' in lname:
        return 'OSR'
    return 'Unknown'


def model_from_name(name: str):
    lname = name.lower()
    if 'yolov8x' in lname:
        return 'YOLOv8x'
    if 'yolov8l' in lname:
        return 'YOLOv8l'
    return 'Unknown'


def init_from_name(name: str):
    lname = name.lower()
    if 'fromcls' in lname:
        return 'cls'
    if 'fromscratch' in lname:
        return 'scratch'
    return 'unknown'


parser = argparse.ArgumentParser()
parser.add_argument('--input_csv', required=True)
parser.add_argument('--output_csv', required=True)
parser.add_argument('--output_md', required=True)
args = parser.parse_args()

records = []
with open(args.input_csv, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for item in reader:
        result_csv = item['result_csv']
        dataset_yaml = item['converted_dir'] + '/dataset.yaml'
        old_count = int(item['old_count'])
        split_tag = item['split_tag']
        name = item['name']

        if not Path(result_csv).exists():
            continue
        if not Path(dataset_yaml).exists():
            continue

        names = load_names(dataset_yaml)
        rows, all_row = parse_eval_csv(result_csv)

        old_names = names[:old_count]
        new_names = names[old_count:]

        old_scores = [safe_float(find_row_for_class(rows, n)['mAP50']) for n in old_names if find_row_for_class(rows, n)]
        new_scores = [safe_float(find_row_for_class(rows, n)['mAP50']) for n in new_names if find_row_for_class(rows, n)]
        all_scores = [safe_float(r['mAP50']) for r in rows]

        old_map50 = mean(old_scores)
        new_map50 = mean(new_scores)
        all_map50 = safe_float(all_row['mAP50']) if all_row and safe_float(all_row['mAP50']) is not None else mean(all_scores)

        records.append({
            'Experiment': name,
            'Method': method_from_name(name),
            'Split': split_tag.replace('_', '+') if split_tag != '10_10' else '10+10',
            'Model': model_from_name(name),
            'Init': init_from_name(name),
            'Old mAP50': old_map50,
            'New mAP50': new_map50,
            'All mAP50': all_map50,
            'Eval CSV': result_csv,
        })

# Normalize split names exactly
for r in records:
    if r['Split'] == '15+5':
        pass
    elif r['Split'] == '19+1':
        pass
    elif r['Split'] == '10+10':
        pass
    else:
        # fallback for tags like 15+5 already ok
        r['Split'] = r['Split'].replace('_', '+')

split_order = {'19+1': 0, '15+5': 1, '10+10': 2}
method_order = {'ABR': 0, 'OSR': 1}
records.sort(key=lambda x: (split_order.get(x['Split'], 99), method_order.get(x['Method'], 99), x['Experiment']))

Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
with open(args.output_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['Experiment', 'Method', 'Split', 'Model', 'Init', 'Old mAP50', 'New mAP50', 'All mAP50', 'Eval CSV'])
    writer.writeheader()
    for r in records:
        writer.writerow({
            'Experiment': r['Experiment'],
            'Method': r['Method'],
            'Split': r['Split'],
            'Model': r['Model'],
            'Init': r['Init'],
            'Old mAP50': fmt(r['Old mAP50']),
            'New mAP50': fmt(r['New mAP50']),
            'All mAP50': fmt(r['All mAP50']),
            'Eval CSV': r['Eval CSV'],
        })

# Markdown table
headers = ['Experiment', 'Method', 'Split', 'Model', 'Init', 'Old mAP50', 'New mAP50', 'All mAP50']
lines = []
lines.append('| ' + ' | '.join(headers) + ' |')
lines.append('| ' + ' | '.join(['---'] * len(headers)) + ' |')
for r in records:
    lines.append('| ' + ' | '.join([
        r['Experiment'], r['Method'], r['Split'], r['Model'], r['Init'],
        fmt(r['Old mAP50']), fmt(r['New mAP50']), fmt(r['All mAP50'])
    ]) + ' |')
Path(args.output_md).write_text('\n'.join(lines), encoding='utf-8')

print('Saved CSV:', args.output_csv)
print('Saved MD :', args.output_md)
