#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
class_distribution.py
---------------------
Compute bbox (instance) counts per class for COCO annotations, per split and total.

Expected files (any subset is fine):
  dataset/annotations_train.json
  dataset/annotations_val.json
  dataset/annotations_test.json

Outputs:
  /mnt/data/reports/class_distribution_per_split.csv   # long format
  /mnt/data/reports/class_distribution_wide.csv        # wide format (columns per split + total)
  /mnt/data/reports/class_distribution_plot.png        # optional bar plot of TOTAL
Usage:
  python class_distribution.py --root dataset [--no-plot]
"""
import os
import json
import argparse
from collections import Counter, defaultdict
import pandas as pd
import matplotlib.pyplot as plt

def load_split(path):
    if not os.path.isfile(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def count_per_class(coco):
    if coco is None:
        return {}, {}
    categories = {c['id']: c['name'] for c in coco.get('categories', [])}
    counter = Counter([ann['category_id'] for ann in coco.get('annotations', [])])
    return categories, counter

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='dataset', help='Folder with annotations_*.json')
    ap.add_argument('--no-plot', action='store_true', help='Disable plot generation')
    args = ap.parse_args()

    root = args.root
    files = {
        'train': os.path.join(root, 'annotations_train.json'),
        'val':   os.path.join(root, 'annotations_val.json'),
        'test':  os.path.join(root, 'annotations_test.json'),
    }

    per_split_counts = {}
    categories_ref = {}
    for split, fp in files.items():
        coco = load_split(fp)
        cats, cnt = count_per_class(coco)
        per_split_counts[split] = cnt
        if cats:
            categories_ref.update(cats)

    records = []
    all_cat_ids = sorted(categories_ref.keys())
    for cat_id in all_cat_ids:
        name = categories_ref[cat_id]
        total = 0
        row = {'category_id': cat_id, 'class_name': name}
        for split in ['train', 'val', 'test']:
            val = per_split_counts.get(split, {}).get(cat_id, 0)
            row[f'{split}_count'] = val
            total += val
        row['total_count'] = total
        records.append(row)

    df_wide = pd.DataFrame(records).sort_values('total_count', ascending=False)

    rows_long = []
    for _, r in df_wide.iterrows():
        for split in ['train', 'val', 'test']:
            rows_long.append({
                'category_id': int(r['category_id']),
                'class_name': r['class_name'],
                'split': split,
                'bbox_count': int(r[f'{split}_count'])
            })
        rows_long.append({
            'category_id': int(r['category_id']),
            'class_name': r['class_name'],
            'split': 'total',
            'bbox_count': int(r['total_count'])
        })
    df_long = pd.DataFrame(rows_long)

    out_dir = os.path.join(root, "reports")
    os.makedirs(out_dir, exist_ok=True)
    path_long = os.path.join(out_dir, 'class_distribution_per_split.csv')
    path_wide = os.path.join(out_dir, 'class_distribution_wide.csv')
    df_long.to_csv(path_long, index=False)
    df_wide.to_csv(path_wide, index=False)

    if not args.no_plot and not df_wide.empty:
        plt.figure(figsize=(12, 7))
        plt.barh(df_wide['class_name'], df_wide['total_count'])
        plt.xlabel('BBox count (total)')
        plt.ylabel('Class')
        plt.title('Class distribution (total bbox count per class)')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plot_path = os.path.join(out_dir, 'class_distribution_plot.png')
        plt.savefig(plot_path)
        print(f'[OK] Saved plot: {plot_path}')
    else:
        plot_path = None

    print('[OK] Saved:')
    print(' -', path_long)
    print(' -', path_wide)

if __name__ == '__main__':
    main()
