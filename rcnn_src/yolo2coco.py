#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yolo2coco.py
------------
Convert YOLO-format labels (cx cy w h normalized) into COCO JSON for splits: train/val/test.

Expected layout:
  dataset/
    images/
      train/*.jpg|*.png|*.jpeg|*.bmp
      val/*.jpg|*.png|*.jpeg|*.bmp
      test/*.jpg|*.png|*.jpeg|*.bmp     (optional)
    labels/
      train/*.txt
      val/*.txt
      test/*.txt                        (optional)
  data.yaml                             (optional; with 'names' list)

Usage examples:
  # If you have data.yaml with class names in dataset/
  python yolo2coco.py --root dataset --from-yaml data.yaml

  # Or pass classes directly
  python yolo2coco.py --root dataset --classes apple tomato potato

Outputs:
  dataset/annotations_train.json
  dataset/annotations_val.json
  dataset/annotations_test.json (only if test split present)
"""

import os
import glob
import json
import argparse
from typing import List, Tuple, Dict
from PIL import Image

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp")

def read_classes_from_yaml(yaml_path: str) -> List[str]:
    # Minimal YAML reader to avoid pyyaml dependency; expects a line like: names: [a, b, c]
    # or a block:
    # names:
    #   - a
    #   - b
    #   - c
    cls = []
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f"YAML not found: {yaml_path}")
    with open(yaml_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    in_names_block = False
    for line in lines:
        s = line.strip()
        if s.startswith("names:"):
            if "[" in s and "]" in s:
                inside = s[s.find("[")+1:s.rfind("]")]
                cls = [x.strip().strip("'\"") for x in inside.split(",") if x.strip()]
                return cls
            else:
                in_names_block = True
                continue
        if in_names_block:
            if s.startswith("- "):
                cls.append(s[2:].strip().strip("'\""))
            else:
                if cls:
                    return cls
    if not cls:
        raise ValueError("Could not parse class names from YAML. Ensure it has a 'names' list.")
    return cls

def find_images(img_dir: str) -> List[str]:
    files = []
    for ext in IMG_EXTS:
        files.extend(glob.glob(os.path.join(img_dir, f"*{ext}")))
    return sorted(files)

def yolo_to_coco_bbox(cx: float, cy: float, w: float, h: float, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    x = (cx - w / 2.0) * img_w
    y = (cy - h / 2.0) * img_h
    bw = w * img_w
    bh = h * img_h
    return x, y, bw, bh

def convert_split(root: str, split: str, class_names: List[str]) -> Tuple[Dict, int, int]:
    img_dir = os.path.join(root, "images", split)
    lbl_dir = os.path.join(root, "labels", split)

    images, annotations, categories = [], [], []
    ann_id = 1
    img_count = 0
    ann_count = 0

    if not os.path.isdir(img_dir):
        print(f"[WARN] Images dir missing for split '{split}': {img_dir}")
        return {}, 0, 0

    # categories
    for cid, name in enumerate(class_names, start=1):
        categories.append({"id": cid, "name": name, "supercategory": "object"})

    img_paths = find_images(img_dir)
    for i, img_path in enumerate(img_paths, start=1):
        fname = os.path.basename(img_path)
        try:
            with Image.open(img_path) as im:
                w, h = im.size
        except Exception as e:
            print(f"[WARN] Cannot open image '{img_path}': {e}")
            continue

        images.append({"id": i, "file_name": fname, "width": w, "height": h})
        img_count += 1

        base = os.path.splitext(fname)[0]
        yolo_txt = os.path.join(lbl_dir, base + ".txt")
        if not os.path.exists(yolo_txt):
            continue

        with open(yolo_txt, "r", encoding="utf-8") as f:
            for ln, line in enumerate(f, start=1):
                parts = line.strip().split()
                if len(parts) != 5:
                    if len(parts) == 0:
                        continue
                    print(f"[WARN] Bad line in {yolo_txt}:{ln} -> '{line.strip()}'")
                    continue
                try:
                    cls = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:])
                except Exception as e:
                    print(f"[WARN] Parse error in {yolo_txt}:{ln}: {e}")
                    continue
                if cls < 0 or cls >= len(class_names):
                    print(f"[WARN] Class id {cls} out of range 0..{len(class_names)-1} in {yolo_txt}:{ln}")
                    continue

                x, y, bw_px, bh_px = yolo_to_coco_bbox(cx, cy, bw, bh, w, h)

                x = max(0.0, min(x, w - 1))
                y = max(0.0, min(y, h - 1))
                bw_px = max(0.0, min(bw_px, w - x))
                bh_px = max(0.0, min(bh_px, h - y))
                if bw_px <= 1e-6 or bh_px <= 1e-6:
                    continue

                annotations.append({
                    "id": ann_id,
                    "image_id": i,
                    "category_id": cls + 1,  # COCO category ids start at 1
                    "bbox": [x, y, bw_px, bh_px],
                    "area": bw_px * bh_px,
                    "iscrowd": 0,
                    "segmentation": []
                })
                ann_id += 1
                ann_count += 1

    coco = {"images": images, "annotations": annotations, "categories": categories}
    return coco, img_count, ann_count

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root folder containing images/ and labels/ subfolders")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"], help="Dataset splits to convert")
    ap.add_argument("--from-yaml", dest="yaml", default=None, help="Path to YOLO data.yaml (to read class names)")
    ap.add_argument("--classes", nargs="*", default=None, help="Class names in YOLO order (override YAML)")
    args = ap.parse_args()

    if args.classes:
        class_names = args.classes
    elif args.yaml:
        yaml_path = args.yaml
        if not os.path.isabs(yaml_path):
            yaml_path = os.path.join(args.root, yaml_path)
        class_names = read_classes_from_yaml(yaml_path)
    else:
        raise SystemExit("Provide classes via --classes or --from-yaml <data.yaml>")

    print(f"[INFO] Classes ({len(class_names)}): {class_names}")

    for split in args.splits:
        coco, n_img, n_ann = convert_split(args.root, split, class_names)
        if coco:
            out = os.path.join(args.root, f"annotations_{split}.json")
            with open(out, "w", encoding="utf-8") as f:
                json.dump(coco, f)
            print(f"[OK] Saved {out}  (images: {n_img}, annotations: {n_ann})")
        else:
            print(f"[SKIP] Split '{split}' not converted (missing images dir)")

if __name__ == "__main__":
    main()
