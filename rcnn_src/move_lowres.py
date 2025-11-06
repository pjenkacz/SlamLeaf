#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
move_lowres.py
--------------
Scan RCNN/images/{train,val,test} and move images whose min(side) < THRESH_PX
to RCNN/images_low_resolution/{split}. Move corresponding YOLO .txt labels from
RCNN/labels/{split} to RCNN/labels_low_resolution/{split}.

Usage:
  python move_lowres.py --root RCNN --threshold 480 [--dry-run]

Notes:
- Keeps folder structure by split.
- If a matching label doesn't exist, it logs a warning and still moves the image.
- Generates a JSON report with moved files.
"""
import os
import glob
import argparse
import shutil
from PIL import Image
import json

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
SPLITS = ["train", "val", "test"]

def list_images(img_dir):
    files = []
    for ext in IMG_EXTS:
        files.extend(glob.glob(os.path.join(img_dir, f"*{ext}")))
    return sorted(files)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root folder containing images/ and labels/ subfolders (e.g., RCNN)")
    ap.add_argument("--threshold", type=int, default=480, help="Move if min(image width,height) < threshold (px)")
    ap.add_argument("--dry-run", action="store_true", help="Only print what would be moved")
    args = ap.parse_args()

    root = args.root
    thr = args.threshold
    dry = args.dry_run

    src_images_root = os.path.join(root, "images")
    src_labels_root = os.path.join(root, "labels")
    dst_images_root = os.path.join(root, "images_low_resolution")
    dst_labels_root = os.path.join(root, "labels_low_resolution")

    moved = {split: [] for split in SPLITS}
    total_checked = {split: 0 for split in SPLITS}

    for split in SPLITS:
        img_dir = os.path.join(src_images_root, split)
        lbl_dir = os.path.join(src_labels_root, split)
        if not os.path.isdir(img_dir):
            print(f"[SKIP] Missing images dir: {img_dir}")
            continue

        ensure_dir(os.path.join(dst_images_root, split))
        ensure_dir(os.path.join(dst_labels_root, split))

        for img_path in list_images(img_dir):
            total_checked[split] += 1
            try:
                with Image.open(img_path) as im:
                    w, h = im.size
            except Exception as e:
                print(f"[WARN] Cannot open image: {img_path} ({e})")
                continue

            if w < thr and h < thr:
                base = os.path.splitext(os.path.basename(img_path))[0]
                label_src = os.path.join(lbl_dir, base + ".txt")
                img_dst = os.path.join(dst_images_root, split, os.path.basename(img_path))
                label_dst = os.path.join(dst_labels_root, split, base + ".txt")

                action = "MOVE" if not dry else "WOULD MOVE"
                print(f"[{action}] {img_path} ({w}x{h}) -> {img_dst}")
                if os.path.isfile(label_src):
                    print(f"[{action}] {label_src} -> {label_dst}")
                else:
                    print(f"[WARN] Label not found for image: {label_src}")

                if not dry:
                    shutil.move(img_path, img_dst)
                    if os.path.isfile(label_src):
                        shutil.move(label_src, label_dst)

                moved[split].append({
                    "image": img_path,
                    "image_dst": img_dst,
                    "width": w, "height": h,
                    "label": label_src if os.path.isfile(label_src) else None,
                    "label_dst": label_dst if os.path.isfile(label_src) else None
                })

    report_path = os.path.join(root, "low_resolution_moved_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "threshold_px": thr,
            "totals_checked": total_checked,
            "moved": moved
        }, f, indent=2, ensure_ascii=False)

    print("\n[OK] Done.")
    print(f" - Report: {report_path}")
    for split in SPLITS:
        print(f"   {split}: moved {len(moved[split])} / checked {total_checked[split]}")

if __name__ == "__main__":
    main()
