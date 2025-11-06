import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

ANNOT_FILE = "annotations_train.json"

with open(ANNOT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

bboxes = []
cats = []

for ann in data["annotations"]:
    x, y, w, h = ann["bbox"]
    if w > 0 and h > 0:
        bboxes.append((w, h))
        cats.append(ann["category_id"])

bboxes = np.array(bboxes)
widths = bboxes[:, 0]
heights = bboxes[:, 1]
ratios = widths / heights
areas = widths * heights

print(f"Liczba bboxów: {len(bboxes):,}")
print(f"Średnia szerokość: {np.mean(widths):.1f}px, mediana: {np.median(widths):.1f}px")
print(f"Średnia wysokość: {np.mean(heights):.1f}px, mediana: {np.median(heights):.1f}px")
print(f"Zakres szerokości: {np.min(widths):.1f}px – {np.max(widths):.1f}px")
print(f"Zakres wysokości: {np.min(heights):.1f}px – {np.max(heights):.1f}px")

for p in [10, 25, 50, 75, 90]:
    print(f"{p}th percentyl szerokości: {np.percentile(widths, p):.1f}px, wysokości: {np.percentile(heights, p):.1f}px")

ratio_bins = [0.5, 1.0, 2.0, 3.0, 4.0]
hist, edges = np.histogram(ratios, bins=ratio_bins)
print("\n📐 Proporcje bboxów:")
for i in range(len(hist)):
    print(f"  {edges[i]:.1f}–{edges[i+1]:.1f}: {hist[i]} bboxów")

plt.figure(figsize=(6,5))
plt.scatter(widths, heights, s=10, alpha=0.3)
plt.xlabel("Szerokość [px]")
plt.ylabel("Wysokość [px]")
plt.title("Rozkład rozmiarów bboxów (train)")
plt.grid(True)
plt.show()