import os
import random
import cv2
import matplotlib.pyplot as plt
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances

# === ŚCIEŻKI ===
ROOT = "RCNN"  # katalog główny z images/ i annotations_test.json
IMG_DIR = os.path.join(ROOT, "images", "test")
ANN_FILE = os.path.join(ROOT, "annotations_test.json")

register_coco_instances("my_test", {}, ANN_FILE, IMG_DIR)

dataset_dicts = DatasetCatalog.get("my_test")
metadata = MetadataCatalog.get("my_test")

print(f"Załadowano {len(dataset_dicts)} obrazów z pliku: {ANN_FILE}")

for d in random.sample(dataset_dicts, 100):  # 5 losowych
    img_path = d["file_name"]
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.8)
    vis = visualizer.draw_dataset_dict(d)
    plt.figure(figsize=(10, 10))
    plt.imshow(vis.get_image()[:, :, ::-1])
    plt.axis("off")
    plt.title(os.path.basename(img_path))
    plt.show()
