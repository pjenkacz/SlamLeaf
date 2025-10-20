from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

base = Path(r"C:\Users\Majkel\Desktop\STUDIA\praca\PlantDoc\runs\detect")

MODELS = {
    "YOLO11n": base / "plantdoc_yolo11n" / "weights"  / "best.pt",
    "YOLO11n + color80": base / "plantdoc_yolo11n_80augcolor" / "weights"  / "best.pt",
    "YOLO11n + colorblur2": base / "plantdoc_yolo11n_phase2_color_blur" / "weights"  / "best.pt",
    "YOLO11s": base / "plantdoc_yolo11s" / "weights"  / "best.pt",
    #"YOLO11n + blur3": base / "plantdoc_yolo11n_phase3_color_blur" / "weights"  / "best.pt",
}

TEST_DIR = Path(r"C:\Users\Majkel\Desktop\STUDIA\praca\images\pomidory2_jpg")
OUT_DIR  = Path(r"C:\Users\Majkel\Desktop\STUDIA\praca\PlantDoc\runs\compare_panels")
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMGSZ  = 640
CONF   = 0.4
MAX_IMG = None     # ile obrazów, None - wszystkie
DEVICE = 0 if torch.cuda.is_available() else "cpu"

def ensure_same_height(imgs, target_h=None):
    if target_h is None:
        target_h = max(i.shape[0] for i in imgs)
    out = []
    for i in imgs:
        h, w = i.shape[:2]
        if h != target_h:
            new_w = int(w * (target_h / h))
            i = cv2.resize(i, (new_w, target_h), interpolation=cv2.INTER_AREA)
        out.append(i)
    return out

def add_title_bar(img, title, bar_h=88):
    h, w = img.shape[:2]
    bar = np.full((bar_h, w, 3), 255, dtype=np.uint8)
    base_scale = 0.8
    scale = base_scale * (w / 1280) ** 0.5 * (bar_h / 34) ** 0.75
    cv2.putText(bar, title, (10, bar_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), 2, cv2.LINE_AA)
    return np.vstack([bar, img])

def plot_and_stack(model_objs, model_names, src_path):
    orig_bgr = cv2.imread(str(src_path))
    if orig_bgr is None:
        raise RuntimeError(f"blad odczytu: {src_path}")
    orig_panel = add_title_bar(orig_bgr, "ORIGINAL")

    pred_panels = []
    for name, model in zip(model_names, model_objs):
        res = model.predict(
            source=str(src_path),
            conf=CONF,
            imgsz=IMGSZ,
            device=DEVICE,
            verbose=False
        )
        vis = res[0].plot()
        vis = add_title_bar(vis, name)
        pred_panels.append(vis)


    panels = [orig_panel] + pred_panels
    panels = ensure_same_height(panels)
    row = cv2.hconcat(panels)
    return row

def save_jpeg(path, img_bgr, quality=65):
    cv2.imwrite(str(path), img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])

def downscale_max_width(img_bgr, max_w=2200):
    h, w = img_bgr.shape[:2]
    if w <= max_w:
        return img_bgr
    new_h = int(h * (max_w / w))
    return cv2.resize(img_bgr, (max_w, new_h), interpolation=cv2.INTER_AREA)
def main():

    model_objs = []
    model_names = []
    for name, w in MODELS.items():
        model_objs.append(YOLO(w))
        model_names.append(name)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    images = [p for p in TEST_DIR.iterdir() if p.suffix.lower() in exts]
    images.sort()
    if MAX_IMG:
        images = images[:MAX_IMG]

    print(f"Porównuję {len(images)} obrazów dla {len(MODELS)} modeli...")
    for img_path in tqdm(images):
        panel = plot_and_stack(model_objs, model_names, img_path)
        panel = downscale_max_width(panel, max_w=2200)
        stem = img_path.stem
        out_jpg = OUT_DIR / f"{stem}_panel.jpg"
        save_jpeg(out_jpg, panel, quality=80)

    sample = OUT_DIR / f"{images[0].stem}_panel.jpg"
    panel_bgr = cv2.imread(str(sample))
    panel_rgb = cv2.cvtColor(panel_bgr, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(16,8))
    plt.imshow(panel_rgb)
    plt.axis('off')
    plt.title(sample.name)
    plt.show()

if __name__ == "__main__":
    main()
