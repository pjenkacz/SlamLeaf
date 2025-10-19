from ultralytics import YOLO
from pathlib import Path
import torch
import time
import csv
import numpy as np
import statistics

MODEL_PATH = Path(r"C:\Users\Majkel\Desktop\STUDIA\praca\PlantDoc\runs\detect\plantdoc_yolo11s\weights\best.pt")
#DATA = Path(r"C:\Users\Majkel\Desktop\STUDIA\praca\PlantDoc\PlantDoc-4\data.yaml")
#TEST_DIR   = Path(r"C:\Users\Majkel\Desktop\STUDIA\praca\PlantDoc\PlantDoc-4\test\images")
TEST_DIR   = Path(r"C:\Users\Majkel\Desktop\STUDIA\praca\images\pomidory2_jpg")
SAVE_DIR   = Path(r"C:\Users\Majkel\Desktop\STUDIA\praca\PlantDoc\runs\predict\myphotos_yolo11s")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Uruchamianie predykcji na: {device.upper()}")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(MODEL_PATH))

    t0 = time.time()
    times = []
    results = model.predict(
        source=str(TEST_DIR),
        #data=str(DATA),
        conf=0.4,
        save=True,
        project=str(SAVE_DIR.parent),
        name=SAVE_DIR.name,
        exist_ok=True,
        device=0 if device == "cuda" else "cpu",
        imgsz=640,
        workers=0,
        save_txt = True,
    )

    total_time = time.time() - t0
    # save prediction times to CSV
    csv_path = SAVE_DIR / "timings.csv"
    rows = []
    for r in results:
        path = getattr(r, "path", None) or getattr(r, "orig_img", None)
        path_str = str(path) if path is not None else ""
        spd = getattr(r, "speed", {}) or {}
        pre = float(spd.get("preprocess", 0.0))
        inf = float(spd.get("inference", 0.0))
        post = float(spd.get("postprocess", 0.0))
        total_ms = pre + inf + post
        n_det = 0
        classes = []
        try:
            if r.boxes is not None and r.boxes.cls is not None:
                n_det = int(r.boxes.cls.numel())
                # map id -> class name
                if hasattr(r, "names") and r.names:
                    classes = [r.names[int(i)] for i in r.boxes.cls.tolist()]
                else:
                    classes = [str(int(i)) for i in r.boxes.cls.tolist()]
        except Exception:
            pass

        rows.append({
            "image_path": path_str,
            "preprocess_ms": pre,
            "inference_ms": inf,
            "postprocess_ms": post,
            "total_ms": total_ms,
            "detections": n_det,
            "classes": "|".join(classes)
        })
    inf_list = [row["inference_ms"] for row in rows if row["inference_ms"] > 0]
    total_list = [row["total_ms"] for row in rows if row["total_ms"] > 0]
    avg_inf = statistics.mean(inf_list) if inf_list else 0.0
    med_inf = statistics.median(inf_list) if inf_list else 0.0
    avg_total = statistics.mean(total_list) if total_list else 0.0
    fps = 1000.0 / avg_inf if avg_inf > 0 else 0.0

    summary_row = {
        "image_path": "__SUMMARY__",
        "preprocess_ms": "",
        "inference_ms": round(avg_inf, 3),
        "postprocess_ms": "",
        "total_ms": round(avg_total, 3),
        "detections": len(rows),
        "classes": f"FPS≈{fps:.2f}, total_runtime_s={total_time:.3f}"
    }
    rows.append(summary_row)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image_path", "preprocess_ms", "inference_ms", "postprocess_ms", "total_ms", "detections",
                        "classes"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n Wyniki zapisane w: {SAVE_DIR}")
    print(f" Liczba przetworzonych obrazów: {len(results)}")
    print(f"Całkowity czas : {total_time:.3f} s")
    print(f"Średni czas inferencji (ms/obraz): {avg_inf:.2f}  →  FPS ≈ {fps:.2f}")
    print(f" CSV: {csv_path}")

if __name__ == "__main__":
    main()
