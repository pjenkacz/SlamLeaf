# train_phase1.py
from ultralytics import YOLO
import torch
from pathlib import Path
import yaml

DATA   = Path(r"C:\Users\Majkel\Desktop\STUDIA\praca\PlantDoc\PlantDoc-4\data.yaml")
IMGSZ  = 640
BATCH  = 8
EPOCHS = 70
MODEL  = "yolo11n.pt"
HYP_YAML = r"C:\Users\Majkel\Desktop\STUDIA\praca\PlantDoc\plantDoc.yaml"

with open(HYP_YAML, "r", encoding="utf-8") as f:
    hyp = yaml.safe_load(f) or {}


def main():
    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(MODEL)
    model.train(
        data=str(DATA),
        imgsz=IMGSZ,
        batch=BATCH,
        epochs=EPOCHS,
        project="./runs/detect",
        name="plantdoc_80_color",
        workers=0,
        device=device,
        **hyp
    )
    print(r"BEST: C:\Users\Majkel\Desktop\STUDIA\praca\PlantDoc\runs\detect\plantdoc_70_color\weights\best.pt")

if __name__ == "__main__":
    main()
