# train_phase2_blur.py
from ultralytics import YOLO
import torch
from pathlib import Path

DATA   = Path(r"C:\Users\Majkel\Desktop\STUDIA\praca\PlantDoc\PlantDoc-4\data.yaml")
WEIGHTS_PHASE1 = Path(r"C:\Users\Majkel\Desktop\STUDIA\praca\PlantDoc\runs\detect\plantdoc_phase2_blur\weights\best.pt")
IMGSZ  = 640
BATCH  = 8
EPOCHS = 20

def main():
    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(str(WEIGHTS_PHASE1))
    model.train(
        data=str(DATA),
        imgsz=IMGSZ,
        batch=BATCH,
        epochs=EPOCHS,
        project=r"C:\Users\Majkel\Desktop\STUDIA\praca\PlantDoc\runs\detect",
        name="plantdoc_phase3_blur",
        workers=0,
        device=device,
        # lekka augmentacja barwy/jasności, bardzo lekkie zmiany geometryczne
        hsv_h=0.008, hsv_s=0.25, hsv_v=0.15,
        fliplr=0.30, degrees=1.0, translate=0.01, scale=0.02, shear=0.0,
        mosaic=0.0, mixup=0.0, copy_paste=0.0,
        seed=0,
    )
    print(r"BEST: C:\Users\Majkel\Desktop\STUDIA\praca\PlantDoc\runs\detect\plantdoc_phase3_blur\weights\best.pt")

if __name__ == "__main__":
    main()
