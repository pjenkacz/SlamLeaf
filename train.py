from ultralytics import YOLO
import torch
from pathlib import Path


DATA = Path(r"C:\Users\Majkel\Desktop\STUDIA\praca\PlantDoc\PlantDoc-4\data.yaml")

IMGSZ  = 640
BATCH  = 8
EPOCHS = 100


MODEL_WEIGHTS = "yolo11n.pt"
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = YOLO(MODEL_WEIGHTS)
    results = model.train(
        data=str(DATA),
        imgsz=IMGSZ,
        batch=BATCH,
        epochs=EPOCHS,
        project="./runs/detect",
        name="plantdoc_yolo11_640",
        workers=0,
        device=0 if device == "cuda" else "cpu",
    )

    print("\nTrening zakończony.")
    print("Najlepsze wagi:", "./runs/detect/plantdoc_yolo11_640/weights/best.pt")

if __name__ == "__main__":
    main()