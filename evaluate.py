from ultralytics import YOLO

MODEL = r"C:\Users\Majkel\Desktop\STUDIA\praca\PlantDoc\runs\detect\plantdoc_yolo11s\weights\best.pt"
DATA  = r"C:\Users\Majkel\Desktop\STUDIA\praca\PlantDoc\PlantDoc-4\data.yaml"

model = YOLO(MODEL)
def main():
    metrics = model.val(
        data=DATA,
        split='test',     # ⬅️ kluczowe: test zamiast domyślnego val
        imgsz=640,
        batch=8,
        conf=0.001,
        iou=0.6,
        device=0,
        save_json=False,
        project=r"C:\Users\Majkel\Desktop\STUDIA\praca\PlantDoc\runs\evaluate",
        name="yolo11s_test",
    )

    print(metrics)  # m.in. mAP50, mAP50-95, Precision, Recall

if __name__ == "__main__":
    main()