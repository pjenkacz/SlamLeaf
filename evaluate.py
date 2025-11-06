from ultralytics import YOLO

MODEL = r"runs/detect/plantdoc_yolo_11m_512/weights/best.pt"
DATA  = r"PlantDoc-4/data.yaml"

model = YOLO(MODEL)
def main():
    metrics = model.val(
        data=DATA,
        split='test',
        imgsz=640,
        batch=8,
        conf=0.001,
        iou=0.6,
        device=0,
        save_json=False,
        project=r"C:\Users\Majkel\Desktop\STUDIA\praca\PlantDoc\runs\evaluate",
        name="yolo11n_optimized_test",
    )

    print(metrics)  # m.in. mAP50, mAP50-95, Precision, Recall

if __name__ == "__main__":
    main()