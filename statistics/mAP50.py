import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

base = Path(r"C:\Users\Majkel\Desktop\STUDIA\praca\PlantDoc\runs\detect")

models = {
    "YOLO11n": base / "plantdoc_yolo11n" / "results.csv",
    "YOLO11s": base / "plantdoc_yolo11s" / "results.csv",
    "YOLO11n + color80": base / "plantdoc_yolo11n_80augcolor" / "results.csv",
    "YOLO11n + blur2": base / "plantdoc_yolo11n_phase2_color_blur" / "results.csv",
    "YOLO11n + blur3": base / "plantdoc_yolo11n_phase3_color_blur" / "results.csv",
}

dfs = {name: pd.read_csv(path) for name, path in models.items()}

plt.figure(figsize=(8,5))
for name, df in dfs.items():
    plt.plot(df["epoch"], df["metrics/mAP50-95(B)"], label=name)
plt.xlabel("Epoka")
plt.ylabel("mAP50-95")
plt.title("Porównanie mAP dla modeli YOLO11 (PlantDoc)")
plt.legend()
plt.grid()
plt.show()
