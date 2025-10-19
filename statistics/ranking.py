import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

base = Path(r"C:\Users\Majkel\Desktop\STUDIA\praca\PlantDoc\runs\detect")

models = {
    "YOLO11n": base / "plantdoc_yolo11n" / "results.csv",
    "YOLO11n + color80": base / "plantdoc_yolo11n_80augcolor" / "results.csv",
    "YOLO11s": base / "plantdoc_yolo11s" / "results.csv",
    "YOLO11n + blur2": base / "plantdoc_yolo11n_phase2_color_blur" / "results.csv",
    "YOLO11n + blur3": base / "plantdoc_yolo11n_phase3_color_blur" / "results.csv",
}

dfs = {name: pd.read_csv(path) for name, path in models.items()}

summary = []
for name, df in dfs.items():
    last = df.iloc[-1]
    summary.append({
        "Model": name,
        "Precision": last["metrics/precision(B)"],
        "Recall": last["metrics/recall(B)"],
        "mAP50": last["metrics/mAP50(B)"],
        "mAP50-95": last["metrics/mAP50-95(B)"],
    })

summary_df = (
    pd.DataFrame(summary)
      .sort_values("mAP50-95", ascending=False)
      .reset_index(drop=True)
)

# 👉 (opcjonalnie) wykres
ax = summary_df.plot(x="Model", y=["mAP50","mAP50-95","Precision","Recall"], kind="bar")
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.show()
