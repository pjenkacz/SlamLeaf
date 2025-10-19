
import os, shutil, random
from pathlib import Path
import cv2
import albumentations as A


ROOT = Path(r"C:\Users\Majkel\Desktop\STUDIA\praca\PlantDoc\PlantDoc-4")     
SRC  = ROOT / "train" / "images"
DST  = ROOT / "train" / "images_aug"
LBL  = ROOT / "train" / "labels"
DST.mkdir(parents=True, exist_ok=True)

FRACTION = 0.4
random.seed(0)

# augmentacja lekki blur, lekkie zmiany jasności, kontrastu, saturacji
aug = A.Compose([
    A.GaussianBlur(blur_limit=(3, 3), p=0.25),
    A.RandomBrightnessContrast(brightness_limit=0.2,
                               contrast_limit=0.2, p=0.35),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30,
                         val_shift_limit=20, p=0.35),
], p=1.0)

EXT = {".jpg", ".jpeg", ".png"}

imgs = [p for p in SRC.iterdir() if p.suffix.lower() in EXT]
random.shuffle(imgs)
take = int(len(imgs) * FRACTION)
chosen = imgs[:take]

print(f"Wybrano do augmentacji: {take} / {len(imgs)}")

for img_path in chosen:
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    aug_img = aug(image=img)["image"]
    out_name = img_path.stem + "_aug" + img_path.suffix
    out_path = DST / out_name
    cv2.imwrite(str(out_path), aug_img)

    lbl_src = LBL / (img_path.stem + ".txt")
    if lbl_src.exists():
        lbl_dst = LBL / (Path(out_name).stem + ".txt")
        shutil.copy(lbl_src, lbl_dst)

print("Nowe obrazy w:", DST)
