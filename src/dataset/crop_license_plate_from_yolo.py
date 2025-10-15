import os
import cv2
from glob import glob
import random

def crop_from_yolo_labels(img_dir, label_dir, output_dir, num_images):
    
    os.makedirs(output_dir, exist_ok=True)
    img_paths = glob(os.path.join(img_dir, "*.*"))
    img_paths = [p for p in img_paths if p.lower().endswith((".jpg", ".jpeg", ".png"))]
    if num_images > len(img_paths):
        num_images = len(img_paths)
    img_paths = random.sample(img_paths, num_images)

    counter = 1
    for img_path in img_paths:
        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        base_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(label_dir, base_name + ".txt")
        if not os.path.exists(label_path):
            print(f"⚠️ Không có label cho {img_path}")
            continue

        with open(label_path, "r") as f:
            lines = f.readlines()

        if len(lines) == 0:
            print(f"⚠️ Label rỗng: {label_path}")
            continue

        for line in lines:
            cls, cx, cy, bw, bh = map(float, line.strip().split())

            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)


            crop = img[y1:y2, x1:x2]
            ext = os.path.splitext(img_path)[-1].lower()
            save_path = os.path.join(output_dir, f"{counter}{ext}")
            cv2.imwrite(save_path, crop)

            print(f"✅ Saved {save_path}")
            counter += 1


img_dir = r"D:\Model\LicensePlate\LicensePlateDataset\images\train"
label_dir = r"D:\Model\LicensePlate\LicensePlateDataset\labels\train"
output_dir = r"D:\Model\LicensePlate\LicensePlateDataset\ImageCrop"
num_images = 1500

crop_from_yolo_labels(img_dir, label_dir, output_dir, num_images)

