import cv2
import os
import albumentations as A
from tqdm import tqdm
import numpy as np


INPUT_IMAGE_DIR = "/content/License-Plate---OCR-3/train/images"
INPUT_LABEL_DIR = "/content/License-Plate---OCR-3/train/labels"

OUTPUT_IMAGE_DIR = "/content/train/images"
OUTPUT_LABEL_DIR = "/content/train/labels"

NUM_AUGMENTATIONS_PER_IMAGE = 3

transform = A.Compose([

    A.Rotate(limit=10, p=0.3, border_mode=cv2.BORDER_CONSTANT, value=0),
    A.Perspective(scale=(0.05, 0.1), p=0.3),

    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.4),

    A.OneOf([
        A.MotionBlur(blur_limit=15, p=1.0),
        A.GlassBlur(sigma=0.8, max_delta=4, p=1.0),
        A.GaussianBlur(blur_limit=(3, 7), p=1.0),
    ], p=0.5),

    A.OneOf([
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
        A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
    ], p=0.4),
    A.ImageCompression(quality_lower=40, quality_upper=90, p=0.5),

], bbox_params=A.BboxParams(
    format='yolo', 
    label_fields=['class_labels'],
    min_area=10,
    min_visibility=0.2 
))



def load_yolo_labels(label_path):
    bboxes = []
    class_labels = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            class_id = int(parts[0])
            bbox = [float(p) for p in parts[1:]]
            bboxes.append(bbox)
            class_labels.append(class_id)
    return bboxes, class_labels

def save_yolo_labels(output_path, bboxes, class_labels):
    with open(output_path, 'w') as f:
        for bbox, class_id in zip(bboxes, class_labels):
           
            line = f"{class_id} {' '.join(f'{coord:.6f}' for coord in bbox)}\n"
            f.write(line)


def augment_dataset():

    os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

    image_filenames = [f for f in os.listdir(INPUT_IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    print(f"Bắt đầu tăng cường dữ liệu cho {len(image_filenames)} ảnh...")

    for filename in tqdm(image_filenames, desc="Đang xử lý"):
        try:
            base_name, _ = os.path.splitext(filename)
            image_path = os.path.join(INPUT_IMAGE_DIR, filename)
            label_path = os.path.join(INPUT_LABEL_DIR, f"{base_name}.txt")

           
            if not os.path.exists(label_path):
                continue

            
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            bboxes, class_labels = load_yolo_labels(label_path)

            for i in range(NUM_AUGMENTATIONS_PER_IMAGE):
                augmented = transform(
                    image=image_rgb,
                    bboxes=bboxes,
                    class_labels=class_labels
                )

               
                augmented_image = augmented['image']
                augmented_bboxes = augmented['bboxes']
                augmented_labels = augmented['class_labels']

        
                if not augmented_bboxes:
                    continue


                new_filename_base = f"{base_name}_aug_{i+1}"
                output_image_path = os.path.join(OUTPUT_IMAGE_DIR, f"{new_filename_base}.jpg")
                output_label_path = os.path.join(OUTPUT_LABEL_DIR, f"{new_filename_base}.txt")

                cv2.imwrite(output_image_path, cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR))
                save_yolo_labels(output_label_path, augmented_bboxes, augmented_labels)

        except Exception as e:
            print(f"Lỗi khi xử lý file {filename}: {e}")

    print("\n🎉 Hoàn tất! Dữ liệu đã được tăng cường và lưu thành công.")


if __name__ == "__main__":
    augment_dataset()