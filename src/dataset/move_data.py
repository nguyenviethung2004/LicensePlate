import os
import random
import shutil


folder_A_images = "/content/License_Plate_Recognition-2/valid/images"  
folder_A_labels = "/content/License_Plate_Recognition-2/valid/labels"   


folder_B_images = "/content/License-Plate---OCR-3/valid/images/"
folder_B_labels = "/content/License-Plate---OCR-3/valid/labels/"

os.makedirs(folder_B_images, exist_ok=True)
os.makedirs(folder_B_labels, exist_ok=True)


all_images = [f for f in os.listdir(folder_A_images) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

for i, img_name in enumerate(all_images, start=1):
    src_img = os.path.join(folder_A_images, img_name)
    dst_img = os.path.join(folder_B_images, img_name)


    shutil.copy(src_img, dst_img)

    label_name = os.path.splitext(img_name)[0] + ".txt"
    src_label = os.path.join(folder_A_labels, label_name)
    dst_label = os.path.join(folder_B_labels, label_name)

    if os.path.exists(src_label):
        shutil.copy(src_label, dst_label)


    if i % 1000 == 0:
        print(f"✅ Đã sao chép {i} ảnh...")

print("🎯 Hoàn thành sao chép 20.000 ảnh và nhãn tương ứng!")
