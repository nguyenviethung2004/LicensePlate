import matplotlib.pyplot as plt
from ultralytics import YOLO
import cv2, os


model = YOLO(r"D:\Model\LicensePlate\runs\train2\yolov8n_licenseplate\weights\best.pt")

metrics = model.val(
    data=r"D:\Model\LicensePlate\LicensePlateDataset\dataset.yaml",
    split="test",
    imgsz=640,
    conf=0.25,
    plots=True
)


res = metrics.results_dict
print("Precision:   ", res["metrics/precision(B)"])
print("Recall:      ", res["metrics/recall(B)"])
print("mAP50:       ", res["metrics/mAP50(B)"])
print("mAP50-95:    ", res["metrics/mAP50-95(B)"])


names = ["Precision", "Recall", "mAP50", "mAP50-95"]
values = [
    res["metrics/precision(B)"],
    res["metrics/recall(B)"],
    res["metrics/mAP50(B)"],
    res["metrics/mAP50-95(B)"]
]

plt.figure(figsize=(7, 5))
plt.bar(names, values, color=["skyblue", "orange", "green", "red"])
plt.ylabel("Score")
plt.ylim(0, 1.0)
plt.title("YOLOv8 Test Metrics")
plt.show()

results_dir = metrics.save_dir  
print("📊 Biểu đồ chi tiết lưu tại:", results_dir)

for img_name, title in [
    ("PR_curve.png", "Precision-Recall Curve"),
    ("F1_curve.png", "F1 Curve"),
    ("confusion_matrix.png", "Confusion Matrix")
]:
    path = os.path.join(results_dir, img_name)
    if os.path.exists(path):
        img = cv2.imread(path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(7, 6))
        plt.imshow(img_rgb)
        plt.axis("off")
        plt.title(title)
        plt.show()