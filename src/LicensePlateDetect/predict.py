from ultralytics import YOLO
import cv2

def predict_and_crop(model, img_path, conf=0.25):

    results = model.predict(img_path, conf=conf, imgsz=640, verbose=False)
    boxes = results[0].boxes

    im = cv2.imread(img_path)
    h, w = im.shape[:2]

    if len(boxes) == 0:
        print("❌ Không phát hiện được đối tượng nào.")
        return None

    best_box = boxes[0]
    xyxy = best_box.xyxy[0].cpu().numpy().astype(int)
    x1, y1, x2, y2 = xyxy
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    crop_img = im[y1:y2, x1:x2]

    return crop_img

