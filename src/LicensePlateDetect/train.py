from ultralytics import YOLO
import torch

def train_model():
    model = YOLO('yolov8n.pt')
    print("Bắt đầu quá trình huấn luyện...")
    try:
        results = model.train(
            data=r"D:\Model\LicensePlate\LicensePlateDataset\dataset.yaml",
            epochs=50,             
            imgsz=640,               
            batch=16,               
            lr0=0.002,
            lrf=0.0001, 
            cos_lr=True,
            project="runs/train2",
            name="yolov8n_licenseplate",
            patience=5,             
            device=0,               
            exist_ok=True,           
            project='runs/detect'    
        )
        print("Huấn luyện hoàn tất!")
        print(f"Model tốt nhất đã được lưu tại: {results.save_dir}/weights/best.pt")

    except Exception as e:
        print(f"Đã xảy ra lỗi trong quá trình huấn luyện: {e}")

if __name__ == '__main__':
    print(f"CUDA (GPU) có sẵn: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Tên GPU: {torch.cuda.get_device_name(0)}")
    train_model()