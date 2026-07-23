from ultralytics import YOLO
import torch
import os

def main():
    # 1. Check for GPU
    device = '0' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Starting training on: {'GPU' if device == '0' else 'CPU'}")

    # 2. Load the YOLOv8 Nano model (Best for real-time edge devices)
    model = YOLO('yolov8n.pt') 

    # 3. Define the path to your data.yaml file
    # Ensure this matches exactly where your folder is!
    yaml_path = os.path.abspath('traffic_dataset/data.yaml')
    
    print(f"📂 Looking for dataset at: {yaml_path}")

    # 4. Start Training
    results = model.train(
        data=yaml_path,                    # Path to the Roboflow config file
        epochs=50,                         # 50 loops through your dataset
        imgsz=640,                         # Standard image size
        batch=16,                          # Number of images processed at once
        device=device,                     # Force GPU if available
        project='Traffic_Models',          # Master folder for saved weights
        name='yolo_run_1',                 # Subfolder for this specific run
        plots=True                         # Generate performance graphs
    )

    print("✅ Training Complete! Look in 'Traffic_Models/yolo_run_1/weights/' for your new best.pt file.")

if __name__ == '__main__':
    main()