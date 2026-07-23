from ultralytics import YOLO
import torch
import os

def main():
   
    device = '0' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Starting training on: {'GPU' if device == '0' else 'CPU'}")


    model = YOLO('yolov8n.pt') 


    yaml_path = os.path.abspath('traffic_dataset/data.yaml')
    
    print(f"📂 Looking for dataset at: {yaml_path}")

    results = model.train(
        data=yaml_path,                    
        epochs=50,                      
        imgsz=640,                         
        batch=16,                          
        device=device,                     
        project='Traffic_Models',         
        name='yolo_run_1',                 
        plots=True                         
    )

    print("✅ Training Complete! Look in 'Traffic_Models/yolo_run_1/weights/' for your new best.pt file.")

if __name__ == '__main__':
    main()
