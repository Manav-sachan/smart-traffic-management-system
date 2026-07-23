import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import cv2
import numpy as np
import time

# --- CONFIGURATION ---
model_path = "traffic_model.pth"   # Your trained model
video_path = "traffic_video.mp4"   # Your video
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Class mapping (Must match your training)
class_dict = {0: 'background', 1: 'car', 2: 'bus', 3: 'bike', 4: 'ambulance'}

# Traffic Light Settings
min_green_time = 5   # Seconds for low traffic
max_green_time = 20  # Seconds for high traffic
emergency_time = 10  # Extra time for ambulance

# --- 1. LOAD THE TRAINED MODEL ---
def get_model(num_classes):
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

print(f"Loading system on {device}...")
model = get_model(len(class_dict))

# Load your trained weights
if torch.cuda.is_available():
    model.load_state_dict(torch.load(model_path))
else:
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

model.to(device)
model.eval()
print("✅ System Ready.")

# --- 2. TRAFFIC LOGIC UTILS ---
def decide_signal_time(vehicle_count, has_ambulance):
    """
    Decides how long the Green light should stay on.
    Logic: More Cars = More Time. Ambulance = Priority.
    """
    if has_ambulance:
        return "EMERGENCY PRIORITY", 99  # Special mode
    
    if vehicle_count == 0:
        status = "Empty Road"
        duration = 0 # Red light
    elif vehicle_count < 5:
        status = "Low Density"
        duration = min_green_time
    elif vehicle_count < 10:
        status = "Medium Density"
        duration = (min_green_time + max_green_time) // 2
    else:
        status = "High Density"
        duration = max_green_time
        
    return status, duration

# --- 3. MAIN LOOP ---
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Save the final output video
out = cv2.VideoWriter('final_output.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height))

# Simulation Variables
current_light = "RED"
time_remaining = 0
last_switch_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Prepare Image
    img_tensor = torchvision.transforms.functional.to_tensor(frame).to(device)
    img_tensor = img_tensor.unsqueeze(0)

    # Inference
    with torch.no_grad():
        prediction = model(img_tensor)[0]

    # Parse Detections
    boxes = prediction['boxes'].cpu().numpy()
    labels = prediction['labels'].cpu().numpy()
    scores = prediction['scores'].cpu().numpy()

    vehicle_count = 0
    has_ambulance = False

    for i in range(len(boxes)):
        if scores[i] > 0.5: # Confidence Threshold
            x1, y1, x2, y2 = boxes[i].astype(int)
            label_id = labels[i]
            label_name = class_dict.get(label_id, 'unknown')
            
            # Check for Ambulance
            if label_name == 'ambulance':
                has_ambulance = True
                color = (0, 0, 255) # Red box for ambulance
            else:
                color = (0, 255, 0) # Green box for others

            # Draw Box & Label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label_name}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            vehicle_count += 1

    # --- TRAFFIC SIGNAL LOGIC ---
    status, recommended_time = decide_signal_time(vehicle_count, has_ambulance)

    # Simulated Timer Logic (Just for visualization)
    # In a real system, this would send a signal to the hardware
    if has_ambulance:
        current_light = "GREEN (PRIORITY)"
        timer_text = "!!!"
        light_color = (0, 255, 0) # Green
    else:
        # Simple cycle: If we are Red, we switch to Green based on density
        current_light = f"GREEN for {recommended_time}s"
        timer_text = str(recommended_time)
        light_color = (0, 255, 0) if recommended_time > 0 else (0, 0, 255)

    # --- DASHBOARD VISUALIZATION ---
    # Draw a semi-transparent panel for stats
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (400, 150), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Display Stats
    cv2.putText(frame, f"Density: {status}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Vehicles: {vehicle_count}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Draw Traffic Light Circle
    cv2.circle(frame, (width - 50, 50), 30, (0, 0, 0), -1)      # Black Background
    cv2.circle(frame, (width - 50, 50), 25, light_color, -1)    # Light Color
    
    # Emergency Alert
    if has_ambulance:
        cv2.putText(frame, "EMERGENCY DETECTED!", (width//2 - 150, height - 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow('Smart Traffic Management', frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("Simulation Complete. Output saved to 'final_output.avi'")