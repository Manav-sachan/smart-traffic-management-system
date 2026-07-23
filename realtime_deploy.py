import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import cv2
import numpy as np
import time

# --- CONFIGURATION ---
model_path = "traffic_model.pth"

# *** FIX IS HERE ***
# We use 0 to force the Laptop Webcam. 
# Do not change this to a URL unless you have a phone connected.
camera_source = 0 

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
class_dict = {0: 'background', 1: 'car', 2: 'bus', 3: 'bike', 4: 'ambulance'}

# --- LOAD MODEL ---
def get_model(num_classes):
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

print(f"Loading system on {device}...")
model = get_model(len(class_dict))

if torch.cuda.is_available():
    model.load_state_dict(torch.load(model_path))
else:
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

model.to(device)
model.eval()
print("✅ System Ready. Opening Camera...")

# --- TRAFFIC LOGIC ---
def get_traffic_status(count, ambulance_found):
    if ambulance_found:
        return "EMERGENCY", (0, 0, 255), 99
    if count == 0:
        return "Empty", (0, 0, 255), 0
    elif count < 3:
        return "Low", (0, 255, 0), 5
    elif count < 7:
        return "Medium", (0, 255, 0), 10
    else:
        return "High", (0, 255, 0), 20

# --- MAIN LOOP ---
cap = cv2.VideoCapture(camera_source)

# Resolution setup
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("❌ ERROR: Could not open webcam. Make sure no other app (Zoom/Teams) is using it.")
    exit()

frame_count = 0
last_boxes = []
last_labels = []
last_scores = []
vehicle_count = 0
has_ambulance = False

print("Press 'q' to exit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break
    
    # Resize for speed
    input_frame = cv2.resize(frame, (640, 480))
    
    # Run AI every 2nd frame
    if frame_count % 2 == 0:
        img_tensor = torchvision.transforms.functional.to_tensor(input_frame).to(device)
        img_tensor = img_tensor.unsqueeze(0)

        with torch.no_grad():
            prediction = model(img_tensor)[0]

        last_boxes = prediction['boxes'].cpu().numpy()
        last_labels = prediction['labels'].cpu().numpy()
        last_scores = prediction['scores'].cpu().numpy()
    
    frame_count += 1
    
    # Draw Results
    vehicle_count = 0
    has_ambulance = False
    
    scale_x = frame.shape[1] / 640
    scale_y = frame.shape[0] / 480

    for i in range(len(last_boxes)):
        if last_scores[i] > 0.5:
            x1 = int(last_boxes[i][0] * scale_x)
            y1 = int(last_boxes[i][1] * scale_y)
            x2 = int(last_boxes[i][2] * scale_x)
            y2 = int(last_boxes[i][3] * scale_y)
            
            label_name = class_dict.get(last_labels[i], 'unknown')
            
            if label_name == 'ambulance':
                has_ambulance = True
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label_name}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            vehicle_count += 1

    status_text, light_color, duration = get_traffic_status(vehicle_count, has_ambulance)
    
    # Dashboard
    cv2.rectangle(frame, (0, 0), (300, 120), (0, 0, 0), -1)
    cv2.putText(frame, f"Status: {status_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Vehicles: {vehicle_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Green Time: {duration}s", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Traffic Light
    cv2.circle(frame, (frame.shape[1] - 50, 50), 30, (50, 50, 50), -1) 
    cv2.circle(frame, (frame.shape[1] - 50, 50), 25, light_color, -1)

    cv2.imshow('Real-Time Traffic System', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()