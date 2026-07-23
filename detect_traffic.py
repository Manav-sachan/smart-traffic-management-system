import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
import cv2
import numpy as np
import os

# --- STEP 1: FIX THE VIDEO PATH ---
# Make sure your video is named EXACTLY 'traffic_video.mp4'
video_filename = 'traffic_video.mp4'

# This checks where the script is currently running
current_folder = os.getcwd()
video_path = os.path.join(current_folder, video_filename)

print("-" * 50)
print(f"Looking for video at: {video_path}")
print("-" * 50)

if not os.path.exists(video_path):
    print("❌ ERROR: File not found!")
    print(f"1. Make sure the video is inside this folder: {current_folder}")
    print(f"2. Check if the file is accidentally named '{video_filename}.mp4' (Windows sometimes hides the extension)")
    print("3. Try renaming your video to simply 'traffic' and update the code to 'traffic.mp4'")
    exit()
else:
    print("✅ Video file found! Proceeding...")

# --- STEP 2: LOAD MODEL (Updated to fix warnings) ---
print("Loading model...")
# New syntax to remove the warnings
weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn(weights=weights)
model.eval()

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f"✅ Model loaded on {device}")

# --- STEP 3: PROCESSING ---
# Classes we care about (Car, Motorcycle, Bus, Truck)
RELEVANT_CLASSES = [3, 4, 6, 8]
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Save output
out = cv2.VideoWriter('output_traffic.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height))

print("Processing video... Press 'q' to stop early.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img_tensor = torchvision.transforms.functional.to_tensor(frame).to(device)
    img_tensor = img_tensor.unsqueeze(0)

    with torch.no_grad():
        prediction = model(img_tensor)[0]

    boxes = prediction['boxes'].cpu().numpy()
    labels = prediction['labels'].cpu().numpy()
    scores = prediction['scores'].cpu().numpy()

    vehicle_count = 0

    for i in range(len(boxes)):
        if scores[i] > 0.6 and labels[i] in RELEVANT_CLASSES:
            x1, y1, x2, y2 = boxes[i].astype(int)
            class_name = COCO_INSTANCE_CATEGORY_NAMES[labels[i]]
            
            # Draw Green Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            vehicle_count += 1

    # Display Count
    cv2.putText(frame, f"Vehicles: {vehicle_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    cv2.imshow('Traffic Detection', frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("Done!")