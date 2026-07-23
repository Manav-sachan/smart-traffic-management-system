import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou
import os
from PIL import Image
import xml.etree.ElementTree as ET
import numpy as np


weights_path = "traffic_model.pth"  
imgs_path = r"train_data/images"
xmls_path = r"train_data/annotations"


class_dict = {'background': 0, 'car': 1, 'bus': 2, 'bike': 3, 'ambulance': 4}


def get_model(num_classes):
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def get_ground_truth(xml_path):
    boxes = []
    labels = []
    if os.path.exists(xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name in class_dict:
                bndbox = obj.find('bndbox')
                xmin = float(bndbox.find('xmin').text)
                ymin = float(bndbox.find('ymin').text)
                xmax = float(bndbox.find('xmax').text)
                ymax = float(bndbox.find('ymax').text)
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(class_dict[name])
    return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Evaluating on: {device}")


if not os.path.exists(weights_path):
    print("❌ ERROR: 'traffic_model.pth' not found. Run training first!")
    exit()

num_classes = len(class_dict)
model = get_model(num_classes)
model.load_state_dict(torch.load(weights_path, map_location=device))
model.to(device)
model.eval()


image_files = [f for f in os.listdir(imgs_path) if f.endswith('.jpg')]
total_correct_predictions = 0
total_actual_vehicles = 0

print("-" * 50)
print("Checking accuracy on labeled images...")
print("-" * 50)

for img_file in image_files:
  
    img_path = os.path.join(imgs_path, img_file)
    xml_path = os.path.join(xmls_path, img_file.replace('.jpg', '.xml'))
    
    img = Image.open(img_path).convert("RGB")
    img_tensor = torchvision.transforms.functional.to_tensor(img).to(device)
    
   
    gt_boxes, gt_labels = get_ground_truth(xml_path)
    
    if len(gt_boxes) == 0:
        continue 
        
    gt_boxes = gt_boxes.to(device)
    total_actual_vehicles += len(gt_boxes)

   
    with torch.no_grad():
        prediction = model([img_tensor])[0]
    
    pred_boxes = prediction['boxes']
    pred_scores = prediction['scores']
    
   
    keep = pred_scores > 0.5
    pred_boxes = pred_boxes[keep]
    
    if len(pred_boxes) > 0:
        
        ious = box_iou(gt_boxes, pred_boxes)
        
     
        matches = (ious.max(dim=1)[0] > 0.5).sum().item()
        total_correct_predictions += matches


accuracy = (total_correct_predictions / total_actual_vehicles) * 100

print("-" * 50)
print(f"Total Vehicles in Manual Labels: {total_actual_vehicles}")
print(f"Total Correctly Detected by AI:  {total_correct_predictions}")
print("-" * 50)
print(f"✅ MODEL ACCURACY (Recall): {accuracy:.2f}%")
print("-" * 50)
