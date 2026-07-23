import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import os


model_path = "traffic_model.pth"       
images_folder = "new_train_images"     
output_labels_folder = "new_yolo_labels"
confidence_threshold = 0.6             


class_map = {1: 0, 2: 1, 3: 2, 4: 3}


if not os.path.exists(output_labels_folder):
    os.makedirs(output_labels_folder)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Loading AI on: {device}")


def get_model(num_classes):
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


model = get_model(5) 
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()


image_files = [f for f in os.listdir(images_folder) if f.endswith('.jpg')]
print(f"Starting auto-annotation for {len(image_files)} images...")
print("-" * 50)

total_boxes_drawn = 0

for img_name in image_files:
    img_path = os.path.join(images_folder, img_name)
    
    
    img = Image.open(img_path).convert("RGB")
    img_width, img_height = img.size
    
    img_tensor = torchvision.transforms.functional.to_tensor(img).to(device).unsqueeze(0)
    
    with torch.no_grad():
        prediction = model(img_tensor)[0]
        
    boxes = prediction['boxes'].cpu().numpy()
    labels = prediction['labels'].cpu().numpy()
    scores = prediction['scores'].cpu().numpy()
    
    txt_name = img_name.replace('.jpg', '.txt')
    txt_path = os.path.join(output_labels_folder, txt_name)
    
 
    with open(txt_path, 'w') as f:
        for i in range(len(boxes)):
            if scores[i] >= confidence_threshold:
                frcnn_label = labels[i]
                
               
                if frcnn_label not in class_map:
                    continue
                    
                yolo_label = class_map[frcnn_label]
                
            
                x1, y1, x2, y2 = boxes[i]
                
               
                x_center = ((x1 + x2) / 2.0) / img_width
                y_center = ((y1 + y2) / 2.0) / img_height
                width = (x2 - x1) / img_width
                height = (y2 - y1) / img_height
                
              
                x_center, y_center = max(0, min(1, x_center)), max(0, min(1, y_center))
                width, height = max(0, min(1, width)), max(0, min(1, height))
                
                f.write(f"{yolo_label} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                total_boxes_drawn += 1

print("-" * 50)
print(f"✅ Auto-Annotation Complete!")
print(f"Generated labels for {len(image_files)} images.")
print(f"Total vehicles detected and labeled: {total_boxes_drawn}")
print(f"Check the '{output_labels_folder}' directory for your .txt files.")
