import os
import torch
import torch.utils.data
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import xml.etree.ElementTree as ET

# --- CONFIGURATION ---
imgs_path = r"train_data/images"
xmls_path = r"train_data/annotations"

class_dict = {
    'background': 0,
    'car': 1,
    'bus': 2,
    'bike': 3,
    'ambulance': 4
}
# Helmet Eye - Smart Traffic Management System
# --- CUSTOM DATASET CLASS ---
class TrafficDataset(torch.utils.data.Dataset):
    def __init__(self, imgs_path, xmls_path, class_dict, transforms=None):
        self.imgs_path = imgs_path
        self.xmls_path = xmls_path
        self.class_dict = class_dict
        self.transforms = transforms
        
        self.imgs = []
        all_files = [f for f in os.listdir(imgs_path) if f.endswith('.jpg')]
        
        print("Checking data quality...")
        for img_file in all_files:
            xml_file = img_file.replace('.jpg', '.xml')
            xml_full_path = os.path.join(self.xmls_path, xml_file)
            
            # Check if XML exists
            if not os.path.exists(xml_full_path):
                continue
            
            # Check if XML actually has boxes inside
            try:
                tree = ET.parse(xml_full_path)
                root = tree.getroot()
                objects = root.findall('object')
                if len(objects) > 0:
                    self.imgs.append(img_file)
            except:
                continue
        
        self.imgs.sort()
        print(f"✅ Final Count: {len(self.imgs)} valid images ready for training.")

    def __getitem__(self, idx):
        img_file = self.imgs[idx]
        img_path = os.path.join(self.imgs_path, img_file)
        img = Image.open(img_path).convert("RGB")
        
        xml_file = img_file.replace('.jpg', '.xml')
        xml_path = os.path.join(self.xmls_path, xml_file)
        
        boxes = []
        labels = []
        
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            for obj in root.findall('object'):
                name = obj.find('name').text
                if name in self.class_dict:
                    label = self.class_dict[name]
                    bndbox = obj.find('bndbox')
                    xmin = float(bndbox.find('xmin').text)
                    ymin = float(bndbox.find('ymin').text)
                    xmax = float(bndbox.find('xmax').text)
                    ymax = float(bndbox.find('ymax').text)
                    
                    # Safety check: Prevent negative or zero-width boxes
                    if xmax > xmin and ymax > ymin:
                        boxes.append([xmin, ymin, xmax, ymax])
                        labels.append(label)
        except:
            pass # Skip corrupted XMLs
        
        # If no valid boxes found, return None (We will filter this out in collate_fn)
        if len(boxes) == 0:
            return None
            
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])
        
        if self.transforms:
            img = self.transforms(img)
            
        return img, target

    def __len__(self):
        return len(self.imgs)

# --- HELPER: SAFE COLLATE FUNCTION ---
# This function removes any "None" items from the batch (images with bad boxes)
def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None 
    return tuple(zip(*batch))

# --- MAIN TRAINING LOOP ---
if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Training on: {device}")

    data_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    
    dataset = TrafficDataset(imgs_path, xmls_path, class_dict, transforms=data_transform)
    
    if len(dataset) == 0:
        print("❌ ERROR: No valid data found!")
        exit()

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=collate_fn
    )

    num_classes = len(class_dict)
    
    # Load Model
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    num_epochs = 32
    print("Starting Training...")
    
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        batch_count = 0
        
        for i, batch in enumerate(data_loader):
            # Skip empty batches (caused by the collate_fn filtering bad images)
            if batch is None:
                continue
                
            images, targets = batch
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Standard Training Step
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()
            batch_count += 1
            
            if i % 2 == 0:
                print(f"Epoch: {epoch+1}, Step: {i}, Loss: {losses.item():.4f}")

        if batch_count > 0:
            avg_loss = epoch_loss / batch_count
            print(f"--- Epoch {epoch+1} Finished. Avg Loss: {avg_loss:.4f} ---")
        else:
            print(f"--- Epoch {epoch+1} Finished. No valid batches processed. ---")

    print("Saving model...")
    torch.save(model.state_dict(), "traffic_model.pth")
    print("✅ Training Complete! Model saved as 'traffic_model.pth'")