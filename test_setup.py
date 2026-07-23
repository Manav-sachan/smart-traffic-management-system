import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import cv2
import numpy as np

print("Libraries imported successfully!")

# 1. Load the pre-trained Faster R-CNN model
# This downloads a model pre-trained on the COCO dataset (includes cars, buses, etc.)
print("Downloading Faster R-CNN model... (this happens only once)")
model = fasterrcnn_resnet50_fpn(pretrained=True)

# 2. Set the model to evaluation mode (essential for inference)
model.eval()

print("✅ Model loaded successfully!")
print("✅ OpenCV version:", cv2.__version__)
print("✅ PyTorch version:", torch.__version__)

if torch.cuda.is_available():
    print("✅ GPU is available (Training will be fast!)")
else:
    print("⚠️ GPU not detected (Running on CPU - slightly slower but works)")