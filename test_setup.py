import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import cv2
import numpy as np

print("Libraries imported successfully!")


print("Downloading Faster R-CNN model... (this happens only once)")
model = fasterrcnn_resnet50_fpn(pretrained=True)


model.eval()

print("✅ Model loaded successfully!")
print("✅ OpenCV version:", cv2.__version__)
print("✅ PyTorch version:", torch.__version__)

if torch.cuda.is_available():
    print("✅ GPU is available (Training will be fast!)")
else:
    print("⚠️ GPU not detected (Running on CPU - slightly slower but works)")
