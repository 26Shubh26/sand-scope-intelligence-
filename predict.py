import torch
import cv2
import numpy as np
import os

from model import get_model
import albumentations as A
from albumentations.pytorch import ToTensorV2

# -------------------------
# START
# -------------------------
print("🚀 START")

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print("Device:", DEVICE)

# -------------------------
# LOAD MODEL
# -------------------------
model = get_model(num_classes=10)
model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

print("✅ Model loaded")

# -------------------------
# LOAD IMAGE
# -------------------------
folder = "Offroad_Segmentation_testImages/Color_Images"
files = os.listdir(folder)

print("Files:", len(files))

image_path = os.path.join(folder, files[0])
print("Using:", image_path)

image = cv2.imread(image_path)

if image is None:
    print("❌ image load fail")
    exit()

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
original = image.copy()

val_transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

augmented = val_transform(image=image)
tensor_image = augmented["image"].unsqueeze(0).to(DEVICE)

with torch.no_grad():
    output = model(tensor_image)
    pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()

print("✅ Prediction done")

# Fixed color palette for 10 classes
colors = np.array([
    [0, 0, 0],         # Class 0: Background/Unknown
    [128, 0, 0],       # Class 1
    [0, 128, 0],       # Class 2
    [128, 128, 0],     # Class 3
    [0, 0, 128],       # Class 4
    [128, 0, 128],     # Class 5
    [0, 128, 128],     # Class 6
    [128, 128, 128],   # Class 7
    [64, 0, 0],        # Class 8
    [192, 0, 0],       # Class 9
], dtype=np.uint8)

segmented = colors[pred]

segmented = cv2.resize(segmented, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_NEAREST)
overlay = cv2.addWeighted(original, 0.6, segmented, 0.4, 0)

cv2.imwrite("output.png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

print("🔥 DONE — check output.png")