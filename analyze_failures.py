import torch
import numpy as np
import os
import cv2
from tqdm import tqdm
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

from dataset import DesertSegDataset
from model import get_model

def compute_iou(pred, mask, num_classes=10):
    iou_list = []
    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (mask == cls)
        intersection = np.logical_and(pred_inds, target_inds).sum()
        union = np.logical_or(pred_inds, target_inds).sum()
        if union == 0:
            continue
        iou_list.append(intersection / union)
    return np.mean(iou_list) if iou_list else 0

def visualize_failure(image, mask, pred, iou, save_path):
    # Denormalize image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image.transpose(1, 2, 0) * std + mean) * 255
    image = image.clip(0, 255).astype(np.uint8)

    # Color palette
    colors = np.array([
        [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
        [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
        [64, 0, 0], [192, 0, 0]
    ], dtype=np.uint8)

    mask_color = colors[mask]
    pred_color = colors[pred]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image)
    axes[0].set_title("Input Image")
    axes[0].axis("off")
    
    axes[1].imshow(mask_color)
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")
    
    axes[2].imshow(pred_color)
    axes[2].set_title(f"Prediction (IoU: {iou:.3f})")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
    model = get_model(num_classes=10)
    model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    val_transform = A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    val_dataset = DesertSegDataset(
        images_dir='val/Color_Images',
        masks_dir='val/Segmentation',
        transform=val_transform
    )

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    results = []
    os.makedirs("failure_analysis", exist_ok=True)

    print("🔍 Analyzing samples for failures...")
    with torch.no_grad():
        for i, (image, mask) in enumerate(tqdm(val_loader)):
            image_dev = image.to(DEVICE)
            output = model(image_dev)
            pred = torch.argmax(output, dim=1).cpu().numpy()[0]
            mask_np = mask.numpy()[0]
            
            iou = compute_iou(pred, mask_np)
            results.append((iou, i, image.numpy()[0], mask_np, pred))

    # Sort by IoU (worst first)
    results.sort(key=lambda x: x[0])

    print("📊 Saving top 5 failure cases...")
    for j in range(5):
        iou, idx, img, msk, prd = results[j]
        save_path = f"failure_analysis/failure_{j+1}_iou_{iou:.3f}.png"
        visualize_failure(img, msk, prd, iou, save_path)
        print(f"Saved {save_path}")

    print("📊 Saving top 5 best cases for comparison...")
    for j in range(1, 6):
        iou, idx, img, msk, prd = results[-j]
        save_path = f"failure_analysis/success_{j}_iou_{iou:.3f}.png"
        visualize_failure(img, msk, prd, iou, save_path)
        print(f"Saved {save_path}")

if __name__ == "__main__":
    main()
