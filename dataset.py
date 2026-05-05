import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

class DesertSegDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        
        # Ensure image and mask filenames match exactly
        self.image_filenames = sorted(os.listdir(images_dir))
        
        self.mapping = {
            100: 0,
            200: 1,
            300: 2,
            500: 3,
            550: 4,
            600: 5,
            700: 6,
            800: 7,
            7100: 8,
            10000: 9
        }

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, img_name)
        
        # Read image
        image = cv2.imread(img_path)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            raise FileNotFoundError(f"Image not found at {img_path}")
            
        # Read mask (using IMREAD_UNCHANGED to keep exact values like 7100, 10000)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(f"Mask not found at {mask_path}")

        # Remap mask pixel values to new labels
        remapped_mask = np.zeros_like(mask, dtype=np.int64)
        for orig_id, new_id in self.mapping.items():
            remapped_mask[mask == orig_id] = new_id

        # Apply albumentations
        if self.transform:
            augmented = self.transform(image=image, mask=remapped_mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            mask = remapped_mask
            
        # Convert to tensors if not already done by ToTensorV2
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        else:
            image = image.float()
            
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).long()
        else:
            mask = mask.long()
            
        return image, mask

def get_dataloaders(batch_size=4):
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomCrop(512, 512),
        A.RandomBrightnessContrast(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    val_transform = A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    train_dataset = DesertSegDataset(
        images_dir='train/Color_Images',
        masks_dir='train/Segmentation',
        transform=train_transform
    )

    val_dataset = DesertSegDataset(
        images_dir='val/Color_Images',
        masks_dir='val/Segmentation',
        transform=val_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    return train_loader, val_loader

if __name__ == "__main__":
    train_loader, val_loader = get_dataloaders(batch_size=4)
    
    print("Loading one batch from train_loader...")
    for images, masks in train_loader:
        print("Train batch images shape:", images.shape)
        print("Train batch masks shape:", masks.shape)
        break
        
    print("\nLoading one batch from val_loader...")
    for images, masks in val_loader:
        print("Val batch images shape:", images.shape)
        print("Val batch masks shape:", masks.shape)
        break
