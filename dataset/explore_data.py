import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def main():
    # 1. Define directories
    train_rgb_dir = "dataset/Train/rgb"
    val_rgb_dir = "dataset/Val/rgb"
    test_dir = "dataset/testImages"
    
    train_seg_dir = "dataset/Train/seg"
    val_seg_dir = "dataset/Val/seg"

    # Helper function to find images safely
    def count_images(dir_path):
        if not os.path.exists(dir_path):
            return 0
        extensions = ('*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff')
        files = []
        for ext in extensions:
            files.extend(glob.glob(os.path.join(dir_path, ext)))
        return len(files)

    # Count number of images
    num_train = count_images(train_rgb_dir)
    num_val = count_images(val_rgb_dir)
    num_test = count_images(test_dir)
    total_images = num_train + num_val + num_test

    # 2. Load all segmentation masks
    train_masks = []
    if os.path.exists(train_seg_dir):
        for ext in ('*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff'):
            train_masks.extend(glob.glob(os.path.join(train_seg_dir, ext)))
            
    val_masks = []
    if os.path.exists(val_seg_dir):
        for ext in ('*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff'):
            val_masks.extend(glob.glob(os.path.join(val_seg_dir, ext)))

    all_mask_paths = train_masks + val_masks

    # 3 & 4. Extract unique pixel values and calculate pixel frequency
    expected_classes = [100, 200, 300, 500, 550, 600, 700, 800, 7100, 10000]
    class_counts = {cls: 0 for cls in expected_classes}
    other_classes = {}

    if all_mask_paths:
        print("Processing masks to calculate pixel frequencies...")
        for mask_path in tqdm(all_mask_paths):
            # Read mask using IMREAD_UNCHANGED to preserve values > 255 (e.g., 16-bit)
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            
            if mask is None:
                print(f"Warning: Could not read mask {mask_path}")
                continue
                
            unique_vals, counts = np.unique(mask, return_counts=True)
            
            for val, count in zip(unique_vals, counts):
                if val in class_counts:
                    class_counts[val] += count
                else:
                    other_classes[val] = other_classes.get(val, 0) + count

    total_pixels = sum(class_counts.values()) + sum(other_classes.values())
    
    classes_found = list(class_counts.keys()) + list(other_classes.keys())
    classes_found.sort()

    # 5. Plot a bar chart of class distribution
    os.makedirs("outputs", exist_ok=True)
    
    labels = [str(cls) for cls in expected_classes]
    counts = [class_counts[cls] for cls in expected_classes]

    plt.figure(figsize=(10, 6))
    plt.bar(labels, counts, color='skyblue', edgecolor='black')
    plt.xlabel('Class IDs (Pixel Values)')
    plt.ylabel('Pixel Frequency')
    plt.title('Class Distribution across Dataset')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('outputs/class_distribution.png')
    
    # 7. Print summary
    print("\n" + "="*40)
    print("DATASET EXPLORATION SUMMARY")
    print("="*40)
    print(f"Total images (Train/Val/Test): {total_images}")
    print(f"Classes found: {classes_found}")
    
    # 6. Print warning if any class has less than 5% of total pixels
    print("\n--- Imbalance Warnings ---")
    imbalance_found = False
    
    if total_pixels > 0:
        for cls in expected_classes:
            freq = class_counts[cls]
            percentage = (freq / total_pixels) * 100
            if percentage < 5.0:
                print(f"WARNING: Class {cls} has less than 5% of total pixels ({percentage:.2f}%)")
                imbalance_found = True

        for cls, freq in other_classes.items():
            percentage = (freq / total_pixels) * 100
            print(f"WARNING: Unexpected class {cls} has {percentage:.2f}% of pixels")
            imbalance_found = True
    else:
        print("No masks were processed to check for imbalance.")

    if not imbalance_found and total_pixels > 0:
        print("No class imbalance detected (all >5%).")

if __name__ == "__main__":
    main()
