import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from dataset import get_dataloaders
from model import get_model

def train():
    # Device setup
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load data
    train_loader, val_loader = get_dataloaders(batch_size=4)

    # Load model
    model = get_model(num_classes=10)
    model.to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    # Training loop configuration
    num_epochs = 20
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        
        # --- Train Phase ---
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc="Training")
        
        for images, masks in train_pbar:
            images = images.to(device)
            # Assuming masks are appropriately shaped for CrossEntropyLoss
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Compute loss
            loss = criterion(outputs, masks)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_train_loss = train_loss / len(train_loader)

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_pbar = tqdm(val_loader, desc="Validation")
        
        with torch.no_grad():
            for images, masks in val_pbar:
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                val_loss += loss.item()
                val_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
                
        avg_val_loss = val_loss / len(val_loader)

        # Print epoch summary
        print(f"Epoch {epoch+1} Summary - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            print(f"Validation loss decreased ({best_val_loss:.4f} --> {avg_val_loss:.4f}). Saving model!")
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')

if __name__ == "__main__":
    train()
