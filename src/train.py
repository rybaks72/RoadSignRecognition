import os
import re

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import GTSRBModel

# --- Configuration ---
IMG_HEIGHT = 48
IMG_WIDTH = 48
NUM_CLASSES = 43  # GTSRB has 43 classes
EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 0.0005

# Change this path if your dataset is somewhere else.
DATASET_PATH = 'data/archive'
TRAIN_DIR = os.path.join(DATASET_PATH, 'Train')


def get_next_model_paths(directory: str = 'models'):
    """Determine the next model version number and return the path for the next model."""
    os.makedirs(directory, exist_ok=True)
    files = os.listdir(directory)
    # Pattern to match gtsrb_model_{number}.pth
    pattern = re.compile(r"^gtsrb_model_(\d+)\.pth$")

    max_num = 0
    for f in files:
        match = pattern.match(f)
        if match:
            try:
                num = int(match.group(1))
                max_num = max(max_num, num)
            except ValueError:
                continue

    next_num = max_num + 1
    final_name = f"gtsrb_model_{next_num}.pth"

    return os.path.join(directory, final_name)


def train_model(
    train_dir: str = TRAIN_DIR,
    model_output_path: str = None,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
):
    """Train the GTSRB CNN model and save weights to the models directory."""
    if model_output_path is None:
        model_output_path = get_next_model_paths()

    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training directory not found: {train_dir}")

    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)

    train_transform = transforms.Compose([
        transforms.Resize((IMG_WIDTH + 8, IMG_HEIGHT + 8)),
        transforms.RandomResizedCrop((IMG_WIDTH, IMG_HEIGHT), scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
        transforms.ToTensor(),
        transforms.Normalize((0.3403, 0.3121, 0.3214), (0.1340, 0.1295, 0.1386))
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_WIDTH, IMG_HEIGHT)),
        transforms.ToTensor(),
        transforms.Normalize((0.3403, 0.3121, 0.3214), (0.1340, 0.1295, 0.1386))
    ])

    train_dataset_full = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset_full = datasets.ImageFolder(train_dir, transform=val_transform)

    num_samples = len(train_dataset_full)
    train_size = int(0.8 * num_samples)
    val_size = num_samples - train_size
    
    # Use a fixed seed for reproducibility of the split
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset_full, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    # The validation dataset should use the val_dataset_full's transform
    val_dataset = torch.utils.data.Subset(val_dataset_full, val_dataset.indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GTSRBModel(NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    print(model)
    print("Starting model training...")

    history = {'train_loss': [], 'train_accuracy': [], 'val_loss': [], 'val_accuracy': []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_accuracy)

        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = 100 * correct_val / total_val
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch + 1}/{epochs}, "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%"
        )

    torch.save(model.state_dict(), model_output_path)
    print(f"Final model saved as '{model_output_path}'")

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    # Use a matching name for the history plot
    base_name = os.path.splitext(os.path.basename(model_output_path))[0]
    plot_path = os.path.join(os.path.dirname(model_output_path), f"{base_name}_history.png")
    plt.savefig(plot_path)
    print(f"Training history plot saved to '{plot_path}'")
    
    # Try to show the plot, but don't block if it's not possible
    try:
        plt.show()
    except Exception:
        pass

    return history


if __name__ == "__main__":
    train_model()
