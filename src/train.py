import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

# Assuming model.py is in the same directory, import the model class
from model import GTSRBModel

# --- Configuration ---
IMG_HEIGHT = 30
IMG_WIDTH = 30
CHANNELS = 3
NUM_CLASSES = 43 # GTSRB has 43 classes
EPOCHS = 15 # Can be increased for better performance
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# Path to the extracted dataset
DATASET_PATH = '/content/RoadSignRecognition/data/gtsrb-german-traffic-sign/'
TRAIN_DIR = os.path.join(DATASET_PATH, 'Train')
TEST_DIR = os.path.join(DATASET_PATH, 'Test') # Note: This will be used for testing, not validation here.

# Ensure the training directory exists
if not os.path.exists(TRAIN_DIR):
    raise FileNotFoundError(f"Training directory not found: {TRAIN_DIR}")

# --- Data Preprocessing and Augmentation ---
# Define transformations for training data with augmentation
train_transform = transforms.Compose([
    transforms.Resize((IMG_WIDTH, IMG_HEIGHT)),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
    transforms.ToTensor(),
    transforms.Normalize((0.3403, 0.3121, 0.3214), (0.1340, 0.1295, 0.1386)) # GTSRB dataset mean and std
])

# Define transformations for validation/test data (no augmentation, just resize and normalize)
val_transform = transforms.Compose([
    transforms.Resize((IMG_WIDTH, IMG_HEIGHT)),
    transforms.ToTensor(),
    transforms.Normalize((0.3403, 0.3121, 0.3214), (0.1340, 0.1295, 0.1386))
])

# Load the full training dataset
full_train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)

# Split the training dataset into training and validation sets
train_size = int(0.8 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_train_dataset, [train_size, val_size])

# Adjust validation dataset transform (remove augmentation)
val_dataset.dataset.transform = val_transform

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# --- Model Building ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GTSRBModel(NUM_CLASSES).to(device)

# --- Model Compilation (Optimizer and Loss Function) ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(model)

# --- Training Loop ---
print("Starting model training...")
history = {'train_loss': [], 'train_accuracy': [], 'val_loss': [], 'val_accuracy': []}
best_val_accuracy = 0.0

for epoch in range(EPOCHS):
    model.train() # Set model to training mode
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    for i, (inputs, labels) in enumerate(train_loader):
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

    model.eval() # Set model to evaluation mode
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad(): # Disable gradient calculation for validation
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

    print(f'Epoch {epoch+1}/{EPOCHS}, ' \
          f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, ' \
          f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')

    # Save the best model based on validation accuracy
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), 'gtsrb_model_best.pth')
        print("Saved best model weights to 'gtsrb_model_best.pth'")

print("Model training finished.")

# --- Save the final trained model (if not already saved as best) ---
torch.save(model.state_dict(), 'gtsrb_final_model.pth')
print("Final model saved as 'gtsrb_final_model.pth'")

# Optional: Plot training history
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

plt.show()
