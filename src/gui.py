import tkinter as tk
from tkinter import filedialog, Label, Button, Frame
from PIL import Image, ImageTk
import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import os

# Define the model class
class MultiScaleCNN(nn.Module):
    def __init__(self, num_classes=43):
        super(MultiScaleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5)
        self.fc1 = nn.Linear(64*5*5 + 128*1*1, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        feat1 = x.view(-1, 64*5*5)
        x = F.relu(self.conv3(x))
        feat2 = x.view(-1, 128*1*1)
        combined = torch.cat((feat1, feat2), dim=1)
        x = F.relu(self.fc1(combined))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Define classes
classes = {
    0: 'Speed limit (20km/h)', 1: 'Speed limit (30km/h)', 2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)', 4: 'Speed limit (70km/h)', 5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)', 7: 'Speed limit (100km/h)', 8: 'Speed limit (120km/h)',
    9: 'No passing', 10: 'No passing for vehicles over 3.5 metric tons',
    11: 'Right-of-way at the next intersection', 12: 'Priority road', 13: 'Yield',
    14: 'Stop', 15: 'No vehicles', 16: 'Vehicles over 3.5 metric tons prohibited',
    17: 'No entry', 18: 'General caution', 19: 'Dangerous curve to the left',
    20: 'Dangerous curve to the right', 21: 'Double curve', 22: 'Bumpy road',
    23: 'Slippery road', 24: 'Road narrows on the right', 25: 'Road work',
    26: 'Traffic signals', 27: 'Pedestrians', 28: 'Children crossing',
    29: 'Bicycles crossing', 30: 'Beware of ice/snow', 31: 'Wild animals crossing',
    32: 'End of all speed and passing limits', 33: 'Turn right ahead',
    34: 'Turn left ahead', 35: 'Ahead only', 36: 'Go straight or right',
    37: 'Go straight or left', 38: 'Keep right', 39: 'Keep left',
    40: 'Roundabout mandatory', 41: 'End of no passing',
    42: 'End of no passing by vehicles over 3.5 metric tons'
}

# Data transformations
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
])

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiScaleCNN(num_classes=43).to(device)
model_path = 'models/traffic_sign_model.pth'

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
else:
    print(f"Model not found at {model_path}. Please train the model first.")
    # Handle missing model case if necessary, e.g., disable prediction button
    
def predict(image):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        class_id = predicted.item()
    return classes.get(class_id, "Unknown")

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Road Sign Recognition")
        self.file_path = None

        self.title_label = Label(root, text="Road Sign Recognition Application", font=("Helvetica", 16))
        self.title_label.pack(pady=10)

        self.authors_label = Label(root, text="Authors: Sylwia Rybak, Wojciech Sendek, Stanisław Zieliński, Jakub Szostak")
        self.authors_label.pack()

        self.controls_frame = Frame(root)
        self.controls_frame.pack(pady=10)

        self.upload_button = Button(self.controls_frame, text="Upload Image", command=self.upload_image)
        self.upload_button.pack(side=tk.LEFT, padx=5)

        self.classify_button = Button(self.controls_frame, text="Classify Sign", command=self.classify_image, state=tk.DISABLED)
        self.classify_button.pack(side=tk.LEFT, padx=5)

        self.reset_button = Button(self.controls_frame, text="Clear", command=self.reset)
        self.reset_button.pack(side=tk.LEFT, padx=5)

        self.image_label = Label(root)
        self.image_label.pack(pady=10)

        self.result_label = Label(root, text="", font=("Helvetica", 12))
        self.result_label.pack(pady=10)

    def upload_image(self):
        self.file_path = filedialog.askopenfilename()
        if self.file_path:
            image = Image.open(self.file_path)
            image.thumbnail((300, 300))
            self.photo = ImageTk.PhotoImage(image)
            self.image_label.config(image=self.photo)
            self.classify_button.config(state=tk.NORMAL)
            self.result_label.config(text="")

    def classify_image(self):
        if self.file_path:
            image = Image.open(self.file_path).convert('RGB')
            prediction = predict(image)
            self.result_label.config(text=f"Predicted Class: {prediction}")

    def reset(self):
        self.file_path = None
        self.image_label.config(image='')
        self.result_label.config(text="")
        self.classify_button.config(state=tk.DISABLED)

def main():
    root = tk.Tk()
    app = App(root)
    root.mainloop()

if __name__ == "__main__":
    main()
