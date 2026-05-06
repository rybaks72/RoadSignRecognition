import torch
from torchvision import transforms
from PIL import Image
import os
from model import GTSRBModel

def predict(image_path, model_path='models/gtsrb_final_model.pth'):
    # Define classes (GTSRB has 43 classes)
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
        transforms.Resize((30, 30)),
        transforms.ToTensor(),
        transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
    ])

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GTSRBModel(num_classes=43).to(device)
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please train the model first.")
        return None

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load and transform image
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        class_id = predicted.item()
        
    return classes.get(class_id, "Unknown")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        result = predict(img_path)
        if result:
            print(f"Predicted Class: {result}")
    else:
        print("Usage: python predict.py <image_path>")
