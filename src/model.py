import torch
import torch.nn as nn
import torch.nn.functional as F

class GTSRBModel(nn.Module):
    def __init__(self, num_classes):
        super(GTSRBModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.25)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.25)

        # Calculate flattened size based on input image (30x30) and pooling layers
        # (30 / 2 / 2) = 7.5 -> floor to 7.  So 7x7 input to the linear layer
        self.fc1 = nn.Linear(64 * 7 * 7, 256) # 64 filters * (30/4) * (30/4) approx
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = self.dropout1(x)

        x = F.relu(self.bn2(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        x = self.dropout2(x)

        x = x.view(-1, 64 * 7 * 7) # Flatten
        x = F.relu(self.bn3(self.fc1(x)))
        x = self.dropout3(x)
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    # Simple test for the model
    model = GTSRBModel(43) # GTSRB has 43 classes
    # Create a dummy input tensor (batch_size, channels, height, width)
    dummy_input = torch.randn(1, 3, 30, 30)
    output = model(dummy_input)
    print("Model output shape:", output.shape)
    print(model)
