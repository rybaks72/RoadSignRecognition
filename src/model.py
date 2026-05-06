import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleCNN(nn.Module):
    def __init__(self, num_classes=43):
        super(MultiScaleCNN, self).__init__()
        
        # Stage 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Stage 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Stage 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5)
        # No pool3 here, input 5x5 -> output 1x1
        
        # Multi-scale feature combination
        # Output of Stage 2 (after pool2) will be branched out
        # Output of Stage 3 will be combined with branched Stage 2
        
        # Fully connected layers
        # Input size calculation:
        # Input: 32x32
        # After conv1: 28x28
        # After pool1: 14x14
        # After conv2: 10x10
        # After pool2: 5x5 (Feature set 1)
        # After conv3: 1x1 (Feature set 2)
        
        self.fc1 = nn.Linear(64*5*5 + 128*1*1, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Stage 1
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        
        # Stage 2
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        feat1 = x.view(-1, 64*5*5) # Branch out
        
        # Stage 3
        x = F.relu(self.conv3(x))
        feat2 = x.view(-1, 128*1*1)
        
        # Concatenate features
        combined = torch.cat((feat1, feat2), dim=1)
        
        x = F.relu(self.fc1(combined))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

if __name__ == "__main__":
    model = MultiScaleCNN()
    print(model)
    dummy_input = torch.randn(1, 3, 32, 32)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
