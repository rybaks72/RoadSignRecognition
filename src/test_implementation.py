import torch
import torch.nn as nn
import torch.optim as optim

from model import GTSRBModel


def test_model_structure():
    print("Testing GTSRBModel structure...")
    model = GTSRBModel(num_classes=43)
    model.eval()
    dummy_input = torch.randn(1, 3, 30, 30)
    output = model(dummy_input)
    assert output.shape == (1, 43), f"Expected shape (1, 43), got {output.shape}"
    print("Model structure test passed!")


def test_training_iteration():
    print("Testing a single training iteration with mock data...")
    model = GTSRBModel(num_classes=2)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Mock data: 2 samples, 3x30x30 images, 2 classes.
    inputs = torch.randn(2, 3, 30, 30)
    labels = torch.tensor([0, 1])

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    print(f"Loss after 1 iteration: {loss.item()}")
    assert loss.item() > 0
    print("Training iteration test passed!")


if __name__ == "__main__":
    test_model_structure()
    test_training_iteration()
