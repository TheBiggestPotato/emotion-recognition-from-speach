import torch
import torch.nn as nn
import torch.optim as optim

class SpeechEmotionRecognitionModel(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(SpeechEmotionRecognitionModel, self).__init__()
        # Define your model architecture here
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * input_shape[0] * input_shape[1], 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.GELU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Define forward pass
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss:.4f}")
    return model

def save_model(model, file_path):
    torch.save(model.state_dict(), file_path)

def load_model(file_path, model):
    model.load_state_dict(torch.load(file_path))
    return model
