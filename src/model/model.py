import torch
import torch.nn as nn
import torch.optim as optim

class SpeechEmotionRecognitionModel(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.n_features = 8
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, self.n_features, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(self.n_features),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(self.n_features, 2 * self.n_features, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(2 * self.n_features),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(2 * self.n_features, 4 * self.n_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4 * self.n_features),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(4 * self.n_features, 4 * self.n_features, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(4 * self.n_features),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4096, 120),
            nn.ReLU(),
            nn.Linear(120, 1))

    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.classifier(x)
        return x