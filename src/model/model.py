import torch
import torch.nn as nn
import torch.optim as optim

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.n_features = 16
        
        self.layer1 = None

        self.layer2 = nn.Sequential(
            nn.Conv2d(self.n_features, 2 * self.n_features, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(2 * self.n_features),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(2 * self.n_features, 4 * self.n_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4 * self.n_features),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        # self.layer4 = nn.Sequential(
        #     nn.Conv2d(4 * self.n_features, 4 * self.n_features, kernel_size=2, stride=1, padding=1),
        #     nn.BatchNorm2d(4 * self.n_features),
        #     nn.LeakyReLU(),
        #     nn.MaxPool2d(kernel_size=2)
        # )

        self.classifier = None
        self.intensity_predictor = None

    def forward(self, x):
        input_channels = x.size(1)

        if self.layer1 is None or self.layer1[0].in_channels != input_channels:
            self.layer1 = nn.Sequential(
                nn.Conv2d(input_channels, self.n_features, kernel_size=3, stride=1, padding=0),
                nn.BatchNorm2d(self.n_features),
                nn.LeakyReLU(),
                nn.MaxPool2d(kernel_size=2)
            ).to(x.device)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)

        x = x.view(x.size(0), -1)

        linear_input_size = x.size(1)

        if self.classifier is None or self.classifier[0].in_features != linear_input_size:
            self.classifier = nn.Sequential(
                nn.Linear(linear_input_size, 512),
                nn.LeakyReLU(),
                nn.Linear(512, 256),
                nn.LeakyReLU(),
                nn.Linear(256, 6),
            ).to(x.device)

        # if self.intensity_predictor is None or self.intensity_predictor[0].in_features != linear_input_size:
        #     self.intensity_predictor = nn.Sequential(
        #         nn.Linear(linear_input_size, 512),
        #         nn.LeakyReLU(),
        #         nn.Linear(512, 1)
        #     ).to(x.device)

        emotion_classification_output = self.classifier(x)
        # emotion_intensity_output = self.intensity_predictor(x)
        
        return emotion_classification_output #, emotion_intensity_output

