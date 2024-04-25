import torch
import torch.nn as nn
import torch.optim as optim

class SpeechEmotionRecognitionModel(nn.Module):
    def __init__(self):
        super(SpeechEmotionRecognitionModel, self).__init__()

        self.conv_stft = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ).float()
        self.fc_amplitude = nn.Linear(1, 32).float()
        self.fc_envelope = nn.Linear(1, 32).float()
        self.fc_frequency = nn.Linear(1, 32).float()
        
        stft_output_shape = self.conv_stft(torch.zeros((1, 1, 32, 32))).numel()
        amplitude_output_shape = self.fc_amplitude(torch.zeros((1, 1))).numel()
        envelope_output_shape = self.fc_envelope(torch.zeros((1, 1))).numel()
        frequency_output_shape = self.fc_frequency(torch.zeros((1, 1))).numel()
        
        combined_input_size = (stft_output_shape + amplitude_output_shape +
                               envelope_output_shape + frequency_output_shape)

        self.fc_combine = nn.Sequential(
            nn.Linear(combined_input_size, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.fc_output = nn.Linear(32, 2)

        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, features):
        stft = features['stft']
        stft_output = self.conv_stft(stft).view(stft.size(0), -1)

        amplitude = features['amplitude'].view(-1, 1)
        amplitude_output = self.fc_amplitude(amplitude)
        
        envelope = features['envelope'].view(-1, 1)
        envelope_output = self.fc_envelope(envelope)
        
        frequency = features['frequency'].view(-1, 1)
        frequency_output = self.fc_frequency(frequency)

        stft_output = stft_output.view(32, -1)
        amplitude_output = amplitude_output.view(32, -1)
        envelope_output = envelope_output.view(32, -1)
        frequency_output = frequency_output.view(32, -1)

        print("stft_output shape:", stft_output.shape)
        print("amplitude_output shape:", amplitude_output.shape)
        print("envelope_output shape:", envelope_output.shape)
        print("frequency_output shape:", frequency_output.shape)
        
        combined_features = torch.cat([stft_output, amplitude_output, envelope_output, frequency_output], dim=1).T

        print(f'combined_features shape: {combined_features.shape}')

        print(self.fc_combine)
        
        combined_output = self.fc_combine(combined_features)
        print(f'combined_output shape: {combined_output.shape}')
        
        output = self.fc_output(combined_output)
        
        return output