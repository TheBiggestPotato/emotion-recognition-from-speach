import json
import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss, MSELoss
from model import SpeechEmotionRecognitionModel
from data_processing import Preprocessor, MetadataLoader
from util import Utility
import numpy as np
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm 

def train_model(model, train_loader, num_epochs, device):
    model.to(device)
    model.train()
    
    emotion_criterion = CrossEntropyLoss()
    intensity_criterion = MSELoss()

    optimizer = Adam(model.parameters(), lr=0.001)

    progress_bar = tqdm(range(num_epochs), desc='Training')

    for epoch in progress_bar:
        running_loss = 0.0
        
        all_y_true = []
        all_y_pred = []
        
        for inputs, labels in train_loader:

            inputs = inputs.to(device)
            
            
            
            outputs = model(inputs)

            emotion_loss = emotion_criterion(stft_output, emotion_labels)
            
            intensity_levels_reshaped = intensity_levels.view(-1, envelope_output.shape[1])

            intensity_loss = intensity_criterion(envelope_output, intensity_levels_reshaped)

            loss = emotion_loss + intensity_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            y_pred = torch.argmax(stft_output, dim=1).cpu().numpy()
            all_y_pred.append(y_pred)
            all_y_true.append(emotion_labels.cpu().numpy())

        avg_loss = running_loss / len(train_loader)

        all_y_true = np.concatenate(all_y_true)
        all_y_pred = np.concatenate(all_y_pred)

        metrics = evaluate_metrics(all_y_true, all_y_pred)

        progress_bar.set_postfix(loss=avg_loss, **metrics)
        
    return model

config_file_path = 'config.json'

num_epochs = 10

preprocessor = Preprocessor(config_file_path, mode='train')

model = SpeechEmotionRecognitionModel()

train_loader = torch.utils.data.DataLoader(preprocessor, batch_size=32, collate_fn=custom_collate, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trained_model = train_model(model, train_loader, num_epochs, device)