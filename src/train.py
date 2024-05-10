import json
import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss, MSELoss
from model import Model, evaluate_metrics
from data_processing import Preprocessor
from util import Utility
import numpy as np
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm 
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error
from torch.nn.utils.rnn import pad_sequence

def custom_collate_fn(batch):

    max_shape = list(batch[0]['stft'].shape)
    for item in batch:
        for dim in range(len(max_shape)):
            max_shape[dim] = max(max_shape[dim], item['stft'].shape[dim])

    padded_stft = []
    emotion_targets = []
    emotion_level_targets = []

    for item in batch:
        stft = item['stft']
        padding = [(0, max_shape[i] - stft.shape[i]) for i in range(len(max_shape))]
        stft_padded = np.pad(stft, padding, mode='constant')
        padded_stft.append(stft_padded)
        emotion_targets.append(item['emotion'])
        emotion_level_targets.append(item['emotion_level'])

    inputs = torch.tensor(np.array(padded_stft))
    emotion_targets = torch.tensor(emotion_targets)
    emotion_level_targets = torch.tensor(emotion_level_targets)
    
    return {'stft': inputs, 'emotion': emotion_targets, 'emotion_level': emotion_level_targets}



def train_model(model, train_dataset, config, device):

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=custom_collate_fn
    )

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), 1e-3)
    classification_loss_fn = torch.nn.CrossEntropyLoss()
    #regression_loss_fn = torch.nn.MSELoss()

    num_epochs = config["num_epochs"]

    progress_bar = tqdm(range(num_epochs), desc='Training')

    for epoch in progress_bar:
        total_classification_loss = 0.0
        total_regression_loss = 0.0

        all_classification_y_true = []
        all_classification_y_pred = []
        all_regression_y_true = []
        all_regression_y_pred = []

        for batch in train_loader:
            inputs = batch['stft'].float().to(device)
            emotion_targets = batch['emotion'].to(device)
            emotion_level_targets = batch['emotion_level'].float().to(device)

            #classification_output, regression_output = model(inputs)
            classification_output = model(inputs)

            classification_loss = classification_loss_fn(classification_output, emotion_targets)
            #regression_loss = regression_loss_fn(regression_output.squeeze(), emotion_level_targets)

            total_loss = classification_loss #+ regression_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            total_classification_loss += classification_loss.item()
            #total_regression_loss += regression_loss.item()

            all_classification_y_true.extend(emotion_targets.detach().cpu().numpy())
            all_classification_y_pred.extend(torch.argmax(classification_output, dim=1).detach().cpu().numpy())
            # all_regression_y_true.extend(emotion_level_targets.detach().cpu().numpy())
            # all_regression_y_pred.extend(regression_output.detach().cpu().numpy())

        avg_classification_loss = total_classification_loss / len(train_loader)
        #avg_regression_loss = total_regression_loss / len(train_loader)
        #avg_loss = (avg_classification_loss + avg_regression_loss) / 2

        classification_accuracy = accuracy_score(all_classification_y_true, all_classification_y_pred)

        #regression_mae = mean_absolute_error(all_regression_y_true, all_regression_y_pred)

        #regression_rmse = np.sqrt(mean_squared_error(all_regression_y_true, all_regression_y_pred))

        #progress_bar.set_postfix(loss=avg_loss, classification_accuracy=classification_accuracy, regression_mae=regression_mae, regression_rmse=regression_rmse)

        progress_bar.set_postfix(loss=total_classification_loss, classification_accuracy=classification_accuracy)

        #print(f'Epoch {epoch + 1}: Classification Accuracy: {classification_accuracy}, MAE: {regression_mae}, RMSE: {regression_rmse}')
        print(f'Epoch {epoch + 1}: Classification Accuracy: {classification_accuracy}')

    return model

# Usage example
config_file_path = 'config.json'

# Instantiate preprocessor
preprocessor = Preprocessor(config_file_path, mode='train')
with open(config_file_path, 'r') as f:
    config = json.load(f)


# Create the model
model = Model()

# Choose the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Train the model
trained_model = train_model(model, preprocessor, config, device)

print("Training completed.")