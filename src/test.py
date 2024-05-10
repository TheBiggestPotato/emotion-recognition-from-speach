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


def test(model, test_loader, criterion):
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch['stft'], batch['emotion']
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs, _ = model(inputs)
            
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted_labels = torch.max(outputs, 1)
            correct_predictions += (predicted_labels == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / len(test_loader)

    accuracy = correct_predictions / total_samples
    
    print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    
    return avg_loss, accuracy


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Model()
model.to(device)

test_preprocessor = Preprocessor('config.json', mode='test')
test_loader = DataLoader(test_preprocessor, batch_size=16, shuffle=False)

criterion = nn.CrossEntropyLoss()

test_loss, test_accuracy = test(model, test_loader, criterion)

print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")