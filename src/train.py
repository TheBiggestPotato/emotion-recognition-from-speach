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

def evaluate_metrics(y_true, y_pred, average='weighted'):
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average=average)
    metrics['recall'] = recall_score(y_true, y_pred, average=average)
    metrics['f1_score'] = f1_score(y_true, y_pred, average=average)
    return metrics

def train_model(model, train_loader, num_epochs, device):
    model.to(device)
    model.train()
    
    # Define loss functions
    emotion_criterion = CrossEntropyLoss()
    intensity_criterion = MSELoss()

    # Define optimizer
    optimizer = Adam(model.parameters(), lr=0.001)
    
    # Progress bar for tracking progress during training
    progress_bar = tqdm(range(num_epochs), desc='Training')

    for epoch in progress_bar:
        running_loss = 0.0
        
        # Accumulating predictions and labels for metric evaluation
        all_y_true = []
        all_y_pred = []
        
        for inputs, labels in train_loader:
            # Move inputs and labels to the device
            inputs = {key: val.to(device).float() for key, val in inputs.items()}
            emotion_labels, intensity_levels = labels[:, 0].long().to(device), labels[:, 1].float().to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Assuming the model outputs a tuple of outputs (stft_output, amplitude_output, envelope_output, frequency_output)
            # You may need to adjust according to your model's actual output structure.
            stft_output, amplitude_output, envelope_output, frequency_output = outputs
            
            # Assuming emotion classification uses stft_output
            emotion_loss = emotion_criterion(stft_output, emotion_labels)
            
            # Assuming intensity prediction uses envelope_output
            # Reshape intensity_levels to match the shape of envelope_output
            intensity_levels_reshaped = intensity_levels.view(-1, envelope_output.shape[1])
            
            # Calculate intensity loss using reshaped intensity_levels
            intensity_loss = intensity_criterion(envelope_output, intensity_levels_reshaped)
            
            # Total loss
            loss = emotion_loss + intensity_loss
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Accumulate the running loss
            running_loss += loss.item()
            
            # Append predictions and labels for metrics calculation
            y_pred = torch.argmax(stft_output, dim=1).cpu().numpy()
            all_y_pred.append(y_pred)
            all_y_true.append(emotion_labels.cpu().numpy())
        
        # Calculate average loss per epoch
        avg_loss = running_loss / len(train_loader)
        
        # Combine all predictions and labels
        all_y_true = np.concatenate(all_y_true)
        all_y_pred = np.concatenate(all_y_pred)
        
        # Calculate metrics
        metrics = evaluate_metrics(all_y_true, all_y_pred)
        
        # Update progress bar with metrics and loss
        progress_bar.set_postfix(loss=avg_loss, **metrics)
        
    return model

# Custom collate function for the data loader
def custom_collate(batch):
    # Extract features and labels from the batch
    features, labels = zip(*batch)
    
    # Define feature keys (adjust this list if you have additional features)
    feature_keys = ['stft', 'amplitude', 'envelope', 'frequency']
    
    # Initialize dictionary for padded features
    padded_features = {}
    
    # Determine the maximum shape for each feature
    max_shapes = {key: [0] * len(features[0][key].shape) for key in feature_keys}
    for key in feature_keys:
        for feature in features:
            # Calculate the maximum size for each dimension
            for i, dim_size in enumerate(feature[key].shape):
                max_shapes[key][i] = max(max_shapes[key][i], dim_size)
    
    # Pad each feature tensor in the batch
    for key in feature_keys:
        padded_tensors = []
        for feature in features:
            tensor = feature[key]
            # Convert NumPy array to PyTorch tensor if necessary
            if not isinstance(tensor, torch.Tensor):
                tensor = torch.from_numpy(tensor)
            
            # Calculate padding for each dimension
            padding = []
            for i, dim_size in enumerate(tensor.shape):
                pad_before = 0
                pad_after = max_shapes[key][i] - dim_size
                padding = [(pad_before, pad_after)] + padding
            
            # Flatten the list of padding tuples
            padding = [p for pad_tuple in padding for p in pad_tuple]
            
            # Apply padding to the tensor
            padded_tensor = F.pad(tensor, padding, mode='constant', value=0)
            padded_tensors.append(padded_tensor)
        
        # Stack the padded tensors and store in padded_features
        padded_features[key] = torch.stack(padded_tensors)
    
    # Convert labels to a PyTorch tensor
    # Ensure that the target labels are the correct shape and data type
    label_tensors = torch.tensor([(Utility.emotion_label_to_number(label[0]), float(label[1])) for label in labels], dtype=torch.float32)
    
    return padded_features, label_tensors

# Function to save the model
def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)

# Load configurations
config_file_path = 'config.json'

# Create Preprocessor instance
preprocessor = Preprocessor(config_file_path, mode='train')

# Initialize model
model = SpeechEmotionRecognitionModel()

# Create DataLoader
train_loader = torch.utils.data.DataLoader(preprocessor, batch_size=32, collate_fn=custom_collate, shuffle=True)

# Define training parameters
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Train the model
trained_model = train_model(model, train_loader, num_epochs, device)

# Save the trained model
save_model(trained_model, "src/model/saved_models/model.pth")