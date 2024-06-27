import os
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import numpy as np
from ECE import _ECELoss  # Assuming you have an implementation of ECE loss
from temperature_scaling.temperature_scaling import ModelWithTemperature

# Load metadata
metadata_file = 'UrbanSound8K/metadata/UrbanSound8K.csv'
audio_dir = 'UrbanSound8K/audio'
metadata = pd.read_csv(metadata_file)

# Helper function to load and preprocess audio data
def load_audio_file(file_path, sample_rate=22050):
    signal, sr = librosa.load(file_path, sr=sample_rate)
    return signal, sr

def pad_sequence(seq, max_length):
    if len(seq) > max_length:
        return seq[:max_length]
    else:
        return np.pad(seq, ((0, 0), (0, max_length - seq.shape[1])), 'constant')

def extract_features(signal, sr, n_mfcc=13, max_length=174, n_fft=512):
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft)
    padded_mfccs = pad_sequence(mfccs, max_length)
    return torch.tensor(padded_mfccs, dtype=torch.float32)

class UrbanSoundDataset(Dataset):
    def __init__(self, metadata, audio_dir, transform=None, max_length=174):
        self.metadata = metadata
        self.audio_dir = audio_dir
        self.transform = transform
        self.max_length = max_length

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        audio_file_path = os.path.join(self.audio_dir, 'fold' + str(self.metadata.iloc[idx, 5]), self.metadata.iloc[idx, 0])
        signal, sr = load_audio_file(audio_file_path)
        label = self.metadata.iloc[idx, 6]
        
        if self.transform:
            signal = self.transform(signal, sr, max_length=self.max_length)
            
        return signal, label

# Create datasets and dataloaders with consistent tensor sizes
train_metadata, test_metadata = train_test_split(metadata, test_size=0.2, stratify=metadata['class'])
train_metadata, val_metadata = train_test_split(train_metadata, test_size=0.2, stratify=train_metadata['class'])

train_dataset = UrbanSoundDataset(train_metadata, audio_dir, transform=extract_features)
val_dataset = UrbanSoundDataset(val_metadata, audio_dir, transform=extract_features)
test_dataset = UrbanSoundDataset(test_metadata, audio_dir, transform=extract_features)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the CNN model
class CNN(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.relu = nn.ReLU()
        self.fc1_input_dim = self._get_conv_output(input_shape)
        self.fc1 = nn.Linear(self.fc1_input_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def _get_conv_output(self, shape):
        with torch.no_grad():
            input = torch.zeros(1, *shape)
            output = self.pool(self.relu(self.conv1(input)))
            output = self.pool(self.relu(self.conv2(output)))
            return int(np.prod(output.size()))

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize model, criterion, and optimizer
input_shape = (1, 13, 174)  # Assuming MFCCs with 13 coefficients and padded to 174 time steps
num_classes = len(metadata['class'].unique())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in loader:
        inputs = inputs.to(device).unsqueeze(1)  # Add channel dimension
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(loader.dataset)

def evaluate_model(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device).unsqueeze(1)  # Add channel dimension
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return running_loss / len(loader.dataset), accuracy

# Training multiple models
num_models = 5
models = []
best_model_paths = []

for i in range(num_models):
    print(f"Training model {i+1}/{num_models}")
    model = CNN(input_shape, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    early_stopping_patience = 2
    best_val_loss = float('inf')
    patience_counter = 0

    num_epochs_stage1 = 5
    for epoch in range(num_epochs_stage1):
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)
        print(f'Stage 1 - Epoch {epoch+1}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_path = f'best_model_{i+1}.pth'
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break
    
    best_model_paths.append(best_model_path)
    models.append(model)

# Load the best models
for i, model in enumerate(models):
    model.load_state_dict(torch.load(best_model_paths[i]))

# Evaluate and average predictions
logits_list = []
labels_list = []

for model in models:
    model.eval()
    with torch.no_grad():
        model_logits = []
        model_labels = []
        for inputs, labels in val_loader:
            inputs = inputs.to(device).unsqueeze(1)  # Add channel dimension
            logits = model(inputs)
            model_logits.append(logits)
            if len(labels_list) == 0:
                labels_list.append(labels.to(device))
        logits_list.append(torch.cat(model_logits))

# Check the shapes to ensure they match
logits_avg = torch.mean(torch.stack(logits_list), dim=0)
labels = torch.cat(labels_list).to(device)

print(f'Logits shape: {logits_avg.shape}')
print(f'Labels shape: {labels.shape}')

# Calculate the ECE with the averaged logits
ece_criterion = _ECELoss().to(device)
ece = ece_criterion(logits_avg, labels)
print('Expected Calibration Error (ECE) Averaged:', ece.item())
