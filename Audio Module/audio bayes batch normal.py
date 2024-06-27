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
import torchbnn as bnn  # Import Bayesian layers from torchbnn

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
    return torch.tensor(padded_mfccs, dtype=torch.float32).unsqueeze(0)  # Add channel dimension

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

# Define the Bayesian CNN model with batch normalization
class BayesianCNN(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(BayesianCNN, self).__init__()
        self.conv1 = bnn.BayesConv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1, prior_mu=0, prior_sigma=0.1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = bnn.BayesConv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, prior_mu=0, prior_sigma=0.1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.fc1_input_dim = self._get_conv_output(input_shape)
        self.fc1 = bnn.BayesLinear(in_features=self.fc1_input_dim, out_features=128, prior_mu=0, prior_sigma=0.1)
        self.fc2 = bnn.BayesLinear(in_features=128, out_features=num_classes, prior_mu=0, prior_sigma=0.1)

    def _get_conv_output(self, shape):
        with torch.no_grad():
            input = torch.zeros(1, *shape)
            output = self.conv1(input)
            output = self.pool(self.relu(self.bn1(output)))
            output = self.conv2(output)
            output = self.pool(self.relu(self.bn2(output)))
            return int(torch.flatten(output, 1).shape[1])

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))  # Apply conv1 followed by batch norm and ReLU
        x = self.pool(x)                        # Apply pooling
        x = self.relu(self.bn2(self.conv2(x)))  # Apply conv2 followed by batch norm and ReLU
        x = self.pool(x)                        # Apply pooling
        x = x.view(x.size(0), -1)               # Flatten for fully connected layers
        x = self.relu(self.fc1(x))              # Fully connected layer 1
        x = self.fc2(x)                         # Fully connected layer 2
        return x

# Initialize model, criterion, and optimizer
input_shape = (1, 13, 174)  # Assuming MFCCs with 13 coefficients and padded to 174 time steps
num_classes = len(metadata['class'].unique())
model = BayesianCNN(input_shape, num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Early stopping parameters
early_stopping_patience = 2
best_val_loss = float('inf')
patience_counter = 0

# Training and evaluation functions
def train_model(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)  # Directly use outputs for loss
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
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)  # Directly use outputs for loss
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)  # Directly use outputs for predictions
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return running_loss / len(loader.dataset), accuracy

# Stage 1: Initial Training
num_epochs_stage1 = 5
for epoch in range(num_epochs_stage1):
    train_loss = train_model(model, train_loader, criterion, optimizer, device)
    val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)
    print(f'Stage 1 - Epoch {epoch+1}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save the best model
        torch.save(model.state_dict(), 'best_model_stage1.pth')
    else:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered.")
            break

# Load the best model
model.load_state_dict(torch.load('best_model_stage1.pth'))

# Evaluate the Calibrated Model before TST
logits_list = []
labels_list = []
model.eval()
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        logits = model(inputs)
        logits_list.append(logits)
        labels_list.append(labels)

logits = torch.cat(logits_list).to(device)
labels = torch.cat(labels_list).to(device)

# Print shapes for debugging
print(f'Logits shape: {logits.shape}')
print(f'Labels shape: {labels.shape}')

# Calculate the ECE with the logits
ece_criterion = _ECELoss().to(device)
ece = ece_criterion(logits, labels)  # Directly use logits without mean

print('Expected Calibration Error (ECE) Uncalibrated:', ece.item())

# Calculate NLL (Negative Log Likelihood)
nll_criterion = nn.CrossEntropyLoss().to(device)
nll = nll_criterion(logits, labels)

print('Negative Log Likelihood (NLL) Uncalibrated:', nll.item())
