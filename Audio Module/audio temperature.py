import os
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import numpy as np
from torch.nn import functional as F

# Define the ModelWithTemperature class
class ModelWithTemperature(nn.Module):
    def __init__(self, model):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    def set_temperature(self, valid_loader):
        self.cuda()
        nll_criterion = nn.CrossEntropyLoss().cuda()
        ece_criterion = _ECELoss().cuda()

        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label in valid_loader:
                input = input.cuda().unsqueeze(1)  # Add channel dimension
                logits = self.model(input)
                logits_list.append(logits)
                labels_list.append(label)
            logits = torch.cat(logits_list).cuda()
            labels = torch.cat(labels_list).cuda()

        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss
        optimizer.step(eval)

        after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
        print('Optimal temperature: %.3f' % self.temperature.item())
        print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))

        return self


class _ECELoss(nn.Module):
    def __init__(self, n_bins=15):
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

# Define the CNN model
class CNN(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=3)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=3)

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
            output = self.pool(self.relu(self.conv3(output)))
            output = self.pool(self.relu(self.conv4(output)))
            return int(np.prod(output.size()))

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Assuming you have your data loading and preprocessing functions as previously defined

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

# Initialize model, criterion, and optimizer
input_shape = (1, 13, 174)  # Assuming MFCCs with 13 coefficients and padded to 174 time steps
num_classes = len(metadata['class'].unique())
model = CNN(input_shape, num_classes)
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

# Stage 1: Initial Training
num_epochs_stage1 = 5  # Adjust this number based on your requirements

for epoch in range(num_epochs_stage1):
    train_loss = train_model(model, train_loader, criterion, optimizer, device)
    val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)
    print(f'Epoch {epoch+1}/{num_epochs_stage1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            print('Early stopping triggered')
            break

# Load the best model
model.load_state_dict(torch.load('best_model.pth'))

# Apply temperature scaling
model_with_temp = ModelWithTemperature(model)
model_with_temp.set_temperature(val_loader)

# Evaluate on test set
test_loss, test_accuracy = evaluate_model(model_with_temp, test_loader, criterion, device)
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

