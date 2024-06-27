from sklearn.datasets import fetch_20newsgroups_vectorized
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import os
from ECE import _ECELoss  # Assuming you have an implementation of ECE loss
import torchbnn as bnn

# Fetch the dataset
data = fetch_20newsgroups_vectorized(subset='all')

# Define the document counts for train/validation/test
train_docs = 9034
validation_docs = 2259
test_docs = 7528

# Split the dataset into train/validation/test sets
# First, split into train and remaining data
train_data, remaining_data, train_target, remaining_target = train_test_split(data.data, data.target, train_size=train_docs)

# Then, split remaining data into validation and test sets
validation_data, test_data, validation_target, test_target = train_test_split(remaining_data, remaining_target, test_size=test_docs, stratify=remaining_target)

# Convert data to PyTorch tensors
train_data_tensor = torch.tensor(train_data.toarray(), dtype=torch.float32)
train_target_tensor = torch.tensor(train_target, dtype=torch.int64)
validation_data_tensor = torch.tensor(validation_data.toarray(), dtype=torch.float32)
validation_target_tensor = torch.tensor(validation_target, dtype=torch.int64)
test_data_tensor = torch.tensor(test_data.toarray(), dtype=torch.float32)
test_target_tensor = torch.tensor(test_target, dtype=torch.int64)

# Define the directory to save data and models
save_dir = 'saved_models'
os.makedirs(save_dir, exist_ok=True)

# Save validation data
torch.save((validation_data_tensor, validation_target_tensor), os.path.join(save_dir, 'validation_data.pth'))

# Define the Bayesian Deep Averaging Network (Bayesian DAN) model
class BayesianDAN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, depth=3):
        super(BayesianDAN, self).__init__()
        self.fc1 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=input_size, out_features=hidden_size)
        self.relu = nn.ReLU()
        self.fc_hidden = nn.ModuleList([bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=hidden_size, out_features=hidden_size) for _ in range(depth - 2)])
        self.fc_out = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=hidden_size, out_features=output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        for layer in self.fc_hidden:
            x = self.relu(layer(x))
        x = self.fc_out(x)
        return x

# Define helper functions for training and evaluation
def train_model(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
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
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
    return running_loss / len(loader.dataset)

# Define hyperparameters
input_size = train_data_tensor.shape[1]
hidden_size = 256
output_size = len(data.target_names)
depth = 3
batch_size = 64
learning_rate = 0.001
num_epochs = 10

# Create DataLoader for training and validation
train_dataset = TensorDataset(train_data_tensor, train_target_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_dataset = TensorDataset(validation_data_tensor, validation_target_tensor)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size)

# Initialize model, loss function, and optimizer
model = BayesianDAN(input_size, hidden_size, output_size, depth)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    train_loss = train_model(model, train_loader, criterion, optimizer, device)
    val_loss = evaluate_model(model, validation_loader, criterion, device)
    print(f'Epoch {epoch+1}, Validation Loss: {val_loss:.4f}')

# Evaluate the model
logits_list = []
labels_list = []
model.eval()
with torch.no_grad():
    for inputs, labels in validation_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        logits = model(inputs)
        logits_list.append(logits)
        labels_list.append(labels)
logits = torch.cat(logits_list).to(device)
labels = torch.cat(labels_list).to(device)

# Calculate the ECE with the logits
ece_criterion = _ECELoss().to(device)
ece = ece_criterion(logits, labels)
print('Expected Calibration Error (ECE):', ece.item())

# Save trained model
torch.save(model.state_dict(), os.path.join(save_dir, 'bayesian_dan_model.pth'))

# Test the model
test_dataset = TensorDataset(test_data_tensor, test_target_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
test_accuracy = correct / total
print(f'Test Accuracy: {test_accuracy:.4f}')
