import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from ECE import _ECELoss

# Load the validation data
validation_data, validation_target = torch.load('saved_models/validation_data.pth')

# Convert data to PyTorch tensors
validation_data_tensor = validation_data
validation_target_tensor = validation_target

# Define the DataLoader for validation
validation_dataset = TensorDataset(validation_data_tensor, validation_target_tensor)
valid_loader = DataLoader(validation_dataset, batch_size=64)

# Define the Deep Averaging Network (DAN) model
class DAN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, depth=3):
        super(DAN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc_hidden = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(depth - 2)])
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        for layer in self.fc_hidden:
            x = self.relu(layer(x))
        x = self.fc_out(x)
        return x  # Note: Remove softmax from forward method

input_size = validation_data_tensor.shape[1]
hidden_size = 256
output_size = 20  # Assuming there are 20 classes
depth = 3
model = DAN(input_size, hidden_size, output_size, depth)
model.load_state_dict(torch.load('saved_models/dan_model.pth'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Scale the model with temperature scaling
temperature = 2.0  # Define the temperature scaling parameter
scaled_model = nn.Sequential(nn.Softmax(dim=1))  # Apply softmax to get probabilities
scaled_model.to(device)

# Forward pass through the original model to get logits
logits_list = []
labels_list = []
model.eval()
with torch.no_grad():
    for input, label in valid_loader:
        input = input.to(device)
        logits = model(input)
        logits_list.append(logits)
        labels_list.append(label)
logits = torch.cat(logits_list).to(device)
labels = torch.cat(labels_list).to(device)

# Apply temperature scaling to the logits
scaled_logits = logits / temperature
scaled_probs = torch.softmax(scaled_logits, dim=1)

# Calculate the ECE with the scaled logits
ece_criterion = _ECELoss().to(device)
ece = ece_criterion(scaled_logits, labels)
print('Expected Calibration Error (ECE) with temperature scaling:', ece.item())
