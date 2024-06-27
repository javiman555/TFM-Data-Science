import torch
import torch.nn as nn
import torch.optim as optim
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

# Define the vector scaling model
class VectorScaling(nn.Module):
    def __init__(self, input_size, output_size):
        super(VectorScaling, self).__init__()
        self.W = nn.Parameter(torch.ones(output_size))  # Initialize W as ones
        self.b = nn.Parameter(torch.zeros(output_size))  # Initialize b as zeros

    def forward(self, x):
        scaled_logits = x * self.W + self.b
        return scaled_logits

vector_scaling_model = VectorScaling(output_size, output_size)
vector_scaling_model.to(device)

# Define the optimizer and loss function
optimizer = optim.Adam(vector_scaling_model.parameters(), lr=0.01)
loss_function = nn.CrossEntropyLoss()

# Training loop to optimize W and b
num_epochs = 10
for epoch in range(num_epochs):
    vector_scaling_model.train()
    total_loss = 0.0
    for input, label in valid_loader:
        input = input.to(device)
        label = label.to(device)

        # Forward pass through the original DAN model
        logits = model(input)

        # Forward pass through the vector scaling model
        scaled_logits = vector_scaling_model(logits)

        # Calculate loss
        loss = loss_function(scaled_logits, label)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * input.size(0)

    # Calculate average loss
    average_loss = total_loss / len(valid_loader.dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {average_loss:.4f}")

# After optimization, get the optimum temperature values from the diagonal of W
optimum_temperature = vector_scaling_model.W.data.cpu().numpy()
print("Optimum Temperature (Vector Scaling):", optimum_temperature)


temperature = torch.tensor(optimum_temperature, device=device)

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
scaled_logits = logits / temperature.unsqueeze(0)  # Unsqueeze to match dimensions
scaled_probs = torch.softmax(scaled_logits, dim=1)

# Calculate the ECE with the scaled logits
ece_criterion = _ECELoss().to(device)
ece = ece_criterion(scaled_logits, labels)
print('Expected Calibration Error (ECE) with temperature scaling:', ece.item())

