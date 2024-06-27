import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.isotonic import IsotonicRegression
from torch.nn.functional import softmax
from ECE import _ECELoss

class ModelWithIsotonicRegression(nn.Module):
    """
    A thin decorator, which wraps a model with isotonic regression calibration
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model, device):
        super(ModelWithIsotonicRegression, self).__init__()
        self.model = model
        self.ir_model = IsotonicRegression(out_of_bounds='clip')
        self.device = device

    def forward(self, input):
        logits = self.model(input)
        return self.isotonic_regression(logits)

    def isotonic_regression(self, logits):
        """
        Perform isotonic regression calibration on logits
        """
        probs = softmax(logits, dim=1)
        calibrated_probs = torch.tensor(self.ir_model.transform(probs.cpu().numpy()), dtype=torch.float32, device=self.device)
        return calibrated_probs

    def fit_isotonic_regression(self, valid_loader):
        """
        Fit the isotonic regression model to calibrate the probabilities of the model
        valid_loader (DataLoader): validation set loader
        """
        # Collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label in valid_loader:
                input = input.to(self.device)
                logits = self.model(input)
                logits_list.append(logits)
                labels_list.append(label)
    
        # Concatenate logits and labels
        logits = torch.cat(logits_list).to(self.device)
        labels = torch.cat(labels_list).to(self.device)
    
        # Convert logits to probabilities using softmax
        probs = softmax(logits, dim=1)
    
        # Flatten the probabilities array and labels
        flattened_probs = probs.view(-1).cpu().numpy()
        flattened_labels = labels.cpu().numpy()
    
        # Fit isotonic regression model to the flattened probabilities
        self.ir_model.fit(flattened_probs, flattened_labels)
    
        return self


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

# Load the DAN model
model = DAN(input_size, hidden_size, output_size, depth)
model.load_state_dict(torch.load('saved_models/dan_model.pth'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Instantiate ModelWithIsotonicRegression
scaled_model = ModelWithIsotonicRegression(model, device)

# Fit isotonic regression to calibrate probabilities
scaled_model.fit_isotonic_regression(valid_loader)

# Forward pass through the model to get calibrated probabilities
calibrated_probs_list = []
labels_list = []
scaled_model.eval()
with torch.no_grad():
    for input, label in valid_loader:
        input = input.to(device)
        calibrated_probs = scaled_model(input)
        calibrated_probs_list.append(calibrated_probs)
        labels_list.append(label)
calibrated_probs = torch.cat(calibrated_probs_list).to(device)
labels = torch.cat(labels_list).to(device)

# Instantiate ECELoss and calculate ECE
ece_criterion = _ECELoss().to(device)
ece = ece_criterion(calibrated_probs, labels)
print('Expected Calibration Error (ECE) after isotonic regression:', ece.item())
