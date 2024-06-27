import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from ECE import _ECELoss

class ModelWithHistogramBinning(nn.Module):
    """
    A thin decorator, which wraps a model with histogram binning calibration
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model, num_bins=10):
        super(ModelWithHistogramBinning, self).__init__()
        self.model = model
        self.num_bins = num_bins

    def forward(self, input):
        logits = self.model(input)
        return self.histogram_binning(logits)

    def histogram_binning(self, logits):
        """
        Perform histogram binning calibration on logits
        """
        probs = torch.softmax(logits, dim=1)
        bin_edges = torch.linspace(0, 1, self.num_bins + 1, device=logits.device)
        calibrated_probs = torch.zeros_like(probs)
        for i in range(self.num_bins):
            bin_mask = (probs >= bin_edges[i]) & (probs < bin_edges[i+1])
            bin_size = bin_mask.sum(dim=1, keepdim=True).clamp(min=1)
            calibrated_probs += bin_mask.float() * (probs.mean(dim=1, keepdim=True) / bin_size)
        return calibrated_probs

    # This function probably should live outside of this class, but whatever
    def set_histogram_binning_parameters(self, valid_loader):
        """
        Tune the histogram binning calibration parameters of the model (using the validation set).
        No parameters need to be optimized for histogram binning.
        valid_loader (DataLoader): validation set loader
        """
        pass  # No parameters need to be optimized for histogram binning

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

# Instantiate ModelWithHistogramBinning
scaled_model = ModelWithHistogramBinning(model, num_bins=15)


logits_list = []
labels_list = []
scaled_model.eval()
with torch.no_grad():
    for input, label in valid_loader:
        input = input.to(device)
        logits = scaled_model(input)
        logits_list.append(logits)
        labels_list.append(label)
logits = torch.cat(logits_list).to(device)
labels = torch.cat(labels_list).to(device)

# Instantiate ECELoss and calculate ECE
ece_criterion = _ECELoss().to(device)
ece = ece_criterion(logits, labels)
print('Expected Calibration Error (ECE):', ece.item())
