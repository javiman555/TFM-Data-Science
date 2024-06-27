import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from ECE import _ECELoss

class ModelWithMatrixScaling(nn.Module):
    """
    A thin decorator, which wraps a model with matrix scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model, input_size, output_size):
        super(ModelWithMatrixScaling, self).__init__()
        self.model = model
        self.W = nn.Parameter(torch.randn(output_size, input_size))  # Matrix scaling parameters
        self.b = nn.Parameter(torch.randn(output_size))  # Matrix scaling bias

    def forward(self, input):
        logits = self.model(input)
        return self.matrix_scale(logits)

    def matrix_scale(self, logits):
        """
        Perform matrix scaling on logits
        """
        scaled_logits = torch.matmul(logits, self.W.t()) + self.b.unsqueeze(0)
        return scaled_logits

    # This function probably should live outside of this class, but whatever
    def set_matrix_scaling_parameters(self, valid_loader):
        """
        Tune the matrix scaling parameters of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        self.cuda()
        nll_criterion = nn.CrossEntropyLoss().cuda()
        ece_criterion = _ECELoss().cuda()

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label in valid_loader:
                input = input.cuda()
                logits = self.model(input)
                logits_list.append(logits)
                labels_list.append(label)
            logits = torch.cat(logits_list).cuda()
            labels = torch.cat(labels_list).cuda()

        # Calculate NLL and ECE before matrix scaling
        before_matrix_scaling_nll = nll_criterion(logits, labels).item()
        before_matrix_scaling_ece = ece_criterion(logits, labels).item()
        print('Before matrix scaling - NLL: %.3f, ECE: %.3f' % (before_matrix_scaling_nll, before_matrix_scaling_ece))

        # Next: optimize the matrix scaling parameters w.r.t. NLL
        optimizer = optim.LBFGS([self.W, self.b], lr=0.01, max_iter=50)

        def eval():
            optimizer.zero_grad()
            scaled_logits = self.matrix_scale(logits)
            loss = nll_criterion(scaled_logits, labels)
            loss.backward()
            return loss
        optimizer.step(eval)

        # Calculate NLL and ECE after matrix scaling
        after_matrix_scaling_nll = nll_criterion(self.matrix_scale(logits), labels).item()
        after_matrix_scaling_ece = ece_criterion(self.matrix_scale(logits), labels).item()
        print('Optimal matrix scaling parameters:')
        print('W:', self.W)
        print('b:', self.b)
        print('After matrix scaling - NLL: %.3f, ECE: %.3f' % (after_matrix_scaling_nll, after_matrix_scaling_ece))

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

# Instantiate ModelWithMatrixScaling
scaled_model = ModelWithMatrixScaling(model, input_size, output_size)
scaled_model.set_matrix_scaling_parameters(valid_loader)

# Forward pass through the model to get logits
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
