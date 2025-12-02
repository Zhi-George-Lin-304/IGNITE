import torch
import torch.nn as nn
import torch.nn.functional as F

from lstm_gcn_rl.dataset import MolecularDataset

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=5, dropout=0.25):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True)

        # Fully connected output layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden=None):
        if x.dim() == 3:  # batch mode: (batch, seq_len, input_dim)
            batch_size = x.size(0)
            seq_len = x.size(1)
        elif x.dim() == 2:  # no batch: (seq_len, input_dim)
            batch_size = 1
            seq_len = x.size(0)
            x = x.unsqueeze(0)  # add batch dim: (1, seq_len, input_dim)

        
        if hidden is None:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        else:
            h0, c0 = hidden

        # Forward propagate LSTM
        out, hidden = self.lstm(x, (h0, c0))

        # Apply the fully connected layer
        out = self.fc(out)

        return out, hidden

class GCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)  # No bias in the linear layer
        self.bias = nn.Parameter(torch.zeros(1, output_dim))  # Trainable bias matrix
        self.batch_norm = nn.BatchNorm1d(output_dim)  # Batch normalization
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)  # LeakyReLU activation

    def forward(self, A_normalized, H):
        HW = self.linear(H)
        HW_plus_B = HW + self.bias

        # Reshape for batch normalization
        batch_size, num_nodes, num_features = HW_plus_B.shape
        HW_plus_B = HW_plus_B.view(-1, num_features)  # Flatten to (batch_size * num_nodes, num_features)
        HW_plus_B = self.batch_norm(HW_plus_B)  # Apply batch normalization
        HW_plus_B = HW_plus_B.view(batch_size, num_nodes, num_features)  # Reshape back

        output = torch.matmul(A_normalized, HW_plus_B)
        return self.leaky_relu(output)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        """
        Args:
            input_dim: int
            hidden_dims: list of int
            output_dim: int
        """
        super(MLP, self).__init__()
        dims = [input_dim] + hidden_dims + [output_dim]
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        """
        Args:
            input_dim: int, number of input atom features
            hidden_dims: list of int, output dimensions for each GCN layer
        """
        super(GCN, self).__init__()
        dims = [input_dim] + hidden_dims
        self.gcn_layers = nn.ModuleList([
            GCNLayer(dims[i], dims[i+1]) for i in range(len(dims) - 1)
        ])

    def forward(self, A_normalized, X, num_atoms):
        H = X
        for layer in self.gcn_layers:
            H = layer(A_normalized, H)

        # Mask invalid atoms (padding)
        mask = torch.arange(H.size(1), device=H.device).expand(H.size(0), -1) < num_atoms.unsqueeze(1)
        H_masked = H * mask.unsqueeze(2)
        embeddings = H_masked.sum(dim=1) / num_atoms.unsqueeze(1).float()
        return embeddings

