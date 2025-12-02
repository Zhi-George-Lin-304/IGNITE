import torch
import torch.nn as nn
from torch import Tensor
from typing import List


class GCNLayer(nn.Module):
    """
    A single Graph Convolutional Network (GCN) layer with:
        - Linear transform (no bias)
        - Trainable bias vector
        - BatchNorm
        - LeakyReLU activation
        - A_normalized @ H propagation

    Args:
        input_dim (int): Number of input features per node.
        output_dim (int): Number of output features per node.

    Shape:
        A_normalized: (B, N, N)   where B=batch size, N=max_atoms
        H:             (B, N, input_dim)
        Output:        (B, N, output_dim)
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(1, output_dim))
        self.batch_norm = nn.BatchNorm1d(output_dim)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, A_normalized: Tensor, H: Tensor) -> Tensor:
        """
        Forward pass of the GCN layer.

        Args:
            A_normalized (Tensor): Normalized adjacency matrix, shape (B, N, N).
            H (Tensor): Node feature matrix, shape (B, N, input_dim).

        Returns:
            Tensor: Updated node features, shape (B, N, output_dim).
        """
        HW = self.linear(H)
        HW_plus_B = HW + self.bias  # broadcast bias across nodes

        # BatchNorm expects shape (batch*N, features)
        batch_size, num_nodes, num_features = HW_plus_B.shape
        HW_bn = HW_plus_B.view(-1, num_features)
        HW_bn = self.batch_norm(HW_bn)
        HW_bn = HW_bn.view(batch_size, num_nodes, num_features)

        output = torch.matmul(A_normalized, HW_bn)
        return self.leaky_relu(output)



class MLP(nn.Module):
    """
    A simple fully connected feed-forward network (MLP) with ReLU activations.

    Args:
        input_dim (int): Input feature dimension.
        hidden_dims (list[int]): Hidden layer sizes.
        output_dim (int): Output feature dimension.

    Shape:
        x: (B, input_dim)
        Output: (B, output_dim)
    """

    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int) -> None:
        super(MLP, self).__init__()

        dims = [input_dim] + hidden_dims + [output_dim]
        layers = []

        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.network = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (B, input_dim)

        Returns:
            Tensor: Output tensor of shape (B, output_dim)
        """
        return self.network(x)



class GCN(nn.Module):
    """
    Multi-layer GCN encoder that produces fixed-size graph embeddings
    by averaging node embeddings over valid atoms.

    Args:
        input_dim (int): Number of atom features.
        hidden_dims (list[int]): Output feature size of each GCN layer.

    Shape:
        A_normalized: (B, N, N)
        X:            (B, N, input_dim)
        num_atoms:    (B,) number of valid atoms per molecule

    Returns:
        embeddings: (B, hidden_dims[-1])
    """

    def __init__(self, input_dim: int, hidden_dims: List[int]) -> None:
        super(GCN, self).__init__()

        dims = [input_dim] + hidden_dims
        self.gcn_layers = nn.ModuleList([
            GCNLayer(dims[i], dims[i+1]) for i in range(len(dims) - 1)
        ])

    def forward(self, A_normalized: Tensor, X: Tensor, num_atoms: Tensor) -> Tensor:
        """
        Args:
            A_normalized (Tensor): Normalized adjacency matrices (B, N, N)
            X (Tensor): Padded atom feature matrices (B, N, input_dim)
            num_atoms (Tensor): Valid atom count for each graph (B,)

        Returns:
            Tensor: Graph-level embeddings of shape (B, hidden_dims[-1])
        """
        H = X
        for layer in self.gcn_layers:
            H = layer(A_normalized, H)

        # Mask padded atoms
        batch_size, num_nodes, _ = H.shape
        mask = torch.arange(num_nodes, device=H.device).expand(batch_size, -1) < num_atoms.unsqueeze(1)
        H_masked = H * mask.unsqueeze(2)

        # Mean pooling over valid atoms
        embeddings = H_masked.sum(dim=1) / num_atoms.unsqueeze(1).float()
        return embeddings
