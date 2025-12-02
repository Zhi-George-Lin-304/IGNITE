import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple


class LSTMModel(nn.Module):
    """
    A multi-layer LSTM model with a final linear projection layer.

    Args:
        input_dim (int): Dimensionality of input features per time step.
        hidden_dim (int): Number of units in each LSTM layer.
        output_dim (int): Dimensionality of the output features.
        num_layers (int, optional): Number of stacked LSTM layers. Default: 5.
        dropout (float, optional): Dropout applied between LSTM layers. Default: 0.25.

    Input Shape:
        x: (batch_size, seq_len, input_dim)

    Output:
        out: (batch_size, seq_len, output_dim)
        hidden: (h_n, c_n)
            h_n: (num_layers, batch_size, hidden_dim)
            c_n: (num_layers, batch_size, hidden_dim)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 5,
        dropout: float = 0.25
    ) -> None:
        super().__init__()

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        x: Tensor,
        hidden: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """
        Forward pass through the LSTM model.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, input_dim).
            hidden (optional tuple):
                A tuple (h0, c0) of the initial hidden and cell states.
                Each has shape (num_layers, batch_size, hidden_dim).
                If None, initializes states to zeros.

        Returns:
            out (Tensor): Output sequence of shape (batch_size, seq_len, output_dim).
            hidden (tuple): Final hidden state (h_n, c_n).
        """

        if hidden is None:
            h0 = torch.zeros(
                self.lstm.num_layers,
                x.size(0),
                self.lstm.hidden_size,
                device=x.device
            )
            c0 = torch.zeros(
                self.lstm.num_layers,
                x.size(0),
                self.lstm.hidden_size,
                device=x.device
            )
        else:
            h0, c0 = hidden

        lstm_out, hidden = self.lstm(x, (h0, c0))
        out = self.fc(lstm_out)

        return out, hidden
