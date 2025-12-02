import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils import shuffle
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple

from lstm.dataset import gen_data
from lstm.model import LSTMModel
from lstm.tokenizer import build_vocab


def get_args() -> argparse.Namespace:
    """
    Parse command-line arguments for training or generating SMILES using LSTM.

    Returns:
        argparse.Namespace containing:
            train_path (str): Path to training SMILES file.
            val_path (str): Path to validation SMILES file.
            tokenizer_vocab_path (str): Vocabulary-building SMILES file.
            model_path (str): Where to save/load model checkpoint.
            hidden_dim (int): LSTM hidden dimension.
            num_layers (int): Number of stacked LSTM layers.
            dropout (float): Dropout between layers.
            num_epochs (int): Max training epochs.
            batch_size (int): Training batch size.
            learning_rate (float): Learning rate.
            patience (int): Early stopping patience.
            start_token (str): For generation mode.
            max_length (int): Max generated sequence length.
            max_attempts (int): Max generation retries.
    """
    parser = argparse.ArgumentParser(description="Train or generate SMILES with LSTM")

    # Data and vocabulary
    parser.add_argument("--train_path", type=str, default="data/train.txt")
    parser.add_argument("--val_path", type=str, default="data/val.txt")
    parser.add_argument("--tokenizer_vocab_path", type=str, default="data/smiles.csv")
    parser.add_argument("--model_path", type=str, default="checkpoints/best_model.pt")

    # Model hyperparameters
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.25)

    # Training settings
    parser.add_argument("--num_epochs", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=10)

    # Generation arguments
    parser.add_argument("--start_token", type=str, default="C")
    parser.add_argument("--max_length", type=int, default=100)
    parser.add_argument("--max_attempts", type=int, default=10)

    return parser.parse_args()


def main() -> None:
    """
    Main training loop for LSTM SMILES model.

    Loads dataset → builds vocabulary → converts to one-hot →
    trains LSTM with CrossEntropyLoss and early stopping.
    """
    args = get_args()
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load SMILES
    train_data: np.ndarray = np.genfromtxt(args.train_path, dtype='U')
    val_data: np.ndarray = np.genfromtxt(args.val_path, dtype='U')

    # Build vocabulary from file
    char_to_int, int_to_char = build_vocab(args.tokenizer_vocab_path)

    # Determine sequence length
    embed: int = max(len(sm) for sm in np.concatenate((train_data, val_data)))

    # Convert to one-hot
    X_train, Y_train = gen_data(train_data, char_to_int, embed)
    X_val, Y_val = gen_data(val_data, char_to_int, embed)

    # Shuffle training set
    X_train, Y_train = shuffle(X_train, Y_train)

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    Y_val_tensor = torch.tensor(Y_val, dtype=torch.float32).to(device)

    # Data loaders
    train_loader: DataLoader = DataLoader(
        TensorDataset(X_train_tensor, Y_train_tensor),
        batch_size=args.batch_size,
        shuffle=True
    )
    val_loader: DataLoader = DataLoader(
        TensorDataset(X_val_tensor, Y_val_tensor),
        batch_size=args.batch_size
    )

    # Model setup
    input_dim: int = X_train.shape[2]
    output_dim: int = Y_train.shape[2]

    model: LSTMModel = LSTMModel(
        input_dim,
        args.hidden_dim,
        output_dim,
        args.num_layers,
        args.dropout
    ).to(device)

    criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
    optimizer: optim.Adam = optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val_loss: float = float("inf")
    epochs_without_improvement: int = 0

    print("Training started...")

    for epoch in range(args.num_epochs):
        model.train()
        total_train_loss: float = 0.0

        for batch_X, batch_Y in train_loader:
            optimizer.zero_grad()
            output, _ = model(batch_X)
            loss: Tensor = criterion(
                output.reshape(-1, output_dim),
                batch_Y.argmax(dim=-1).reshape(-1)
            )
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        # Validation
        model.eval()
        total_val_loss: float = 0.0

        with torch.no_grad():
            for batch_X, batch_Y in val_loader:
                output, _ = model(batch_X)
                val_loss = criterion(
                    output.reshape(-1, output_dim),
                    batch_Y.argmax(dim=-1).reshape(-1)
                )
                total_val_loss += val_loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)

        print(
            f"Epoch [{epoch+1}/{args.num_epochs}] "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
        )

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), args.model_path)
            print("Validation improved → model saved.")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epoch(s).")

        if epochs_without_improvement >= args.patience:
            print("Early stopping.")
            break

    print("Training complete. Best model saved.")


if __name__ == "__main__":
    main()
