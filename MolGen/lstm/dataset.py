import numpy as np
import torch
from torch import Tensor
from typing import List, Tuple, Dict
from sklearn.utils import shuffle
from torch.utils.data import TensorDataset, DataLoader

from lstm.tokenizer import tokenize_smiles  


def gen_data(
    smiles_list: np.ndarray,
    char_to_int: Dict[str, int],
    max_len: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a list of SMILES into one-hot encoded X (inputs) and Y (targets).

    Args:
        smiles_list (np.ndarray): Array of SMILES strings.
        char_to_int (dict): Mapping from token → integer index.
        max_len (int): Maximum SMILES length for padding.

    Returns:
        X (np.ndarray): One-hot input tensor of shape (N, max_len, vocab_size)
        Y (np.ndarray): One-hot target tensor of shape (N, max_len, vocab_size)
    """

    # ---- Ignore header row "SMILES" ----
    if len(smiles_list) > 0 and smiles_list[0].strip().upper() == "SMILES":
        smiles_list = smiles_list[1:]

    vocab_size = len(char_to_int)

    X = np.zeros((len(smiles_list), max_len, vocab_size), dtype=np.float32)
    Y = np.zeros((len(smiles_list), max_len, vocab_size), dtype=np.float32)

    for i, smiles in enumerate(smiles_list):
        tokens: List[str] = tokenize_smiles(smiles)
        tokens.append("E")  # End token

        for t in range(min(len(tokens), max_len)):
            tok = tokens[t]
            if tok in char_to_int:         # safety check
                X[i, t, char_to_int[tok]] = 1.0

            # Next-token prediction
            if t + 1 < len(tokens) and tokens[t + 1] in char_to_int:
                Y[i, t, char_to_int[tokens[t + 1]]] = 1.0

    return X, Y


def get_dataloaders(
    train_file: str,
    val_file: str,
    char_to_int: Dict[str, int],
    max_len: int,
    batch_size: int,
    device: torch.device
) -> Tuple[DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for training and validation.

    Args:
        train_file (str): Path to training SMILES file.
        val_file (str): Path to validation SMILES file.
        char_to_int (dict): Vocabulary mapping token → index.
        max_len (int): Sequence padding length.
        batch_size (int): Batch size for loaders.
        device (torch.device): Training device (CPU/GPU).

    Returns:
        train_loader (DataLoader)
        val_loader (DataLoader)
    """
    train_data = np.genfromtxt(train_file, dtype='U')
    val_data = np.genfromtxt(val_file, dtype='U')

    # Generate one-hot encoded datasets
    X_train, Y_train = gen_data(train_data, char_to_int, max_len)
    X_val, Y_val = gen_data(val_data, char_to_int, max_len)

    # Shuffle training data
    X_train, Y_train = shuffle(X_train, Y_train)

    # Convert to PyTorch tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    Y_val_t = torch.tensor(Y_val, dtype=torch.float32).to(device)

    train_loader = DataLoader(
        TensorDataset(X_train_t, Y_train_t),
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        TensorDataset(X_val_t, Y_val_t),
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader
