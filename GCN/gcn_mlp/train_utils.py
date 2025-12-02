import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from typing import List, Tuple, Literal

from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader as PyGDataLoader


def _get_loss_function(name: str) -> nn.Module:
    """
    Return a PyTorch loss function based on its name.

    Args:
        name (str): One of {"mse", "mae", "huber"} (case-insensitive).

    Returns:
        nn.Module: Instantiated loss function.

    Raises:
        ValueError: If the name is not supported.
    """
    name = name.lower()
    if name == "mse":
        return nn.MSELoss()
    elif name == "mae":
        return nn.L1Loss()
    elif name == "huber":
        return nn.SmoothL1Loss()
    else:
        raise ValueError(f"Unsupported loss function: {name}")


def train_and_validate(
    gcn_model: nn.Module,
    mlp_model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: Optimizer,
    scheduler: _LRScheduler,
    epochs: int,
    device: torch.device,
    loss_fn_name: Literal["mse", "mae", "huber"] = "mse",
) -> Tuple[List[float], List[float]]:
    """
    Train and validate a GCN + MLP model using a standard PyTorch DataLoader.

    Each batch from the loaders is expected to be:
        (A, X, y, num_atoms)
        A:         (B, N, N)   normalized adjacency
        X:         (B, N, F)   node features
        y:         (B,)        targets
        num_atoms: (B,)        valid atom counts

    Args:
        gcn_model (nn.Module): GCN encoder.
        mlp_model (nn.Module): MLP readout/regressor.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        optimizer (Optimizer): Optimizer instance.
        scheduler (_LRScheduler): Learning rate scheduler (e.g. ReduceLROnPlateau).
        epochs (int): Number of training epochs.
        device (torch.device): Device on which to run training.
        loss_fn_name ({"mse","mae","huber"}): Loss function to use.

    Returns:
        Tuple[List[float], List[float]]:
            - train_loss_list: average training loss per epoch
            - val_loss_list: average validation loss per epoch
    """
    loss_fn = _get_loss_function(loss_fn_name)

    train_loss_list: List[float] = []
    val_loss_list: List[float] = []

    for epoch in range(epochs):
        gcn_model.train()
        mlp_model.train()
        train_loss = 0.0

        for A, X, y, num_atoms in train_loader:
            A, X, y, num_atoms = A.to(device), X.to(device), y.to(device), num_atoms.to(device)
            optimizer.zero_grad()

            embeddings: Tensor = gcn_model(A, X, num_atoms)
            y_pred: Tensor = mlp_model(embeddings)

            loss: Tensor = loss_fn(y_pred.squeeze(), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss_avg = train_loss / len(train_loader)
        train_loss_list.append(train_loss_avg)

        # Validation
        gcn_model.eval()
        mlp_model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for A, X, y, num_atoms in val_loader:
                A, X, y, num_atoms = A.to(device), X.to(device), y.to(device), num_atoms.to(device)
                embeddings = gcn_model(A, X, num_atoms)
                y_pred = mlp_model(embeddings)

                loss = loss_fn(y_pred.squeeze(), y)
                val_loss += loss.item()

        val_loss_avg = val_loss / len(val_loader)
        val_loss_list.append(val_loss_avg)
        scheduler.step(val_loss_avg)

        print(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"Train Loss: {train_loss_avg:.4f} | Val Loss: {val_loss_avg:.4f}"
        )

    return train_loss_list, val_loss_list


def train_and_validate_qm9(
    gcn_model: nn.Module,
    mlp_model: nn.Module,
    train_loader: PyGDataLoader,
    val_loader: PyGDataLoader,
    optimizer: Optimizer,
    scheduler: _LRScheduler,
    epochs: int,
    device: torch.device,
    loss_fn_name: Literal["mse", "mae", "huber"] = "mse",
) -> Tuple[List[float], List[float]]:
    """
    Train and validate a GCN + MLP model using a PyTorch Geometric QM9-style DataLoader.

    Each batch from the loaders is expected to be a PyG `Data` object with:
        batch.x         : node features
        batch.edge_index: edge indices
        batch.batch     : graph assignment vector for each node
        batch.y         : graph-level targets

    Args:
        gcn_model (nn.Module): PyG-compatible GCN encoder.
        mlp_model (nn.Module): MLP readout/regressor.
        train_loader (PyGDataLoader): Training data loader (PyG).
        val_loader (PyGDataLoader): Validation data loader (PyG).
        optimizer (Optimizer): Optimizer instance.
        scheduler (_LRScheduler): Learning rate scheduler.
        epochs (int): Number of training epochs.
        device (torch.device): Device on which to run training.
        loss_fn_name ({"mse","mae","huber"}): Loss function to use.

    Returns:
        Tuple[List[float], List[float]]:
            - train_loss_list: average training loss per epoch
            - val_loss_list: average validation loss per epoch
    """
    loss_fn = _get_loss_function(loss_fn_name)

    train_loss_list: List[float] = []
    val_loss_list: List[float] = []

    for epoch in range(epochs):
        gcn_model.train()
        mlp_model.train()
        train_loss = 0.0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            node_embeddings: Tensor = gcn_model(batch.x, batch.edge_index, batch.batch)
            graph_embeddings: Tensor = global_mean_pool(node_embeddings, batch.batch)

            y_pred: Tensor = mlp_model(graph_embeddings)
            loss: Tensor = loss_fn(y_pred.squeeze(), batch.y.squeeze())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss_avg = train_loss / len(train_loader)
        train_loss_list.append(train_loss_avg)

        # Validation
        gcn_model.eval()
        mlp_model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                node_embeddings = gcn_model(batch.x, batch.edge_index, batch.batch)
                graph_embeddings = global_mean_pool(node_embeddings, batch.batch)
                y_pred = mlp_model(graph_embeddings)

                loss = loss_fn(y_pred.squeeze(), batch.y.squeeze())
                val_loss += loss.item()

        val_loss_avg = val_loss / len(val_loader)
        val_loss_list.append(val_loss_avg)
        scheduler.step(val_loss_avg)

        print(
            f"[Epoch {epoch+1:03d}] "
            f"Train Loss: {train_loss_avg:.4f} | Val Loss: {val_loss_avg:.4f}"
        )

    return train_loss_list, val_loss_list
