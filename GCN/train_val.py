import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from typing import List

from gcn_mlp.model import GCN, MLP
from gcn_mlp.dataset import MolecularDataset
from gcn_mlp.train_utils import train_and_validate


def main() -> None:
    """
    Command-line entry point for training a GCN + MLP model on molecular data.

    Parses CLI arguments, constructs datasets/dataloaders, initializes the
    GCN and MLP models, and trains them jointly. After training, the model
    weights and per-epoch loss curves are saved to disk.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Train a GCN + MLP model on molecular data."
    )
    parser.add_argument('--input_dim', type=int, default=42,
                        help="Dimensionality of atom feature vectors.")
    parser.add_argument('--hidden_dim_gcn', type=int, default=100,
                        help="Hidden dimension of the GCN output layer.")
    parser.add_argument('--hidden_dim_mlp', type=int, default=300,
                        help="Hidden dimension of the MLP.")
    parser.add_argument('--output_dim', type=int, default=1,
                        help="Output dimension (e.g., number of target properties).")
    parser.add_argument('--batch_size', type=int, default=30,
                        help="Batch size for training and validation.")
    parser.add_argument('--epochs', type=int, default=100,
                        help="Number of training epochs.")
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help="Learning rate for Adam optimizer.")
    parser.add_argument('--loss_fn', type=str, default='mse',
                        choices=['mse', 'mae', 'huber'],
                        help="Loss function to use.")
    parser.add_argument('--scheduler_patience', type=int, default=10,
                        help="Patience for ReduceLROnPlateau scheduler.")
    parser.add_argument('--scheduler_factor', type=float, default=0.1,
                        help="Factor by which the learning rate is reduced.")
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help="Training device.")
    parser.add_argument('--model_path', type=str, default='gcn_mlp.pth',
                        help="Path to save trained model weights.")

    args = parser.parse_args()
    device: torch.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Dataset
    train_dataset: MolecularDataset = MolecularDataset("data/train.txt", target_col="ST_split")
    val_dataset: MolecularDataset = MolecularDataset("data/val.txt", target_col="ST_split")

    train_loader: DataLoader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader: DataLoader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Models
    gcn_model: nn.Module = GCN(args.input_dim, [args.hidden_dim_gcn]).to(device)
    mlp_model: nn.Module = MLP(args.hidden_dim_gcn, [args.hidden_dim_mlp], args.output_dim).to(device)

    optimizer: optim.Optimizer = optim.Adam(
        list(gcn_model.parameters()) + list(mlp_model.parameters()),
        lr=args.learning_rate
    )
    scheduler: optim.lr_scheduler.ReduceLROnPlateau = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        verbose=True
    )

    # Train and validate
    train_mse, val_mse = train_and_validate(
        gcn_model,
        mlp_model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        args.epochs,
        device,
        loss_fn_name=args.loss_fn
    )

    # Save model
    gcn_mlp = {
        'gcn_model': gcn_model.state_dict(),
        'mlp_model': mlp_model.state_dict()
    }
    torch.save(gcn_mlp, args.model_path)
    print(f"GCN and MLP models saved to '{args.model_path}'")

    # Save MSE values to CSV
    mse_df: pd.DataFrame = pd.DataFrame({
        "Epoch": list(range(1, args.epochs + 1)),
        "Train_MSE": train_mse,
        "Validation_MSE": val_mse
    })
    mse_csv_path: str = "mse_train_validation_gcn_nn.csv"
    mse_df.to_csv(mse_csv_path, index=False)
    print(f"MSE values saved to '{mse_csv_path}'")


if __name__ == "__main__":
    main()
