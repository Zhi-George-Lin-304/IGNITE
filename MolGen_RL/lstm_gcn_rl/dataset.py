import torch
import os
from tqdm import tqdm
from torch import Tensor
from torch.utils.data import Dataset
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from rdkit import Chem
from lstm_gcn_rl.utils import generate_adjacency_matrix, generate_feature_matrix


class MolecularDataset(Dataset):
    """
    PyTorch Dataset for preparing molecular graph inputs for a GCN model.
    
    Each item returns:
        - normalized adjacency matrix      (max_atoms × max_atoms)
        - padded atom feature matrix       (max_atoms × feature_dim)
        - target property                  (float32)
        - num_atoms                        (# actual atoms before padding)
    """

    def __init__(
        self,
        csv_file: str,
        target_col: str,
        max_atoms: int = 240,
        config: Dict[str, bool] | None = None,
        only_light_atoms: bool = False
    ) -> None:
        """
        Args:
            csv_file (str): Tab-separated .txt/.tsv file containing SMILES and target column.
            target_col (str): Column name for target values.
            max_atoms (int): Maximum number of atoms allowed (padding applied).
            config (dict | None): Feature extraction configuration.
            only_light_atoms (bool): If True, skip molecules with heavy atoms (Z ≥ 9).
        """
        data = pd.read_csv(csv_file, sep="\t").dropna(subset=["SMILES"])
        self.df = data
        self.df["SMILES"] = self.df["SMILES"].astype(str)
        self.df[target_col] = self.df[target_col].astype(float)

        self.smiles: List[str] = []
        self.targets: List[float] = []
        self.max_atoms: int = max_atoms
        self.config = config

        # Filter & collect SMILES
        for _, row in self.df.iterrows():
            mol = Chem.MolFromSmiles(row["SMILES"])
            if mol:
                atomic_nums = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
                if only_light_atoms and any(z >= 9 for z in atomic_nums):
                    continue
                self.smiles.append(row["SMILES"])
                self.targets.append(row[target_col])

        # Convert targets to tensor
        self.targets: Tensor = torch.tensor(self.targets, dtype=torch.float32)

        # Determine feature dimension
        if self.smiles:
            sample_mol = Chem.MolFromSmiles(self.smiles[0])
            self.feature_dim: int = generate_feature_matrix(sample_mol, config=self.config).shape[1]
        else:
            self.feature_dim = 0

    def __len__(self) -> int:
        """Return number of valid molecules."""
        return len(self.smiles)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Generate padded & normalized graph inputs for a single molecule.

        Returns:
            adj_normalized (Tensor):  (max_atoms × max_atoms)
            padded_features (Tensor): (max_atoms × feature_dim)
            target (Tensor):          scalar float32
            num_atoms (Tensor):       scalar int64
        """
        smiles: str = self.smiles[idx]
        mol = Chem.MolFromSmiles(smiles)

        # Raw matrices
        adj: np.ndarray = generate_adjacency_matrix(mol)
        features: np.ndarray = generate_feature_matrix(mol, config=self.config)
        num_atoms: int = adj.shape[0]

        # Pad adjacency
        padded_adj = np.zeros((self.max_atoms, self.max_atoms), dtype=np.float32)
        padded_adj[:num_atoms, :num_atoms] = adj

        # Normalization: A_hat = D^{-1/2}(A + I)D^{-1/2}
        adj_hat = padded_adj + np.eye(self.max_atoms, dtype=np.float32)
        degrees = np.sum(adj_hat, axis=1)
        D_inv_sqrt = np.diag(np.power(degrees, -0.5, where=degrees != 0))
        adj_normalized = D_inv_sqrt @ adj_hat @ D_inv_sqrt

        # Pad features
        padded_features = np.zeros((self.max_atoms, self.feature_dim), dtype=np.float32)
        padded_features[:num_atoms, :] = features

        return (
            torch.tensor(adj_normalized, dtype=torch.float32),
            torch.tensor(padded_features, dtype=torch.float32),
            self.targets[idx],
            torch.tensor(num_atoms, dtype=torch.int64)
        )


def save_dataset_in_chunks(
    dataset: Dataset,
    save_dir: str,
    prefix: str,
    chunk_size: int = 1000
) -> None:
    """
    Save large MolecularDataset objects into multiple .pt chunks.

    Args:
        dataset (Dataset): The MolecularDataset instance.
        save_dir (str): Directory to save chunks.
        prefix (str): Prefix of output filenames.
        chunk_size (int): Number of samples per chunk.
    
    Saves:
        Files named:
            prefix_0_999.pt
            prefix_1000_1999.pt
            ...
    """
    os.makedirs(save_dir, exist_ok=True)
    current_chunk: List[Dict[str, Tensor]] = []
    chunk_idx: int = 0

    for i in tqdm(range(len(dataset)), desc=f"Saving {prefix}"):
        adj, feat, target, num_atoms = dataset[i]

        sample: Dict[str, Tensor] = {
            "adj": adj,
            "feat": feat,
            "target": target,
            "num_atoms": num_atoms
        }
        current_chunk.append(sample)

        # Write chunk when full
        if len(current_chunk) == chunk_size:
            start_idx = chunk_idx * chunk_size
            end_idx = start_idx + chunk_size - 1
            chunk_file = os.path.join(save_dir, f"{prefix}_{start_idx}_{end_idx}.pt")

            torch.save(current_chunk, chunk_file)
            current_chunk = []
            chunk_idx += 1

    # Save remaining samples
    if current_chunk:
        start_idx = chunk_idx * chunk_size
        end_idx = start_idx + len(current_chunk) - 1
        chunk_file = os.path.join(save_dir, f"{prefix}_{start_idx}_{end_idx}.pt")
        torch.save(current_chunk, chunk_file)


# Example usage
# train_dataset = MolecularDataset("data/train.txt", target_col="ST_split")
# val_dataset = MolecularDataset("data/val.txt", target_col="ST_split")

# save_dataset_in_chunks(train_dataset, save_dir="data/train_chunks", prefix="train", chunk_size=1000)
# save_dataset_in_chunks(val_dataset, save_dir="data/val_chunks", prefix="val", chunk_size=1000)
