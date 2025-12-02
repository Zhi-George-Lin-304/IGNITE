import torch
from torch.utils.data import Dataset
import pandas as pd 
import numpy as np
from rdkit import Chem
from gcn_mlp.utils import generate_adjacency_matrix, generate_feature_matrix

class MolecularDataset(Dataset):
    """
    PyTorch Dataset for molecular graphs generated from SMILES strings.

    Each item returns:
        - normalized adjacency matrix   : Tensor (max_atoms * max_atoms)
        - padded feature matrix         : Tensor (max_atoms * feature_dim)
        - target property               : float Tensor
        - number of atoms in molecule   : int Tensor
    """

    def __init__(self, csv_file, target_col, max_atoms=460, config=None, only_light_atoms=False):
        data = pd.read_csv(csv_file, sep="\t").dropna(subset=["SMILES"])
        self.df = data
        self.df["SMILES"] = self.df["SMILES"].astype(str)
        self.df[target_col] = self.df[target_col].astype(float)

        self.smiles = []
        self.targets = []
        self.max_atoms = max_atoms
        self.config = config

        for _, row in self.df.iterrows():
            mol = Chem.MolFromSmiles(row["SMILES"])
            if mol:
                atomic_nums = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
                if only_light_atoms and any(z >= 9 for z in atomic_nums):
                    continue  # skip molecule if it contains atoms with Z >= 9
                self.smiles.append(row["SMILES"])
                self.targets.append(row[target_col])

        self.targets = torch.tensor(self.targets, dtype=torch.float32)

        if self.smiles:
            sample_mol = Chem.MolFromSmiles(self.smiles[0])
            self.feature_dim = generate_feature_matrix(sample_mol, config=self.config).shape[1]
        else:
            self.feature_dim = 0

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smiles = self.smiles[idx]
        mol = Chem.MolFromSmiles(smiles)

        adj = generate_adjacency_matrix(mol)
        features = generate_feature_matrix(mol, config=self.config)

        num_atoms = adj.shape[0]

        padded_adj = np.zeros((self.max_atoms, self.max_atoms), dtype=np.float32)
        padded_adj[:num_atoms, :num_atoms] = adj

        adj_hat = padded_adj + np.eye(self.max_atoms, dtype=np.float32)
        D_inv_sqrt = np.diag(np.power(np.sum(adj_hat, axis=1), -0.5, where=np.sum(adj_hat, axis=1) != 0))
        adj_normalized = D_inv_sqrt @ adj_hat @ D_inv_sqrt

        padded_features = np.zeros((self.max_atoms, self.feature_dim), dtype=np.float32)
        padded_features[:num_atoms, :] = features

        return (
            torch.tensor(adj_normalized, dtype=torch.float32),
            torch.tensor(padded_features, dtype=torch.float32),
            self.targets[idx],
            torch.tensor(num_atoms, dtype=torch.int64)
        )
