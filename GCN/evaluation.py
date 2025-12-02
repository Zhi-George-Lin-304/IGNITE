import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from rdkit import Chem
from typing import List, Dict, Any

from gcn_mlp.dataset import MolecularDataset
from gcn_mlp.model import GCN, MLP

# === Paths ===
model_path: str = 'gcn_mlp.pth'
input_smiles: str = r'\IGNITE\GCN\generated_smiles\generated_smiles.txt'
output_csv: str = 'predicted_generated_smiles.csv'

# === Config ===
max_atoms: int = 240
output_dim: int = 1
batch_size: int = 64
config: Dict[str, bool] | None = None
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Step 1: Convert SMILES to CSV ===
with open(input_smiles, "r") as f:
    smiles_list: List[str] = [
        line.strip()
        for line in f
        if Chem.MolFromSmiles(line.strip()) is not None
    ]

df: pd.DataFrame = pd.DataFrame(
    {"SMILES": smiles_list, "target": [0.0] * len(smiles_list)}
)
temp_csv_path: str = "temp_generated_smiles.csv"
df.to_csv(temp_csv_path, sep="\t", index=False)

# === Step 2: Load dataset ===
dataset: MolecularDataset = MolecularDataset(
    csv_file=temp_csv_path,
    target_col="target",
    max_atoms=max_atoms,
    config=config
)
loader: DataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
input_dim: int = dataset.feature_dim

# === Step 3: Load checkpoint and infer hidden dims ===
checkpoint: Dict[str, Any] = torch.load(model_path, map_location=device)
gcn_state: Dict[str, Tensor] = checkpoint['gcn_model']
mlp_state: Dict[str, Tensor] = checkpoint['mlp_model']

# Infer GCN hidden dims from state_dict
gcn_hidden_dims: List[int] = []
i: int = 0
while f"gcn_layers.{i}.linear.weight" in gcn_state:
    weight_shape = gcn_state[f"gcn_layers.{i}.linear.weight"].shape  # (out_dim, in_dim)
    gcn_hidden_dims.append(weight_shape[0])  # out_dim
    i += 1

# Infer MLP hidden dims from state_dict
mlp_hidden_dims: List[int] = []
i = 0
while f"network.{2 * i}.weight" in mlp_state:  # Only linear layers (indices 0, 2, 4, ...)
    out_dim, in_dim = mlp_state[f"network.{2 * i}.weight"].shape
    if f"network.{2 * (i + 1)}.weight" in mlp_state:
        mlp_hidden_dims.append(out_dim)  # hidden layer
    else:
        output_dim = out_dim  # final output layer
    i += 1

# === Step 4: Rebuild models ===
gcn_model: GCN = GCN(input_dim=input_dim, hidden_dims=gcn_hidden_dims).to(device)
mlp_model: MLP = MLP(input_dim=gcn_hidden_dims[-1], hidden_dims=mlp_hidden_dims, output_dim=output_dim).to(device)

gcn_model.load_state_dict(gcn_state)
mlp_model.load_state_dict(mlp_state)
gcn_model.eval()
mlp_model.eval()

# === Step 5: Predict ===
predictions: List[float] = []

with torch.no_grad():
    for A, X, _, num_atoms in loader:
        A = A.to(device)
        X = X.to(device)
        num_atoms = num_atoms.to(device)

        embeddings: Tensor = gcn_model(A, X, num_atoms)
        outputs: Tensor = mlp_model(embeddings)
        predictions.extend(outputs.squeeze().cpu().numpy().tolist())

# === Step 6: Save predictions for valid molecules ===
valid_smiles: List[str] = dataset.smiles  # Only valid molecules used in dataset
df_valid: pd.DataFrame = pd.DataFrame(
    {"SMILES": valid_smiles, "predicted_value": predictions}
)
df_valid.to_csv(output_csv, index=False)
print(f"Predictions saved to: {output_csv}")
