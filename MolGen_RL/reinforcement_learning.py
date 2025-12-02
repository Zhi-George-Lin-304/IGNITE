import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
import random
from typing import Dict, List, Tuple

from lstm_gcn_rl.model_utils import *  # LSTMModel, load_gcn_mlp_model, sample_smiles_with_logprob, etc.

# Global device
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load token mappings
char_to_int: Dict[str, int] = torch.load('MolGen_RL/data/token/char_to_int.pt')
int_to_char: Dict[int, str] = torch.load('MolGen_RL/data/token/int_to_char.pt')


def main() -> None:
    """
    Reinforcement learning loop for fine-tuning an LSTM SMILES generator
    using a GCN+MLP property predictor as the environment.

    Steps:
        1. Load pretrained LSTM and GCN+MLP models.
        2. Autoregressively sample SMILES from LSTM (policy).
        3. Convert SMILES → RDKit Mol and predict property via GCN+MLP.
        4. Compute reward based on distance to target property.
        5. Update LSTM parameters using REINFORCE.
        6. Periodically save fine-tuned checkpoints.
    """
    device_str: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Match the original LSTM architecture
    input_dim: int = len(char_to_int)
    hidden_dim: int = 256
    output_dim: int = len(char_to_int)
    num_layers: int = 5

    # Load the pre-trained LSTM model
    model: LSTMModel = LSTMModel(input_dim, hidden_dim, output_dim, num_layers).to(device_str)
    model.load_state_dict(torch.load("MolGen_RL/lstm.pt", map_location=device_str))
    model.train()

    optimizer: optim.Optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Load GCN + MLP prediction model
    gcn, mlp = load_gcn_mlp_model('MolGen_RL/gcn_mlp.pth', device=device_str)

    target_gap: float = 0.1  # Desired singlet–triplet gap

    for step in range(1000):
        # Sample a SMILES string and log probability from the LSTM
        smiles_idx, log_prob = sample_smiles_with_logprob(model, char_to_int, device=device_str)
        smiles_str: str = ''.join(int_to_char[i] for i in smiles_idx)
        mol = smiles_to_mol(smiles_str)

        if mol is None:
            continue

        try:
            # Predict the property and compute reward
            pred_gap: float = predict_gap(mol, gcn, mlp, device=device_str)
            reward: float = reward_function(pred_gap, target_gap)
            reinforce_update(log_prob, reward, optimizer)

            if step % 10 == 0:
                print(f"Step {step}: SMILES = {smiles_str}, Gap = {pred_gap:.3f}, Reward = {reward:.4f}")

            if step % 100 == 0:
                torch.save(model.state_dict(), f"lstm_finetuned_step{step}.pt")

        except Exception as e:
            print(f"Skipping molecule due to error: {e}")

    # Final save
    torch.save(model.state_dict(), "lstm_finetuned_final.pt")


if __name__ == "__main__":
    main()
