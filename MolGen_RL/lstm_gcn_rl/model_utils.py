import torch
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, List, Tuple, Optional

from rdkit import Chem
from rdkit.Chem import Mol
from torch import Tensor
from torch.distributions import Categorical
from torch.optim import Optimizer

from lstm_gcn_rl.utils import generate_adjacency_matrix, generate_feature_matrix
from lstm_gcn_rl.model import GCN, MLP


def mol_to_graph_data(
    mol: Mol,
    feature_dim: int = 42,
    max_atoms: int = 460,
    config: Optional[Dict] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Convert an RDKit Mol object into padded graph tensors.

    Args:
        mol (Mol): RDKit molecule.
        feature_dim (int): Number of atom features per node.
        max_atoms (int): Maximum number of atoms (for padding).
        config (dict | None): Optional feature configuration for generate_feature_matrix.

    Returns:
        adj_normalized (Tensor): (1, max_atoms, max_atoms) normalized adjacency matrix.
        padded_features (Tensor): (1, max_atoms, feature_dim) padded feature matrix.
        num_atoms (Tensor): (1,) tensor with the actual number of atoms.
    """
    adj = generate_adjacency_matrix(mol)
    features = generate_feature_matrix(mol, config=config)
    num_atoms = adj.shape[0]

    padded_adj = torch.zeros((max_atoms, max_atoms), dtype=torch.float32)
    padded_adj[:num_atoms, :num_atoms] = torch.tensor(adj, dtype=torch.float32)

    adj_hat = padded_adj + torch.eye(max_atoms, dtype=torch.float32)
    D_inv_sqrt = torch.diag(torch.pow(adj_hat.sum(dim=1), -0.5))
    adj_normalized = D_inv_sqrt @ adj_hat @ D_inv_sqrt

    padded_features = torch.zeros((max_atoms, feature_dim), dtype=torch.float32)
    padded_features[:num_atoms, :] = torch.tensor(features, dtype=torch.float32)

    return (
        adj_normalized.unsqueeze(0),
        padded_features.unsqueeze(0),
        torch.tensor([num_atoms], dtype=torch.int64),
    )


def decode_smiles(indices: List[int], vocab: Dict[str, int]) -> str:
    """
    Decode a list of token indices into a SMILES string.

    Args:
        indices (list[int]): Token indices.
        vocab (dict[str, int]): Token → index mapping.

    Returns:
        str: Decoded SMILES string.
    """
    inv_vocab: Dict[int, str] = {v: k for k, v in vocab.items()}
    return "".join(inv_vocab.get(i, "?") for i in indices)


def _tokens_to_onehot(tokens: Tensor, num_classes: int, device: str | torch.device) -> Tensor:
    """
    Convert token indices to one-hot encoding.

    Args:
        tokens (Tensor): Shape (T,) or (B, T) with integer indices.
        num_classes (int): Vocabulary size.
        device (str | torch.device): Device to place the tensor on.

    Returns:
        Tensor: One-hot tensor of shape (T, C) or (B, T, C).
    """
    if tokens.dim() == 1:
        onehot = torch.zeros(tokens.size(0), num_classes, device=device)
        onehot.scatter_(1, tokens.unsqueeze(1), 1.0)
    elif tokens.dim() == 2:
        onehot = torch.zeros(tokens.size(0), tokens.size(1), num_classes, device=device)
        onehot.scatter_(2, tokens.unsqueeze(2), 1.0)
    else:
        raise ValueError("Unexpected token tensor shape")
    return onehot.float()


def sample_smiles_with_logprob(
    model: torch.nn.Module,
    vocab: Dict[str, int],
    max_len: int = 300,
    device: str | torch.device = "cpu",
) -> Tuple[List[int], Tensor]:
    """
    Sample a SMILES sequence from the LSTM and accumulate log-probabilities.

    Args:
        model (nn.Module): LSTM model with a final linear layer `fc`.
        vocab (dict[str, int]): Token → index mapping (must contain 'C' and 'E').
        max_len (int): Maximum length of generated sequence.
        device (str | torch.device): Device for computation.

    Returns:
        smiles (list[int]): List of sampled token indices (excluding 'E').
        log_prob_sum (Tensor): Scalar tensor with sum of log-probs.
    """
    input_token = torch.tensor([[vocab["C"]]], device=device)
    hidden: Optional[Tuple[Tensor, Tensor]] = None
    smiles: List[int] = []
    log_probs: List[Tensor] = []
    output_dim: int = model.fc.out_features  # type: ignore[attr-defined]

    for _ in range(max_len):
        input_token_onehot = _tokens_to_onehot(input_token, output_dim, device)
        output, hidden = model(input_token_onehot, hidden)
        probs = torch.softmax(output[:, -1, :], dim=-1)
        dist = Categorical(probs)
        token = dist.sample()
        log_prob = dist.log_prob(token)

        token_idx = token.item()
        if token_idx == vocab["E"]:
            break

        smiles.append(token_idx)
        log_probs.append(log_prob)
        input_token = token.unsqueeze(0)

    if len(log_probs) == 0:
        return smiles, torch.tensor(0.0, device=device)

    return smiles, torch.stack(log_probs).sum()


def sample(
    model: torch.nn.Module,
    char_to_int: Dict[str, int],
    int_to_char: Dict[int, str],
    start_token: str = "C",
    end_token: str = "E",
    max_length: int = 100,
    temperature: float = 1.0,
    device: str | torch.device = "cpu",
) -> str:
    """
    Sample a SMILES string from the model using temperature-scaled softmax.

    Args:
        model (nn.Module): LSTM model.
        char_to_int (dict[str, int]): Token → index mapping.
        int_to_char (dict[int, str]): Index → token mapping.
        start_token (str): Starting token.
        end_token (str): End-of-sequence token.
        max_length (int): Maximum length of generated sequence.
        temperature (float): Softmax temperature (>0).
        device (str | torch.device): Device for computation.

    Returns:
        str: Generated SMILES string (without explicit end token).
    """
    model.eval()
    generated: List[int] = [char_to_int[start_token]]
    input_seq = torch.tensor([[generated[-1]]], dtype=torch.long, device=device)
    hidden: Optional[Tuple[Tensor, Tensor]] = None

    for _ in range(max_length):
        output, hidden = model(input_seq, hidden)
        logits = output[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        next_idx = torch.multinomial(probs, 1).item()

        if int_to_char[next_idx] == end_token:
            break

        generated.append(next_idx)
        input_seq = torch.tensor([[next_idx]], dtype=torch.long, device=device)

    return "".join(int_to_char[i] for i in generated)


def smiles_to_mol(smiles_str: str) -> Optional[Mol]:
    """
    Convert a SMILES string to an RDKit Mol object.

    Args:
        smiles_str (str): Input SMILES.

    Returns:
        Mol | None: RDKit molecule or None if parsing fails.
    """
    return Chem.MolFromSmiles(smiles_str)


def predict_gap(
    mol: Mol,
    gcn: GCN,
    mlp: MLP,
    feature_dim: int = 42,
    max_atoms: int = 240,
    config: Optional[Dict] = None,
    device: str | torch.device = "cpu",
) -> float:
    """
    Predict a scalar property (e.g., singlet–triplet gap) using GCN + MLP.

    Args:
        mol (Mol): RDKit molecule.
        gcn (GCN): GCN encoder model.
        mlp (MLP): MLP readout model.
        feature_dim (int): Node feature dimension.
        max_atoms (int): Maximum number of atoms for padding.
        config (dict | None): Feature configuration.
        device (str | torch.device): Device for inference.

    Returns:
        float: Predicted property value.
    """
    with torch.no_grad():
        adj = generate_adjacency_matrix(mol)
        features = generate_feature_matrix(mol, config=config)
        num_atoms = adj.shape[0]

        padded_adj = np.zeros((max_atoms, max_atoms), dtype=np.float32)
        padded_adj[:num_atoms, :num_atoms] = adj
        adj_hat = padded_adj + np.eye(max_atoms, dtype=np.float32)
        D_inv_sqrt = np.diag(
            np.power(
                np.sum(adj_hat, axis=1),
                -0.5,
                where=np.sum(adj_hat, axis=1) != 0,
            )
        )
        adj_normalized = D_inv_sqrt @ adj_hat @ D_inv_sqrt

        padded_features = np.zeros((max_atoms, feature_dim), dtype=np.float32)
        padded_features[:num_atoms, :] = features

        A = torch.tensor(adj_normalized, dtype=torch.float32).unsqueeze(0).to(device)
        X = torch.tensor(padded_features, dtype=torch.float32).unsqueeze(0).to(device)
        num_atoms_tensor = torch.tensor([num_atoms], dtype=torch.long).to(device)

        embedding = gcn(A, X, num_atoms_tensor)
        return mlp(embedding).item()


def reward_function(pred_gap: float, target_gap: float, beta: float = 5.0) -> float:
    """
    Reward function shaped as a Gaussian around the target gap.

    Args:
        pred_gap (float): Predicted gap.
        target_gap (float): Desired gap.
        beta (float): Sharpness of penalty.

    Returns:
        float: Reward in [0, 1].
    """
    return math.exp(-beta * (pred_gap - target_gap) ** 2)


def reinforce_update(
    log_prob: Tensor,
    reward: float,
    optimizer: Optimizer,
) -> None:
    """
    Single-step REINFORCE parameter update.

    Args:
        log_prob (Tensor): Sum of log-probabilities of sampled actions.
        reward (float): Scalar reward.
        optimizer (Optimizer): Optimizer for the policy network.
    """
    loss = -reward * log_prob
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def load_gcn_mlp_model(
    path: str,
    input_dim: int = 42,
    gcn_dim: int = 100,
    mlp_dim: int = 300,
    device: str | torch.device = "cpu",
) -> Tuple[GCN, MLP]:
    """
    Load GCN + MLP models from a checkpoint.

    Args:
        path (str): Path to checkpoint (.pth) with 'gcn_model' and 'mlp_model' keys.
        input_dim (int): Node feature dimension.
        gcn_dim (int): GCN hidden dimension.
        mlp_dim (int): MLP hidden dimension.
        device (str | torch.device): Device to map the models to.

    Returns:
        (gcn, mlp): Loaded and eval-mode GCN and MLP models.
    """
    gcn = GCN(input_dim, [gcn_dim]).to(device)
    mlp = MLP(gcn_dim, [mlp_dim], output_dim=1).to(device)

    checkpoint = torch.load(path, map_location=device)
    gcn.load_state_dict(checkpoint["gcn_model"])
    mlp.load_state_dict(checkpoint["mlp_model"])

    gcn.eval()
    mlp.eval()

    return gcn, mlp
