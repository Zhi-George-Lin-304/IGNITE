import torch
from torch import Tensor
from typing import Dict, Optional
from lstm.utils import is_valid_smiles


def generate_smiles(
    model: torch.nn.Module,
    char_to_int: Dict[str, int],
    int_to_char: Dict[int, str],
    start_token: str = "C",
    max_length: int = 100,
    max_attempts: int = 10,
    device: str | torch.device = "cpu"
) -> Optional[str]:
    """
    Generate a SMILES string autoregressively from an LSTM model.

    Args:
        model (nn.Module):
            Trained LSTM model that outputs (batch, seq, vocab_size).
        char_to_int (dict[str, int]):
            Mapping from token → index.
        int_to_char (dict[int, str]):
            Mapping from index → token.
        start_token (str):
            Initial token to begin generation (default: "C").
        max_length (int):
            Maximum number of tokens to generate.
        max_attempts (int):
            Number of retries until a valid SMILES is produced.
        device (str | torch.device):
            Device for inference (“cpu” or “cuda”).

    Returns:
        str | None:
            A valid generated SMILES string, or None if all attempts fail.
    """
    model.eval()
    vocab_size = len(char_to_int)

    for _ in range(max_attempts):
        hidden = None
        generated_indices = [char_to_int[start_token]]

        for _ in range(max_length):
            # x shape: (batch=1, seq=1, vocab_size)
            x: Tensor = torch.zeros((1, 1, vocab_size), device=device)
            x[0, 0, generated_indices[-1]] = 1.0

            output, hidden = model(x, hidden)  # output: (1, 1, vocab_size)
            next_index: int = output.argmax(dim=-1).item()

            # End token
            if int_to_char[next_index] == "E":
                break

            generated_indices.append(next_index)

        # Decode tokens → SMILES string
        smiles = "".join(int_to_char[idx] for idx in generated_indices)

        if is_valid_smiles(smiles):
            return smiles

    return None
