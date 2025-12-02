import re
from collections import OrderedDict
from itertools import chain
from typing import List, Tuple, Dict

# Pattern for SMILES tokenization with multi-character tokens first
token_pattern = re.compile(
    r'Br|Cl|Si|Se|As|'          # Multi-letter atom symbols
    r'[BCNOPSFHI]|'             # Single-letter atom symbols
    r'as|se|si|'                # lowercase multi-letter tokens
    r'[%()\[\]=#+\-\./\\1234567890]|'  # special chars, digits, bonds
    r'[a-z]'                    # any other lowercase atom
)

# Additional tokens to guarantee inclusion in vocabulary
additional_tokens = {'H', 'b', 'c', 'si', 'n', 'o', 'p', 's'}


def tokenize_smiles(smiles: str) -> List[str]:
    """
    Tokenize a SMILES string into meaningful chemical tokens.

    Args:
        smiles (str): Input SMILES string.

    Returns:
        list[str]: List of tokens extracted using `token_pattern`.
    """
    return token_pattern.findall(smiles)


def build_vocab(smiles_list: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Build a vocabulary dictionary (token → index) and reverse (index → token).

    Steps:
        1. Tokenize each SMILES.
        2. Gather all unique tokens (in order of appearance).
        3. Add required `additional_tokens`.
        4. Add special tokens: 'as', 'si', 'se', 'E'.

    Args:
        smiles_list (list[str]): List of SMILES to build vocabulary from.

    Returns:
        (char_to_int, int_to_char):
            char_to_int (dict): token → integer index
            int_to_char (dict): integer index → token
    """
    # Tokenize all SMILES → list[list[str]]
    tokenized = [tokenize_smiles(s) for s in smiles_list]

    # Flatten and extract in original order (acts like ordered unique)
    unique_tokens = sorted(
        list(OrderedDict.fromkeys(chain.from_iterable(tokenized)))
    )

    # Add guaranteed tokens
    for token in additional_tokens:
        if token not in unique_tokens:
            unique_tokens.append(token)

    # Special tokens used by generator
    for special in ['as', 'si', 'se', 'E']:
        if special not in unique_tokens:
            unique_tokens.append(special)

    # Create forward/backward mappings
    char_to_int: Dict[str, int] = {token: i for i, token in enumerate(unique_tokens)}
    int_to_char: Dict[int, str] = {i: token for token, i in char_to_int.items()}

    return char_to_int, int_to_char
