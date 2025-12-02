import numpy as np
from rdkit import Chem

def generate_adjacency_matrix(mol):
    """
    Generate a weighted adjacency matrix for an RDKit molecule.

    Each atom corresponds to a node; each bond contributes a weight:
        SINGLE   -> 1.0
        DOUBLE   -> 2.0
        TRIPLE   -> 3.0
        AROMATIC -> 1.5

    A self-loop of weight 1.0 is added to every atom.

    The function stores the matrix size in:
        generate_adjacency_matrix.last_size
    so users can retrieve it without changing the return signature.
    
    Args:
        mol (RDKit Mol): Input molecule.

    Returns:
        np.ndarray: (N, N) adjacency matrix.
    """

    num_atoms = mol.GetNumAtoms()
    adj_matrix = np.zeros((num_atoms, num_atoms), dtype=np.float32)

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_type = bond.GetBondType()

        if bond_type == Chem.rdchem.BondType.SINGLE:
            value = 1.0
        elif bond_type == Chem.rdchem.BondType.DOUBLE:
            value = 2.0
        elif bond_type == Chem.rdchem.BondType.AROMATIC:
            value = 1.5
        elif bond_type == Chem.rdchem.BondType.TRIPLE:
            value = 3.0
        else:
            value = 0.0

        adj_matrix[i, j] = adj_matrix[j, i] = value

    np.fill_diagonal(adj_matrix, 1.0)  # self-loop

    generate_adjacency_matrix.last_size = num_atoms

    return adj_matrix



def generate_feature_matrix(mol, config=None):
    """
    Generate an atom-level feature matrix for a molecule.
    
    Args:
        mol: RDKit Mol object
        config: dict specifying which features to include, e.g.
            {
                "atom_type": True,
                "num_hydrogens": True,
                "aromaticity": True,
                "hybridization": True,
                "formal_charge": True,
                "valence_electrons": True,
                "degree": True
            }

    Returns:
        np.ndarray: Feature matrix (n_atoms, n_features)
    """
    if config is None:
        config = {
            "atom_type": True,
            "num_hydrogens": True,
            "aromaticity": True,
            "hybridization": True,
            "formal_charge": True,
            "valence_electrons": True,
            "degree": True
        }

    atom_types = ['As', 'B', 'Br', 'C', 'Cl', 'F', 'I', 'N', 'O', 'P', 'S', 'Se', 'Si']
    valence_electrons = {
        'As': 5, 'B': 3, 'Br': 7, 'C': 4, 'Cl': 7, 'F': 7, 'I': 7,
        'N': 5, 'O': 6, 'P': 5, 'S': 6, 'Se': 6, 'Si': 4
    }

    feature_matrix = []

    for atom in mol.GetAtoms():
        features = []

        if config.get("atom_type", False):
            symbol = atom.GetSymbol()
            features.extend([1 if symbol == t else 0 for t in atom_types])

        if config.get("num_hydrogens", False):
            num_h = atom.GetTotalNumHs()
            features.extend([1 if num_h == i else 0 for i in range(5)])

        if config.get("aromaticity", False):
            features.append(1 if atom.GetIsAromatic() else 0)

        if config.get("hybridization", False):
            hyb = atom.GetHybridization()
            features.extend([
                1 if hyb == Chem.rdchem.HybridizationType.SP else 0,
                1 if hyb == Chem.rdchem.HybridizationType.SP2 else 0,
                1 if hyb == Chem.rdchem.HybridizationType.SP3 else 0,
            ])

        if config.get("formal_charge", False):
            charge = atom.GetFormalCharge()
            features.extend([1 if charge == i else 0 for i in range(-3, 4)])

        if config.get("valence_electrons", False):
            symbol = atom.GetSymbol()
            valence = valence_electrons.get(symbol, 0)
            features.extend([1 if valence == i else 0 for i in range(1, 9)])

        if config.get("degree", False):
            degree = atom.GetDegree()
            features.extend([1 if degree == i else 0 for i in range(5)])

        feature_matrix.append(features)

    return np.array(feature_matrix, dtype=np.float32)
