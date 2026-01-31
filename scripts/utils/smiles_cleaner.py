"""
SMILES cleaner module for standardizing and canonicalizing SMILES.
"""
from typing import Optional, Tuple

try:
    from rdkit import Chem
except Exception:
    Chem = None


def clean_smiles(smiles: str) -> Tuple[Optional[str], bool]:
    """
    Standardize and canonicalize SMILES.

    Args:
        smiles: Input SMILES string

    Returns:
        Tuple of (canonical_smiles, cleaning_applied)
    """
    if not Chem:
        return smiles, False
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return None, False
        Chem.SanitizeMol(mol)
        can = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        return can, True
    except Exception:
        return None, False
