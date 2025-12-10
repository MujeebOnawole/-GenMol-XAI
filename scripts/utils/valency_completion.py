"""
Valency completion module for fixing valency issues in generated molecules.
"""
from typing import Optional, Tuple

try:
    from rdkit import Chem
except Exception:
    Chem = None


def fix_valency(smiles: str) -> Tuple[Optional[str], bool]:
    """
    Attempt to sanitize/fix valency and return a possibly corrected SMILES.

    Args:
        smiles: Input SMILES string

    Returns:
        Tuple of (fixed_smiles_or_none, valency_fixed_flag)
    """
    if not Chem:
        return smiles, False
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if not mol:
            return None, False
        try:
            Chem.SanitizeMol(mol)
            fixed = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
            return fixed, True
        except Exception:
            # Try a second pass after adding hydrogens
            try:
                mol = Chem.AddHs(mol)
                Chem.SanitizeMol(mol)
                mol = Chem.RemoveHs(mol)
                Chem.SanitizeMol(mol)
                fixed = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
                return fixed, True
            except Exception:
                return None, False
    except Exception:
        return None, False
