#!/usr/bin/env python3
"""
Safe Library Builder for GenMol

Builds safe fragment libraries from source CSV files for molecule generation.

Supports 7 library types:
- SA, EC, CA: Pathogen-specific (active against ONE pathogen only)
- SA_EC, SA_CA, CA_EC: Dual-active (active against exactly TWO pathogens)
- TRIPLE: Triple-active (active against ALL THREE pathogens)

Usage:
    python safe_library_builder.py --library SA --base-dir .
    python safe_library_builder.py --library SA_EC --base-dir .
    python safe_library_builder.py --all --base-dir .

Author: GenMol Pipeline
"""

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors
    from rdkit.Chem.inchi import MolToInchiKey
except ImportError:
    Chem = None
    Descriptors = Crippen = Lipinski = rdMolDescriptors = None
    MolToInchiKey = None


SMILES_CANDIDATES = ["smiles", "fragment_smiles", "SMILES", "canonical_smiles"]

SMARTS = {
    "acid_neutral": "[CX3](=O)[OX1H]",
    "acid_carboxylate": "[CX3](=O)[O-]",
    "primary_amine": "[NX3;H2;!$(NC=O)]",
    "secondary_amine": "[NX3;H1;!$(NC=O)]",
    "primary_amine_protonated": "[N+;H3]",
    "secondary_amine_protonated": "[N+;H2]",
}

LIBRARY_TYPES = ["SA", "EC", "CA", "SA_EC", "SA_CA", "CA_EC", "TRIPLE"]


def _canonicalize_smiles(smiles: str) -> Optional[str]:
    """Canonicalize SMILES using RDKit."""
    if not Chem:
        return smiles.strip()
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return None
        return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    except Exception:
        return None


def _compute_props(smiles: str) -> Tuple[Optional[Dict], Optional[str]]:
    """Compute molecular properties."""
    if not Chem:
        return ({}, None)
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return (None, None)
    try:
        props = {
            "MW": float(Descriptors.MolWt(mol)),
            "LogP": float(Crippen.MolLogP(mol)),
            "HBA": int(Lipinski.NumHAcceptors(mol)),
            "HBD": int(Lipinski.NumHDonors(mol)),
            "TPSA": float(rdMolDescriptors.CalcTPSA(mol)),
            "AromRings": int(rdMolDescriptors.CalcNumAromaticRings(mol)),
            "RotBonds": int(Lipinski.NumRotatableBonds(mol)),
        }
        inchikey = MolToInchiKey(mol) if MolToInchiKey else None
        return props, inchikey
    except Exception:
        return (None, None)


def _detect_handles(smiles: str) -> Dict[str, bool]:
    """Detect reactive handles (acid, amine) in fragment."""
    if not Chem:
        return {"carboxylic_acid": False, "primary_amine": False, "secondary_amine": False}
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return {"carboxylic_acid": False, "primary_amine": False, "secondary_amine": False}

    out = {"carboxylic_acid": False, "primary_amine": False, "secondary_amine": False}

    # Check acids
    for pattern_key in ["acid_neutral", "acid_carboxylate"]:
        pattern = Chem.MolFromSmarts(SMARTS.get(pattern_key, ""))
        if pattern and mol.HasSubstructMatch(pattern):
            out["carboxylic_acid"] = True
            break

    # Check amines
    for pattern_key in ["primary_amine", "primary_amine_protonated"]:
        pattern = Chem.MolFromSmarts(SMARTS.get(pattern_key, ""))
        if pattern and mol.HasSubstructMatch(pattern):
            out["primary_amine"] = True
            break

    for pattern_key in ["secondary_amine", "secondary_amine_protonated"]:
        pattern = Chem.MolFromSmarts(SMARTS.get(pattern_key, ""))
        if pattern and mol.HasSubstructMatch(pattern):
            out["secondary_amine"] = True
            break

    return out


def _read_csv_rows(path: str) -> List[Dict[str, str]]:
    """Read CSV file and return list of row dictionaries."""
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        try:
            csv.field_size_limit(1000000000)
        except Exception:
            pass
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _first_smiles_value(row: Dict[str, str]) -> Optional[str]:
    """Extract first valid SMILES from row."""
    for key in SMILES_CANDIDATES:
        for cand in [key, key.lower(), key.upper()]:
            if cand in row and row[cand]:
                return row[cand]
    return None


def _infer_role_from_filename(filename: str) -> Optional[str]:
    """Infer scaffold/substituent role from filename."""
    low = filename.lower()
    if "scaffold" in low:
        return "scaffold"
    if "substituent" in low or "substitutent" in low:
        return "substituent"
    return None


def get_source_files(base_dir: str, library_type: str) -> List[Tuple[str, str]]:
    """
    Get source fragment CSV files for a library type.

    NEW SIMPLIFIED STRUCTURE:
    All fragments are in data/source_fragments/ with consistent naming:
    - {LIBRARY}_positive_scaffolds.csv
    - {LIBRARY}_positive_substituents.csv

    Returns list of (filepath, role) tuples.
    """
    fragments_dir = os.path.join(base_dir, "data", "source_fragments")

    files = [
        (os.path.join(fragments_dir, f"{library_type}_positive_scaffolds.csv"), "scaffold"),
        (os.path.join(fragments_dir, f"{library_type}_positive_substituents.csv"), "substituent"),
    ]

    return files


def build_safe_library(library_type: str, base_dir: str) -> Tuple[List[Dict], List[str]]:
    """
    Build safe library from source fragments.

    Returns (fragments_list, diagnostics_lines).
    """
    files_with_roles = get_source_files(base_dir, library_type)
    now = datetime.utcnow().isoformat()
    seen: Dict[str, Dict] = {}
    diagnostics_lines = []

    for csv_path, role in files_with_roles:
        if not os.path.exists(csv_path):
            diagnostics_lines.append(f"MISSING: {csv_path}")
            print(f"Warning: Source file not found: {csv_path}")
            continue

        rows = _read_csv_rows(csv_path)
        diagnostics_lines.append(f"Loaded: {csv_path} ({len(rows)} rows)")

        for row in rows:
            smi_raw = _first_smiles_value(row)
            if not smi_raw:
                continue

            smi = _canonicalize_smiles(smi_raw)
            if not smi:
                continue

            # Get or generate fragment ID
            frag_id = row.get("fragment_id") or row.get("id")
            if frag_id:
                frag_id = frag_id.strip()
            else:
                sha = hashlib.sha1(smi.encode("utf-8")).hexdigest()[:10]
                frag_id = f"frag-{library_type}-{sha}"

            # Skip duplicates
            if frag_id in seen:
                continue

            # Compute properties
            props, inchikey = _compute_props(smi)
            if props is None:
                continue

            handles = _detect_handles(smi)

            entry = {
                "fragment_id": frag_id,
                "fragment_smiles": smi,
                "inchikey": inchikey or "",
                "library_type": library_type,
                "role": role,
                "reactive_handles": handles,
                "provenance": {
                    "source_csv": os.path.basename(csv_path),
                    "timestamp": now,
                },
                "props": props,
            }
            seen[frag_id] = entry

    return list(seen.values()), diagnostics_lines


def write_safe_library(path: str, data: List[Dict]) -> None:
    """Write safe library to JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(
        description="Build safe fragment libraries for GenMol molecule generation.",
        epilog="""
Library Types:
  SA, EC, CA           - Pathogen-specific (active against ONE pathogen only)
  SA_EC, SA_CA, CA_EC  - Dual-active (active against exactly TWO pathogens)
  TRIPLE               - Triple-active (active against ALL THREE pathogens)

Examples:
  python safe_library_builder.py --library SA --base-dir .
  python safe_library_builder.py --all --base-dir .
        """
    )
    parser.add_argument(
        "--library",
        choices=LIBRARY_TYPES,
        help="Library type to build"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Build all 7 libraries"
    )
    parser.add_argument(
        "--base-dir",
        default=".",
        help="Base directory (default: current directory)"
    )

    args = parser.parse_args()

    if not args.library and not args.all:
        parser.error("Must specify --library or --all")

    # Determine libraries to build
    libraries = LIBRARY_TYPES if args.all else [args.library]

    print("=" * 60)
    print("SAFE LIBRARY BUILDER")
    print("=" * 60)

    for lib_type in libraries:
        print(f"\nBuilding {lib_type} library...")

        # Build library
        fragments, diagnostics = build_safe_library(lib_type, args.base_dir)

        if not fragments:
            print(f"  WARNING: No fragments loaded for {lib_type}")
            continue

        # Output path
        out_path = os.path.join(
            args.base_dir,
            "libraries",
            f"{lib_type}_library",
            f"safe_library_{lib_type}.json"
        )

        # Write library
        write_safe_library(out_path, fragments)

        # Write diagnostics
        diag_path = os.path.join(
            args.base_dir,
            "libraries",
            f"{lib_type}_library",
            f"{lib_type}_builder_diagnostics.txt"
        )
        os.makedirs(os.path.dirname(diag_path), exist_ok=True)
        with open(diag_path, "w", encoding="utf-8") as f:
            f.write(f"Safe Library Builder Diagnostics: {lib_type}\n")
            f.write(f"Generated: {datetime.utcnow().isoformat()}\n\n")
            for line in diagnostics:
                f.write(line + "\n")
            f.write(f"\nTotal fragments: {len(fragments)}\n")

        # Count by role
        scaffolds = sum(1 for f in fragments if f.get("role") == "scaffold")
        substituents = sum(1 for f in fragments if f.get("role") == "substituent")

        print(f"  Total: {len(fragments)} fragments")
        print(f"  Scaffolds: {scaffolds}, Substituents: {substituents}")
        print(f"  Output: {out_path}")

    print("\n" + "=" * 60)
    print("BUILD COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
