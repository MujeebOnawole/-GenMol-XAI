#!/usr/bin/env python3
"""
Novelty Verification Script
Verifies that generated molecules are novel (not present in training data)

Compares generated molecules from prediction files against training data.
Uses InChIKey for robust structural comparison.

Usage:
    python verify_novelty.py --base-dir . --training-dir /path/to/training/data

Arguments:
    --base-dir: Base directory containing data/predictions/ folder
    --training-dir: Directory containing training CSVs (S_aureus_input.csv, etc.)
    --output-dir: Output directory for results (default: base-dir/results)
"""

import pandas as pd
import argparse
from pathlib import Path
from rdkit import Chem
from rdkit.Chem.inchi import MolToInchiKey


def smiles_to_inchikey(smiles):
    """Convert SMILES to InChIKey for robust comparison."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        # Remove atom mapping if present
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        return MolToInchiKey(mol)
    except:
        return None


def canonical_smiles(smiles):
    """Get canonical SMILES without atom mapping."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        return Chem.MolToSmiles(mol, canonical=True)
    except:
        return None


def load_training_data(training_dir):
    """Load training data from all three model input files."""
    training_files = {
        'SA': 'S_aureus_input.csv',
        'EC': 'E_coli_input.csv',
        'CA': 'C_albicans_input.csv'
    }

    all_training_smiles = set()
    all_training_inchikeys = set()

    for model, filename in training_files.items():
        filepath = Path(training_dir) / filename
        if not filepath.exists():
            print(f"  WARNING: {filename} not found in {training_dir}")
            continue

        df = pd.read_csv(filepath)
        print(f"  {model}: {len(df)} compounds from {filename}")

        # Find SMILES column
        smiles_col = None
        for col in ['SMILES', 'smiles', 'Smiles', 'canonical_smiles']:
            if col in df.columns:
                smiles_col = col
                break

        if smiles_col is None:
            print(f"    WARNING: No SMILES column found")
            continue

        # Collect canonical SMILES and InChIKeys
        for smiles in df[smiles_col].dropna():
            can_smi = canonical_smiles(str(smiles))
            if can_smi:
                all_training_smiles.add(can_smi)

            inchikey = smiles_to_inchikey(str(smiles))
            if inchikey:
                all_training_inchikeys.add(inchikey)

    return all_training_smiles, all_training_inchikeys


def load_generated_molecules(base_dir):
    """Load generated molecules from any prediction file."""
    predictions_dir = Path(base_dir) / 'data' / 'predictions'

    # Use SA prediction file (all have same COMPOUND_ID and SMILES)
    pred_file = predictions_dir / 'genmol_SA_predictions.csv'
    if not pred_file.exists():
        # Try alternative naming
        pred_file = predictions_dir / 'genmol_all_input_SA_prediction.csv'

    if not pred_file.exists():
        raise FileNotFoundError(f"No prediction file found in {predictions_dir}")

    df = pd.read_csv(pred_file)
    print(f"  Loaded {len(df)} generated molecules from {pred_file.name}")
    return df


def verify_novelty(gen_df, training_smiles, training_inchikeys):
    """Check each generated molecule for novelty."""
    results = []
    novel_count = 0
    duplicate_count = 0
    failed_count = 0
    duplicates = []

    for _, row in gen_df.iterrows():
        compound_id = row['COMPOUND_ID']
        smiles = row['SMILES']

        can_smi = canonical_smiles(smiles)
        inchikey = smiles_to_inchikey(smiles)

        if can_smi is None or inchikey is None:
            failed_count += 1
            results.append({
                'COMPOUND_ID': compound_id,
                'canonical_smiles': can_smi,
                'inchikey': inchikey,
                'is_novel': None,
                'match_type': 'CONVERSION_FAILED'
            })
            continue

        # Check against training data
        found_in = []
        if inchikey in training_inchikeys:
            found_in.append('InChIKey')
        if can_smi in training_smiles:
            found_in.append('SMILES')

        is_novel = len(found_in) == 0

        if is_novel:
            novel_count += 1
        else:
            duplicate_count += 1
            duplicates.append({
                'COMPOUND_ID': compound_id,
                'smiles': can_smi,
                'inchikey': inchikey,
                'match_type': ', '.join(found_in)
            })

        results.append({
            'COMPOUND_ID': compound_id,
            'canonical_smiles': can_smi,
            'inchikey': inchikey,
            'is_novel': is_novel,
            'match_type': ', '.join(found_in) if found_in else 'NOVEL'
        })

    return results, duplicates, novel_count, duplicate_count, failed_count


def main():
    parser = argparse.ArgumentParser(description='Verify novelty of generated molecules')
    parser.add_argument('--base-dir', default='.', help='Base directory with data/predictions/')
    parser.add_argument('--training-dir', required=True, help='Directory with training CSVs')
    parser.add_argument('--output-dir', default=None, help='Output directory (default: base-dir/results)')
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    training_dir = Path(args.training_dir)
    output_dir = Path(args.output_dir) if args.output_dir else base_dir / 'results'
    output_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("NOVELTY VERIFICATION")
    print("=" * 70)

    # Load training data
    print("\n1. Loading training data...")
    training_smiles, training_inchikeys = load_training_data(training_dir)
    print(f"\n  Unique training SMILES: {len(training_smiles)}")
    print(f"  Unique training InChIKeys: {len(training_inchikeys)}")

    # Load generated molecules
    print("\n2. Loading generated molecules...")
    gen_df = load_generated_molecules(base_dir)

    # Verify novelty
    print("\n3. Checking novelty...")
    results, duplicates, novel, dup, failed = verify_novelty(
        gen_df, training_smiles, training_inchikeys
    )

    # Print summary
    total = len(gen_df)
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nTotal generated molecules: {total}")
    print(f"Novel molecules:           {novel} ({100*novel/total:.1f}%)")
    print(f"Duplicates (in training):  {dup} ({100*dup/total:.1f}%)")
    print(f"Failed conversion:         {failed}")

    # Save results
    results_df = pd.DataFrame(results)
    results_path = output_dir / 'novelty_verification_results.csv'
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")

    if duplicates:
        print(f"\n--- DUPLICATES ({len(duplicates)}) ---")
        for d in duplicates[:5]:
            print(f"  {d['COMPOUND_ID']}: {d['smiles'][:50]}...")

        dup_df = pd.DataFrame(duplicates)
        dup_path = output_dir / 'novelty_duplicates.csv'
        dup_df.to_csv(dup_path, index=False)
        print(f"\nDuplicates saved to: {dup_path}")
    else:
        print("\n*** ALL MOLECULES ARE NOVEL ***")

    return results_df


if __name__ == '__main__':
    main()
