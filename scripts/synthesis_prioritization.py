"""
Synthesis Prioritization Analysis
=================================
1. Analyze identical molecules (sim=1.0) - verify predictions match training
2. Extract "Goldilocks" compounds (0.3-0.4 similarity) - novel yet reliable
3. Prioritize Scenario A actives for synthesis

Author: Generated for GenMol XAI analysis
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up paths
BASE_DIR = Path(r"C:\Users\uqaonawo\OneDrive - The University of Queensland\Desktop\Fragment_XAI_analysis\FBDD\-GenMol-XAI-main")
DATA_DIR = BASE_DIR / "data"
SOURCE_DIR = DATA_DIR / "source_data"
RESULTS_DIR = BASE_DIR / "results" / "similarity_analysis"
PRIORITY_DIR = BASE_DIR / "results" / "synthesis_prioritization"
PRIORITY_DIR.mkdir(parents=True, exist_ok=True)

print("="*70)
print("SYNTHESIS PRIORITIZATION ANALYSIS")
print("="*70)

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n[1] Loading data...")

# Load similarity results
sim_df = pd.read_csv(RESULTS_DIR / "similarity_to_training.csv")
print(f"   Similarity data: {len(sim_df)} compounds")

# Load generated molecules
genmol_df = pd.read_csv(DATA_DIR / "genmol_input.csv")
print(f"   Generated molecules: {len(genmol_df)}")

# Load predictions for all pathogens
predictions = {}
for pathogen in ['SA', 'EC', 'CA']:
    pred_file = DATA_DIR / "predictions" / f"genmol_all_input_{pathogen}_prediction.csv"
    if pred_file.exists():
        predictions[pathogen] = pd.read_csv(pred_file)
        print(f"   {pathogen} predictions: {len(predictions[pathogen])}")

# Load training data
training_data = {}
for pathogen, filename in [('SA', 'S_aureus_input.csv'),
                           ('EC', 'E_coli_input.csv'),
                           ('CA', 'C_albicans_input.csv')]:
    df = pd.read_csv(SOURCE_DIR / filename)
    training_data[pathogen] = df
    print(f"   {pathogen} training: {len(df)} compounds")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def clean_smiles(smiles):
    """Remove atom mapping from SMILES"""
    if pd.isna(smiles):
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        return Chem.MolToSmiles(mol)
    except:
        return None

def get_morgan_fp(smiles, radius=2, n_bits=2048):
    """Generate Morgan fingerprint from SMILES"""
    if pd.isna(smiles):
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    except:
        return None

def find_most_similar_training(query_smiles, training_df, pathogen):
    """Find the most similar training compound and return its details"""
    query_fp = get_morgan_fp(clean_smiles(query_smiles))
    if query_fp is None:
        return None, None, None

    best_sim = 0
    best_compound = None
    best_activity = None

    for idx, row in training_df.iterrows():
        train_fp = get_morgan_fp(row['SMILES'])
        if train_fp is not None:
            sim = DataStructs.TanimotoSimilarity(query_fp, train_fp)
            if sim > best_sim:
                best_sim = sim
                best_compound = row['COMPOUND_ID']
                best_activity = row.get('TARGET', None)

    return best_sim, best_compound, best_activity

# ============================================================================
# PART 1: ANALYZE IDENTICAL MOLECULES (Similarity = 1.0)
# ============================================================================

print("\n[2] Analyzing identical molecules (similarity >= 0.99)...")

identical_results = []

for pathogen in ['SA', 'EC', 'CA']:
    col = f'max_sim_{pathogen}_all'

    # Find compounds with very high similarity (>=0.99, essentially identical)
    high_sim = sim_df[sim_df[col] >= 0.99].copy()

    if len(high_sim) > 0:
        print(f"\n   {pathogen}: Found {len(high_sim)} compounds with similarity >= 0.99")

        # Get predictions for these compounds
        pred_df = predictions[pathogen]
        train_df = training_data[pathogen]

        for idx, row in high_sim.iterrows():
            cmpd_id = row['COMPOUND_ID']
            gen_smiles = row['clean_smiles']

            # Get prediction for this compound
            pred_row = pred_df[pred_df['COMPOUND_ID'] == cmpd_id]
            if len(pred_row) > 0:
                pred_row = pred_row.iloc[0]
                gen_prediction = pred_row['prediction']
                gen_prob = pred_row['ensemble_prediction']
                scenario = pred_row['decision_scenario']
            else:
                gen_prediction = None
                gen_prob = None
                scenario = None

            # Find matching training compound
            sim, train_cmpd, train_activity = find_most_similar_training(gen_smiles, train_df, pathogen)

            identical_results.append({
                'COMPOUND_ID': cmpd_id,
                'pathogen': pathogen,
                'similarity': row[col],
                'generated_smiles': gen_smiles,
                'generated_prediction': gen_prediction,
                'generated_probability': gen_prob,
                'decision_scenario': scenario,
                'matching_training_compound': train_cmpd,
                'training_activity': train_activity,
                'prediction_matches_training': gen_prediction == train_activity if (gen_prediction is not None and train_activity is not None) else None
            })

identical_df = pd.DataFrame(identical_results)

if len(identical_df) > 0:
    identical_df.to_csv(PRIORITY_DIR / "identical_molecules_analysis.csv", index=False)
    print(f"\n   Total identical molecules: {len(identical_df)}")

    # Summary
    matches = identical_df['prediction_matches_training'].sum()
    total_with_data = identical_df['prediction_matches_training'].notna().sum()
    print(f"   Predictions matching training activity: {matches}/{total_with_data}")

    # Show details
    print("\n   Details of identical molecules:")
    print(identical_df[['COMPOUND_ID', 'pathogen', 'similarity', 'generated_prediction',
                        'training_activity', 'prediction_matches_training']].to_string())
else:
    print("   No identical molecules found")

# ============================================================================
# PART 2: GOLDILOCKS COMPOUNDS (0.3-0.4 similarity, Scenario A, Active)
# ============================================================================

print("\n" + "="*70)
print("[3] Extracting GOLDILOCKS compounds (0.3-0.4 similarity, Scenario A actives)")
print("="*70)

print("\n   Goldilocks criteria:")
print("   - Similarity 0.3-0.4: Novel enough to be interesting, similar enough for reliable prediction")
print("   - Scenario A (Tier 3): High agreement + High reliability")
print("   - Predicted Active: Model predicts antimicrobial activity")

goldilocks_results = []

for pathogen in ['SA', 'EC', 'CA']:
    col_all = f'max_sim_{pathogen}_all'
    col_active = f'max_sim_{pathogen}_active'
    col_inactive = f'max_sim_{pathogen}_inactive'

    pred_df = predictions[pathogen]

    # Merge similarity with predictions
    merged = sim_df[['COMPOUND_ID', 'clean_smiles', col_all, col_active, col_inactive]].merge(
        pred_df[['COMPOUND_ID', 'SMILES', 'ensemble_prediction', 'prediction',
                 'decision_scenario', 'reliability_status', 'ensemble_agreement']],
        on='COMPOUND_ID', how='inner'
    )

    # Also merge with genmol for source library info
    merged = merged.merge(
        genmol_df[['COMPOUND_ID', 'source_library', 'product_linkage_type', 'SA_score',
                   'MW', 'LogP', 'TPSA', 'HBD', 'HBA']],
        on='COMPOUND_ID', how='left'
    )

    # Apply Goldilocks criteria
    goldilocks = merged[
        (merged[col_all] >= 0.30) &
        (merged[col_all] <= 0.45) &  # Slightly expanded range
        (merged['decision_scenario'] == 'A') &
        (merged['prediction'] == 1)
    ].copy()

    goldilocks['pathogen_model'] = pathogen
    goldilocks['similarity_to_training'] = goldilocks[col_all]
    goldilocks['similarity_to_actives'] = goldilocks[col_active]
    goldilocks['similarity_to_inactives'] = goldilocks[col_inactive]

    # Calculate "desirability score" - balance of novelty and activity confidence
    # Higher similarity to actives vs inactives is better
    goldilocks['active_inactive_ratio'] = goldilocks[col_active] / (goldilocks[col_inactive] + 0.01)

    goldilocks_results.append(goldilocks)

    print(f"\n   {pathogen} Goldilocks compounds: {len(goldilocks)}")

# Combine all
all_goldilocks = pd.concat(goldilocks_results, ignore_index=True)

print(f"\n   Total Goldilocks compounds: {len(all_goldilocks)}")

if len(all_goldilocks) > 0:
    # Select and reorder columns for output
    output_cols = [
        'COMPOUND_ID', 'pathogen_model', 'source_library', 'product_linkage_type',
        'clean_smiles', 'similarity_to_training', 'similarity_to_actives',
        'similarity_to_inactives', 'active_inactive_ratio',
        'ensemble_prediction', 'decision_scenario', 'reliability_status',
        'MW', 'LogP', 'TPSA', 'HBD', 'HBA', 'SA_score'
    ]

    # Filter to available columns
    output_cols = [c for c in output_cols if c in all_goldilocks.columns]

    goldilocks_output = all_goldilocks[output_cols].copy()

    # Sort by probability (highest first) then by active/inactive ratio
    goldilocks_output = goldilocks_output.sort_values(
        ['pathogen_model', 'ensemble_prediction', 'active_inactive_ratio'],
        ascending=[True, False, False]
    )

    goldilocks_output.to_csv(PRIORITY_DIR / "goldilocks_compounds.csv", index=False)
    print(f"   Saved: goldilocks_compounds.csv")

    # Summary by pathogen
    print("\n   Summary by pathogen:")
    for pathogen in ['SA', 'EC', 'CA']:
        subset = goldilocks_output[goldilocks_output['pathogen_model'] == pathogen]
        if len(subset) > 0:
            print(f"   {pathogen}: {len(subset)} compounds")
            print(f"      Mean similarity: {subset['similarity_to_training'].mean():.3f}")
            print(f"      Mean probability: {subset['ensemble_prediction'].mean():.3f}")
            print(f"      Libraries: {subset['source_library'].value_counts().to_dict()}")

# ============================================================================
# PART 3: EXPANDED PRIORITIZATION - ALL SCENARIO A ACTIVES
# ============================================================================

print("\n" + "="*70)
print("[4] Complete Synthesis Prioritization - All Scenario A Actives")
print("="*70)

priority_results = []

for pathogen in ['SA', 'EC', 'CA']:
    col_all = f'max_sim_{pathogen}_all'
    col_active = f'max_sim_{pathogen}_active'
    col_inactive = f'max_sim_{pathogen}_inactive'

    pred_df = predictions[pathogen]

    # Merge similarity with predictions
    merged = sim_df[['COMPOUND_ID', 'clean_smiles', col_all, col_active, col_inactive]].merge(
        pred_df[['COMPOUND_ID', 'ensemble_prediction', 'prediction',
                 'decision_scenario', 'reliability_status']],
        on='COMPOUND_ID', how='inner'
    )

    # Merge with genmol
    merged = merged.merge(
        genmol_df[['COMPOUND_ID', 'source_library', 'product_linkage_type', 'SA_score',
                   'MW', 'LogP', 'TPSA']],
        on='COMPOUND_ID', how='left'
    )

    # Get all Scenario A actives
    scenario_a_actives = merged[
        (merged['decision_scenario'] == 'A') &
        (merged['prediction'] == 1)
    ].copy()

    scenario_a_actives['pathogen_model'] = pathogen
    scenario_a_actives['similarity_to_training'] = scenario_a_actives[col_all]

    # Assign priority tier based on similarity
    def assign_priority(sim):
        if 0.30 <= sim <= 0.45:
            return 1  # Goldilocks - top priority
        elif 0.45 < sim <= 0.60:
            return 2  # Good - slightly higher similarity, still novel
        elif 0.20 <= sim < 0.30:
            return 3  # Very novel - may need more validation
        elif sim > 0.60:
            return 4  # High similarity - less novel but reliable
        else:
            return 5  # Very low similarity - highest risk

    scenario_a_actives['priority_tier'] = scenario_a_actives['similarity_to_training'].apply(assign_priority)

    priority_results.append(scenario_a_actives)

all_priority = pd.concat(priority_results, ignore_index=True)

print(f"\n   Total Scenario A actives across all models: {len(all_priority)}")

# Priority tier summary
print("\n   Priority Tier Definitions:")
print("   Tier 1 (Goldilocks, 0.30-0.45): Novel + Reliable - TOP PRIORITY")
print("   Tier 2 (0.45-0.60): Moderately novel, good reliability")
print("   Tier 3 (0.20-0.30): Very novel, may need extra validation")
print("   Tier 4 (>0.60): Less novel but highly reliable")
print("   Tier 5 (<0.20): Very dissimilar, highest uncertainty")

print("\n   Distribution by Priority Tier:")
tier_counts = all_priority.groupby(['pathogen_model', 'priority_tier']).size().unstack(fill_value=0)
print(tier_counts.to_string())

# Save complete prioritization
priority_output_cols = [
    'COMPOUND_ID', 'pathogen_model', 'priority_tier', 'source_library',
    'product_linkage_type', 'clean_smiles', 'similarity_to_training',
    'ensemble_prediction', 'decision_scenario', 'reliability_status',
    'MW', 'LogP', 'TPSA', 'SA_score'
]
priority_output_cols = [c for c in priority_output_cols if c in all_priority.columns]

all_priority_sorted = all_priority[priority_output_cols].sort_values(
    ['pathogen_model', 'priority_tier', 'ensemble_prediction'],
    ascending=[True, True, False]
)

all_priority_sorted.to_csv(PRIORITY_DIR / "synthesis_priority_all_scenario_a.csv", index=False)
print(f"\n   Saved: synthesis_priority_all_scenario_a.csv")

# ============================================================================
# PART 4: TOP PICKS FOR EACH PATHOGEN
# ============================================================================

print("\n" + "="*70)
print("[5] TOP 10 Synthesis Candidates per Pathogen (Tier 1 Goldilocks)")
print("="*70)

for pathogen in ['SA', 'EC', 'CA']:
    subset = all_priority[
        (all_priority['pathogen_model'] == pathogen) &
        (all_priority['priority_tier'] == 1)
    ].sort_values('ensemble_prediction', ascending=False).head(10)

    print(f"\n   === {pathogen} TOP 10 ===")
    if len(subset) > 0:
        for i, (idx, row) in enumerate(subset.iterrows(), 1):
            print(f"   {i}. {row['COMPOUND_ID']} | Sim: {row['similarity_to_training']:.3f} | "
                  f"Prob: {row['ensemble_prediction']:.3f} | SA_score: {row.get('SA_score', 'N/A'):.2f} | "
                  f"Library: {row['source_library']}")
    else:
        print(f"   No Tier 1 compounds for {pathogen}")

# Save top picks
top_picks = []
for pathogen in ['SA', 'EC', 'CA']:
    subset = all_priority[
        (all_priority['pathogen_model'] == pathogen) &
        (all_priority['priority_tier'] == 1)
    ].sort_values('ensemble_prediction', ascending=False).head(10)
    top_picks.append(subset)

if top_picks:
    top_picks_df = pd.concat(top_picks, ignore_index=True)
    top_picks_df.to_csv(PRIORITY_DIR / "top_synthesis_candidates.csv", index=False)
    print(f"\n   Saved: top_synthesis_candidates.csv")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("SYNTHESIS PRIORITIZATION SUMMARY")
print("="*70)

print(f"""
Files Generated in {PRIORITY_DIR}:

1. identical_molecules_analysis.csv
   - Compounds with similarity >= 0.99 to training
   - Verification of prediction vs training activity

2. goldilocks_compounds.csv
   - Similarity 0.30-0.45 (novel yet reliable)
   - Scenario A (Tier 3) with high confidence
   - Predicted active

3. synthesis_priority_all_scenario_a.csv
   - All Scenario A actives with priority tiers
   - Tier 1 = Goldilocks (top priority)

4. top_synthesis_candidates.csv
   - Top 10 Tier 1 candidates per pathogen
   - Ready for synthesis prioritization

Key Numbers:
- Total Scenario A actives: {len(all_priority)}
- Tier 1 (Goldilocks): {len(all_priority[all_priority['priority_tier'] == 1])}
- Tier 2: {len(all_priority[all_priority['priority_tier'] == 2])}
""")

print("="*70)
print("ANALYSIS COMPLETE")
print("="*70)
