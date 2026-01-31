"""
Similarity Analysis: Generated Molecules vs Training Data
=========================================================
Addresses boss's feedback:
1. Similarity distribution by pathogen model (CA vs EC vs SA)
2. CA inactive similarity check for potential filtering
3. Azole coupling site audit
4. Fragment size vs similarity correlation
5. Amide vs urea reaction type comparison

Author: Generated for GenMol XAI analysis
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit import DataStructs
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set up paths
BASE_DIR = Path(r"C:\Users\uqaonawo\OneDrive - The University of Queensland\Desktop\Fragment_XAI_analysis\FBDD\-GenMol-XAI-main")
DATA_DIR = BASE_DIR / "data"
SOURCE_DIR = DATA_DIR / "source_data"
RESULTS_DIR = BASE_DIR / "results" / "similarity_analysis"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("="*70)
print("SIMILARITY ANALYSIS: Generated Molecules vs Training Data")
print("="*70)

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

def calculate_max_similarity(query_fp, reference_fps):
    """Calculate maximum Tanimoto similarity to any reference molecule"""
    if query_fp is None:
        return np.nan
    max_sim = 0
    for ref_fp in reference_fps:
        if ref_fp is not None:
            sim = DataStructs.TanimotoSimilarity(query_fp, ref_fp)
            if sim > max_sim:
                max_sim = sim
    return max_sim

def calculate_mean_similarity(query_fp, reference_fps):
    """Calculate mean Tanimoto similarity to reference molecules"""
    if query_fp is None:
        return np.nan
    sims = []
    for ref_fp in reference_fps:
        if ref_fp is not None:
            sims.append(DataStructs.TanimotoSimilarity(query_fp, ref_fp))
    return np.mean(sims) if sims else np.nan

def contains_azole(smiles):
    """Check if molecule contains azole heterocycle (triazole, imidazole, etc.)"""
    if pd.isna(smiles):
        return False
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        # Common azole SMARTS patterns
        azole_patterns = [
            'c1cn[nH]c1',     # imidazole
            'c1c[nH]cn1',     # imidazole (alt)
            'c1nncn1',        # 1,2,4-triazole
            'c1n[nH]cn1',     # 1,2,4-triazole
            'c1ncnc1',        # pyrimidine-like
            'c1nnn[nH]1',     # tetrazole
            '[nR1]1[cR1][nR1][cR1][cR1]1',  # 5-membered with 2N
            '[nR1]1[cR1][cR1][nR1][cR1]1',  # alternate
        ]
        for pattern in azole_patterns:
            patt = Chem.MolFromSmarts(pattern)
            if patt and mol.HasSubstructMatch(patt):
                return True
        return False
    except:
        return False

def get_aromatic_n_count(smiles):
    """Count aromatic nitrogens in molecule"""
    if pd.isna(smiles):
        return 0
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0
        count = 0
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'N' and atom.GetIsAromatic():
                count += 1
        return count
    except:
        return 0

def get_aliphatic_amine_count(smiles):
    """Count aliphatic amines (non-aromatic N with H)"""
    if pd.isna(smiles):
        return 0
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0
        # SMARTS for aliphatic amine (N with at least one H, not aromatic)
        amine_patt = Chem.MolFromSmarts('[NX3;H1,H2;!$(N-C=O);!a]')
        if amine_patt:
            matches = mol.GetSubstructMatches(amine_patt)
            return len(matches)
        return 0
    except:
        return 0

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n[1] Loading data...")

# Load generated molecules
genmol_df = pd.read_csv(DATA_DIR / "genmol_input.csv")
print(f"   Generated molecules: {len(genmol_df)}")

# Clean SMILES (remove atom mapping)
genmol_df['clean_smiles'] = genmol_df['SMILES'].apply(clean_smiles)

# Load training data
training_data = {}
for pathogen, filename in [('SA', 'S_aureus_input.csv'),
                           ('EC', 'E_coli_input.csv'),
                           ('CA', 'C_albicans_input.csv')]:
    df = pd.read_csv(SOURCE_DIR / filename)
    df['clean_smiles'] = df['SMILES'].apply(clean_smiles)
    training_data[pathogen] = df
    actives = df[df['TARGET'] == 1] if 'TARGET' in df.columns else df
    inactives = df[df['TARGET'] == 0] if 'TARGET' in df.columns else pd.DataFrame()
    print(f"   {pathogen} training: {len(df)} total, {len(actives)} actives, {len(inactives)} inactives")

# Load prediction files for Tier classification (correct naming convention)
predictions = {}
for pathogen in ['SA', 'EC', 'CA']:
    pred_file = DATA_DIR / "predictions" / f"genmol_all_input_{pathogen}_prediction.csv"
    if pred_file.exists():
        predictions[pathogen] = pd.read_csv(pred_file)
        print(f"   {pathogen} predictions: {len(predictions[pathogen])}")

# ============================================================================
# CALCULATE FINGERPRINTS
# ============================================================================

print("\n[2] Calculating fingerprints...")

# Generated molecules fingerprints
genmol_df['fp'] = genmol_df['clean_smiles'].apply(get_morgan_fp)
valid_fps = genmol_df['fp'].notna().sum()
print(f"   Generated molecules with valid FPs: {valid_fps}/{len(genmol_df)}")

# Training fingerprints (separated by active/inactive)
training_fps = {}
for pathogen, df in training_data.items():
    df['fp'] = df['clean_smiles'].apply(get_morgan_fp)

    # All training compounds
    all_fps = [fp for fp in df['fp'].dropna().tolist()]

    # Active compounds
    if 'TARGET' in df.columns:
        active_fps = [fp for fp in df[df['TARGET'] == 1]['fp'].dropna().tolist()]
        inactive_fps = [fp for fp in df[df['TARGET'] == 0]['fp'].dropna().tolist()]
    else:
        active_fps = all_fps
        inactive_fps = []

    training_fps[pathogen] = {
        'all': all_fps,
        'active': active_fps,
        'inactive': inactive_fps
    }
    print(f"   {pathogen} fingerprints: {len(all_fps)} all, {len(active_fps)} active, {len(inactive_fps)} inactive")

# ============================================================================
# TASK 1: SIMILARITY DISTRIBUTION BY SOURCE LIBRARY AND PATHOGEN
# ============================================================================

print("\n[3] Task 1: Calculating similarity distributions...")

# For each generated molecule, calculate max similarity to each training set
similarity_results = []

for idx, row in genmol_df.iterrows():
    if idx % 100 == 0:
        print(f"   Processing molecule {idx+1}/{len(genmol_df)}...")

    fp = row['fp']
    result = {
        'COMPOUND_ID': row['COMPOUND_ID'],
        'source_library': row['source_library'],
        'product_linkage_type': row['product_linkage_type'],
        'clean_smiles': row['clean_smiles']
    }

    # Calculate similarity to each pathogen's training set
    for pathogen in ['SA', 'EC', 'CA']:
        result[f'max_sim_{pathogen}_all'] = calculate_max_similarity(fp, training_fps[pathogen]['all'])
        result[f'max_sim_{pathogen}_active'] = calculate_max_similarity(fp, training_fps[pathogen]['active'])
        result[f'max_sim_{pathogen}_inactive'] = calculate_max_similarity(fp, training_fps[pathogen]['inactive'])

    similarity_results.append(result)

sim_df = pd.DataFrame(similarity_results)

# Save raw results
sim_df.to_csv(RESULTS_DIR / "similarity_to_training.csv", index=False)
print(f"   Saved: similarity_to_training.csv")

# ============================================================================
# TASK 1 VISUALIZATION: Distribution plots
# ============================================================================

print("\n[4] Creating distribution plots...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Row 1: Similarity to each pathogen's training (by source library)
for i, pathogen in enumerate(['SA', 'EC', 'CA']):
    ax = axes[0, i]
    for lib in sim_df['source_library'].unique():
        lib_data = sim_df[sim_df['source_library'] == lib][f'max_sim_{pathogen}_all']
        ax.hist(lib_data, alpha=0.5, label=lib, bins=30, density=True)
    ax.set_xlabel('Max Tanimoto Similarity')
    ax.set_ylabel('Density')
    ax.set_title(f'Similarity to {pathogen} Training Set')
    ax.legend(fontsize=8)
    ax.axvline(x=0.7, color='red', linestyle='--', alpha=0.7, label='High similarity (0.7)')

# Row 2: Comparison across pathogens (aggregate)
colors = {'SA': '#DC143C', 'EC': '#1E90FF', 'CA': '#228B22'}
ax = axes[1, 0]
for pathogen in ['SA', 'EC', 'CA']:
    data = sim_df[f'max_sim_{pathogen}_all'].dropna()
    ax.hist(data, alpha=0.5, label=pathogen, bins=30, density=True, color=colors[pathogen])
ax.set_xlabel('Max Tanimoto Similarity')
ax.set_ylabel('Density')
ax.set_title('Generated Molecules: Similarity to Each Training Set')
ax.legend()

# Summary statistics by source library
ax = axes[1, 1]
lib_means = sim_df.groupby('source_library')[['max_sim_SA_all', 'max_sim_EC_all', 'max_sim_CA_all']].mean()
lib_means.plot(kind='bar', ax=ax, color=[colors['SA'], colors['EC'], colors['CA']])
ax.set_ylabel('Mean Max Similarity')
ax.set_title('Mean Similarity by Source Library')
ax.legend(['SA', 'EC', 'CA'])
ax.tick_params(axis='x', rotation=45)

# Similarity correlation between pathogens
ax = axes[1, 2]
ax.scatter(sim_df['max_sim_SA_all'], sim_df['max_sim_EC_all'], alpha=0.5, label='SA vs EC', s=10)
ax.scatter(sim_df['max_sim_SA_all'], sim_df['max_sim_CA_all'], alpha=0.5, label='SA vs CA', s=10)
ax.set_xlabel('Similarity to SA Training')
ax.set_ylabel('Similarity to EC/CA Training')
ax.set_title('Cross-Pathogen Similarity Correlation')
ax.legend()
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "similarity_distributions.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: similarity_distributions.png")

# ============================================================================
# TASK 2: CA INACTIVE SIMILARITY CHECK
# ============================================================================

print("\n[5] Task 2: CA inactive similarity analysis...")

# Get CA predictions
if 'CA' in predictions:
    ca_pred = predictions['CA'].copy()
    # Merge with similarity data
    ca_analysis = ca_pred.merge(sim_df[['COMPOUND_ID', 'max_sim_CA_all', 'max_sim_CA_active', 'max_sim_CA_inactive']],
                                 on='COMPOUND_ID', how='left')

    # Classify by prediction
    if 'prediction' in ca_analysis.columns:
        predicted_active = ca_analysis[ca_analysis['prediction'] == 1]
        predicted_inactive = ca_analysis[ca_analysis['prediction'] == 0]

        print(f"   CA predicted active: {len(predicted_active)}")
        print(f"   CA predicted inactive: {len(predicted_inactive)}")

        # Check if predicted inactives are similar to training inactives
        inactive_high_sim = predicted_inactive[predicted_inactive['max_sim_CA_inactive'] > 0.5]
        print(f"   Predicted inactives with >0.5 similarity to training inactives: {len(inactive_high_sim)}")

        # These could be True Negatives - candidates for filtering
        inactive_high_sim.to_csv(RESULTS_DIR / "ca_potential_true_negatives.csv", index=False)

        # Check for potential False Negatives (predicted inactive but similar to training actives)
        false_neg_candidates = predicted_inactive[
            (predicted_inactive['max_sim_CA_active'] > 0.6) &
            (predicted_inactive['max_sim_CA_inactive'] < 0.5)
        ]
        print(f"   Potential False Negatives (high sim to actives, low to inactives): {len(false_neg_candidates)}")
        false_neg_candidates.to_csv(RESULTS_DIR / "ca_potential_false_negatives.csv", index=False)

        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        ax = axes[0]
        ax.scatter(predicted_active['max_sim_CA_active'], predicted_active['max_sim_CA_inactive'],
                   alpha=0.6, label='Predicted Active', c='green', s=30)
        ax.scatter(predicted_inactive['max_sim_CA_active'], predicted_inactive['max_sim_CA_inactive'],
                   alpha=0.6, label='Predicted Inactive', c='red', s=30)
        ax.set_xlabel('Similarity to CA Active Training')
        ax.set_ylabel('Similarity to CA Inactive Training')
        ax.set_title('CA Predictions: Active vs Inactive Similarity')
        ax.legend()
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=0.6, color='gray', linestyle='--', alpha=0.5)

        ax = axes[1]
        # Distribution of similarity to inactives
        ax.hist(predicted_active['max_sim_CA_inactive'].dropna(), alpha=0.5,
                label='Predicted Active', bins=20, color='green', density=True)
        ax.hist(predicted_inactive['max_sim_CA_inactive'].dropna(), alpha=0.5,
                label='Predicted Inactive', bins=20, color='red', density=True)
        ax.set_xlabel('Similarity to CA Inactive Training')
        ax.set_ylabel('Density')
        ax.set_title('Similarity to Inactives Distribution')
        ax.legend()

        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "ca_inactive_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   Saved: ca_inactive_analysis.png")

# ============================================================================
# TASK 3: AZOLE COUPLING SITE AUDIT
# ============================================================================

print("\n[6] Task 3: Azole coupling site audit...")

# Check amine fragments for azole content and coupling site
azole_analysis = []

for idx, row in genmol_df.iterrows():
    amine_smiles = row.get('amine_fragment_smiles', '')
    if pd.isna(amine_smiles):
        continue

    has_azole = contains_azole(amine_smiles)
    aromatic_n = get_aromatic_n_count(amine_smiles)
    aliphatic_amine = get_aliphatic_amine_count(amine_smiles)

    if has_azole or aromatic_n > 0:
        azole_analysis.append({
            'COMPOUND_ID': row['COMPOUND_ID'],
            'source_library': row['source_library'],
            'amine_fragment_smiles': amine_smiles,
            'amine_fragment_id': row.get('amine_fragment_id', ''),
            'has_azole': has_azole,
            'aromatic_N_count': aromatic_n,
            'aliphatic_amine_count': aliphatic_amine,
            'coupling_likely_ok': aliphatic_amine > 0,  # Has alternative amine for coupling
            'product_linkage_type': row['product_linkage_type'],
            'handle_origin_amine': row.get('handle_origin_amine', '')
        })

azole_df = pd.DataFrame(azole_analysis)

if len(azole_df) > 0:
    print(f"   Fragments with azoles/aromatic N: {len(azole_df)}")
    print(f"   With alternative aliphatic amine (coupling OK): {azole_df['coupling_likely_ok'].sum()}")
    print(f"   Potentially problematic (no alternative amine): {(~azole_df['coupling_likely_ok']).sum()}")

    azole_df.to_csv(RESULTS_DIR / "azole_coupling_audit.csv", index=False)
    print(f"   Saved: azole_coupling_audit.csv")

    # Summary by source library
    azole_summary = azole_df.groupby('source_library').agg({
        'COMPOUND_ID': 'count',
        'coupling_likely_ok': 'sum',
        'has_azole': 'sum'
    }).rename(columns={'COMPOUND_ID': 'total', 'coupling_likely_ok': 'ok_coupling', 'has_azole': 'has_azole'})
    azole_summary['problematic'] = azole_summary['total'] - azole_summary['ok_coupling']
    print("\n   Azole Summary by Library:")
    print(azole_summary.to_string())
else:
    print("   No azole-containing amine fragments found")

# ============================================================================
# TASK 4: FRAGMENT SIZE VS SIMILARITY CORRELATION
# ============================================================================

print("\n[7] Task 4: Fragment size vs similarity analysis...")

# Calculate fragment molecular weights
def get_mw(smiles):
    if pd.isna(smiles):
        return np.nan
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Descriptors.MolWt(mol)
    except:
        pass
    return np.nan

genmol_df['acid_MW'] = genmol_df['acid_fragment_smiles'].apply(get_mw)
genmol_df['amine_MW'] = genmol_df['amine_fragment_smiles'].apply(get_mw)
genmol_df['total_fragment_MW'] = genmol_df['acid_MW'] + genmol_df['amine_MW']

# Merge with similarity
fragment_sim = genmol_df[['COMPOUND_ID', 'source_library', 'acid_MW', 'amine_MW', 'total_fragment_MW']].merge(
    sim_df[['COMPOUND_ID', 'max_sim_SA_all', 'max_sim_EC_all', 'max_sim_CA_all']], on='COMPOUND_ID'
)

# Calculate correlations
print("\n   Correlations (Fragment MW vs Max Similarity):")
for pathogen in ['SA', 'EC', 'CA']:
    corr = fragment_sim[['total_fragment_MW', f'max_sim_{pathogen}_all']].corr().iloc[0, 1]
    print(f"   Total Fragment MW vs {pathogen} similarity: r = {corr:.3f}")

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, pathogen in enumerate(['SA', 'EC', 'CA']):
    ax = axes[i]
    colors_lib = {'SA': 'red', 'EC': 'blue', 'CA': 'green',
                  'SA_EC': 'purple', 'SA_CA': 'orange', 'CA_EC': 'cyan', 'TRIPLE': 'gray'}

    for lib in fragment_sim['source_library'].unique():
        lib_data = fragment_sim[fragment_sim['source_library'] == lib]
        ax.scatter(lib_data['total_fragment_MW'], lib_data[f'max_sim_{pathogen}_all'],
                   alpha=0.5, label=lib, s=20, c=colors_lib.get(lib, 'black'))

    ax.set_xlabel('Total Fragment MW (acid + amine)')
    ax.set_ylabel(f'Max Similarity to {pathogen} Training')
    ax.set_title(f'Fragment Size vs {pathogen} Similarity')
    ax.legend(fontsize=8)

    # Add trend line
    x = fragment_sim['total_fragment_MW'].dropna()
    y = fragment_sim[f'max_sim_{pathogen}_all'].dropna()
    if len(x) > 0 and len(y) > 0:
        z = np.polyfit(x[:len(y)], y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, p(x_line), 'k--', alpha=0.5)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "fragment_size_vs_similarity.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: fragment_size_vs_similarity.png")

# ============================================================================
# TASK 5: AMIDE VS UREA (RING-OPENING) COMPARISON
# ============================================================================

print("\n[8] Task 5: Amide vs Urea reaction type comparison...")

# Merge linkage type with similarity
linkage_sim = genmol_df[['COMPOUND_ID', 'product_linkage_type', 'source_library']].merge(
    sim_df[['COMPOUND_ID', 'max_sim_SA_all', 'max_sim_EC_all', 'max_sim_CA_all']], on='COMPOUND_ID'
)

print(f"\n   Counts: Amide = {(linkage_sim['product_linkage_type'] == 'amide').sum()}, "
      f"Urea = {(linkage_sim['product_linkage_type'] == 'urea').sum()}")

# Statistical comparison
from scipy import stats

print("\n   Mean similarity by linkage type:")
for pathogen in ['SA', 'EC', 'CA']:
    amide_sim = linkage_sim[linkage_sim['product_linkage_type'] == 'amide'][f'max_sim_{pathogen}_all'].dropna()
    urea_sim = linkage_sim[linkage_sim['product_linkage_type'] == 'urea'][f'max_sim_{pathogen}_all'].dropna()

    if len(urea_sim) > 0:
        t_stat, p_val = stats.ttest_ind(amide_sim, urea_sim)
        print(f"   {pathogen}: Amide={amide_sim.mean():.3f}, Urea={urea_sim.mean():.3f}, p={p_val:.4f}")
    else:
        print(f"   {pathogen}: Amide={amide_sim.mean():.3f}, Urea=N/A")

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, pathogen in enumerate(['SA', 'EC', 'CA']):
    ax = axes[i]

    amide_data = linkage_sim[linkage_sim['product_linkage_type'] == 'amide'][f'max_sim_{pathogen}_all']
    urea_data = linkage_sim[linkage_sim['product_linkage_type'] == 'urea'][f'max_sim_{pathogen}_all']

    bp = ax.boxplot([amide_data.dropna(), urea_data.dropna()],
                     labels=['Amide', 'Urea (ring-opening)'],
                     patch_artist=True)

    bp['boxes'][0].set_facecolor('#87CEEB')
    bp['boxes'][1].set_facecolor('#FFA07A')

    ax.set_ylabel(f'Max Similarity to {pathogen} Training')
    ax.set_title(f'{pathogen}: Amide vs Urea Similarity')

plt.tight_layout()
plt.savefig(RESULTS_DIR / "amide_vs_urea_similarity.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: amide_vs_urea_similarity.png")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)

print("\n[A] Overall Similarity Distribution:")
for pathogen in ['SA', 'EC', 'CA']:
    data = sim_df[f'max_sim_{pathogen}_all'].dropna()
    print(f"   {pathogen}: Mean={data.mean():.3f}, Median={data.median():.3f}, "
          f"Std={data.std():.3f}, Min={data.min():.3f}, Max={data.max():.3f}")

print("\n[B] Similarity by Source Library (to own pathogen training):")
library_mapping = {
    'SA': 'SA', 'EC': 'EC', 'CA': 'CA',
    'SA_EC': 'SA', 'SA_CA': 'SA', 'CA_EC': 'CA', 'TRIPLE': 'SA'
}
for lib in sim_df['source_library'].unique():
    primary = library_mapping.get(lib, 'SA')
    lib_data = sim_df[sim_df['source_library'] == lib][f'max_sim_{primary}_all']
    print(f"   {lib} -> {primary}: Mean={lib_data.mean():.3f}, Std={lib_data.std():.3f}")

print("\n[C] High Similarity Compounds (>0.7 to any training):")
high_sim = sim_df[(sim_df['max_sim_SA_all'] > 0.7) |
                   (sim_df['max_sim_EC_all'] > 0.7) |
                   (sim_df['max_sim_CA_all'] > 0.7)]
print(f"   Total: {len(high_sim)}/700 ({100*len(high_sim)/700:.1f}%)")
print(f"   By library: {high_sim['source_library'].value_counts().to_dict()}")

# Save summary
summary_stats = {
    'analysis': 'similarity_to_training',
    'total_compounds': len(genmol_df),
    'valid_fps': valid_fps,
    'mean_sim_SA': sim_df['max_sim_SA_all'].mean(),
    'mean_sim_EC': sim_df['max_sim_EC_all'].mean(),
    'mean_sim_CA': sim_df['max_sim_CA_all'].mean(),
    'high_sim_count': len(high_sim),
    'amide_count': (genmol_df['product_linkage_type'] == 'amide').sum(),
    'urea_count': (genmol_df['product_linkage_type'] == 'urea').sum()
}
pd.DataFrame([summary_stats]).to_csv(RESULTS_DIR / "summary_statistics.csv", index=False)

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print(f"Results saved to: {RESULTS_DIR}")
print("="*70)
