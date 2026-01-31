# GenMol: XAI-Guided Fragment-Based Molecule Generation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![RDKit](https://img.shields.io/badge/RDKit-2023.03+-green.svg)](https://www.rdkit.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A computational pipeline for generating novel antimicrobial molecules through systematic recombination of XAI-extracted pharmacophoric fragments. This repository validates explainable AI (XAI) attributions by demonstrating that fragments identified as important by RGCN models produce active molecules when recombined.

## Overview

This work is part of a thesis series on explainable AI for antimicrobial drug discovery:

| Paper | Focus | Key Contribution |
|-------|-------|------------------|
| Paper 1 | XAI Evaluation Framework | 4-tier framework; RGCN-SME identified as deployment-ready |
| Paper 2 | Fragment Extraction | 17,442 fragments (12,993 positive + 4,449 negative) from 68,736 compounds; SELECT design rules |
| **Paper 3 (This Work)** | Fragment Recombination | 700 molecules via acyl transfer (amide/urea); Tier 3 validation |

### Key Results

| Model | Tier 3 Actives | Hit Rate | Top Library | Top Hit Rate |
|-------|----------------|----------|-------------|--------------|
| **SA** | 68 | 39.8% | SA_CA | 72.2% |
| **EC** | 26 | 17.4% | EC | 70.0% |
| **CA** | 8 | 5.2% | CA | 9.1% |

**Hit Rate Definition**:  
- **Hit Rate** = Overall hit rate across all 7 libraries combined (Total Tier 3 actives / Total Tier 3 compounds)
- **Top Hit Rate** = Best-performing individual library for that pathogen model

*Example*: For SA model, 68 of 171 total Tier 3 compounds were predicted active (39.8% overall), but the SA_CA library achieved 72.2% (26/36), demonstrating library-specific optimization.

---

- **Novelty**: 99.6% (697/700) of generated molecules are novel (not in training data)
- **Linkage Diversity**: 600 amides (85.7%) + 100 ureas (14.3%) from oxazolidinone fragments
- **Library-Specific Validation**: SA library achieves 58% SA hit rate vs 0% EC hit rate
- **XAI Attribution**: 95% positive attributions for Tier 3 actives
- **SA Score Significance**: Tier 3 actives cluster at SA scores 3-4 (p<0.0001 for SA model)

## Repository Structure

```
GenMol/
├── README.md                           <- This file
├── requirements.txt                    <- Python dependencies
├── LICENSE                             <- MIT license
│
├── data/
│   ├── source_data/                    <- Raw pathogen training data
│   │   ├── S_aureus_input.csv
│   │   ├── S_aureus_raw_data.csv
│   │   ├── E_coli_input.csv
│   │   ├── E_coli_raw_data.csv
│   │   ├── C_albicans_input.csv
│   │   └── C_albicans_raw_data.csv
│   │
│   ├── source_fragments/               <- 28 source fragment CSVs (14 positive + 14 negative)
│   │   ├── SA_positive_scaffolds.csv
│   │   ├── SA_positive_substituents.csv
│   │   ├── EC_positive_scaffolds.csv
│   │   ├── EC_positive_substituents.csv
│   │   ├── CA_positive_scaffolds.csv
│   │   ├── CA_positive_substituents.csv
│   │   ├── SA_EC_positive_scaffolds.csv
│   │   ├── SA_EC_positive_substituents.csv
│   │   ├── SA_CA_positive_scaffolds.csv
│   │   ├── SA_CA_positive_substituents.csv
│   │   ├── CA_EC_positive_scaffolds.csv
│   │   ├── CA_EC_positive_substituents.csv
│   │   ├── TRIPLE_positive_scaffolds.csv
│   │   ├── TRIPLE_positive_substituents.csv
│   │   └── (+ corresponding negative fragment CSVs)
│   │
│   ├── genmol_input.csv                <- 700 generated molecules with fragment metadata
│   │
│   ├── predictions/                    <- RGCN model predictions & XAI attributions
│   │   ├── genmol_all_input_SA_prediction.csv  <- 700 compounds vs SA model
│   │   ├── genmol_all_input_EC_prediction.csv  <- 700 compounds vs EC model
│   │   ├── genmol_all_input_CA_prediction.csv  <- 700 compounds vs CA model
│   │   └── pathogen_selective_exemplar_compounds.csv  <- 3 exemplar compounds
│   │
│   └── novelty_duplicates.csv          <- 3 compounds found in training data
│
├── libraries/                          <- Built safe fragment libraries (7 libraries)
│   ├── SA_library/
│   │   ├── safe_library_SA.json
│   │   └── SA_builder_diagnostics.txt
│   ├── EC_library/
│   ├── CA_library/
│   ├── SA_EC_library/
│   ├── SA_CA_library/
│   ├── CA_EC_library/
│   ├── TRIPLE_library/
│   └── fragment_library_handle_analysis.csv  <- Library statistics
│
├── scripts/
│   ├── safe_library_builder.py         <- Build libraries from source_fragments/
│   ├── amide_molecule_generator.py     <- Generate molecules via acyl transfer
│   ├── tier3_prediction_analysis.py    <- Analyze Tier 3 predictions
│   ├── verify_novelty.py               <- Verify molecules not in training data
│   ├── similarity_analysis.py          <- Similarity analysis vs training data
│   ├── synthesis_prioritization.py     <- Goldilocks compound identification
│   ├── fragment_recombination_figures.py <- Generate recombination mechanism figures
│   ├── library_chemical_space.py       <- Chemical space visualization
│   ├── library_diversity_figure.py     <- Diversity analysis figures
│   └── utils/
│       ├── valency_completion.py
│       └── smiles_cleaner.py
│
├── data_analysis/                      <- Analysis results and statistics
│   ├── library_hit_rates_summary.csv   <- Tier 3 hit rates for all libraries
│   ├── library_diversity_statistics.csv
│   ├── library_statistics_detailed.csv
│   ├── similarity_analysis/            <- Similarity analysis outputs
│   │   ├── similarity_to_training.csv
│   │   ├── similarity_distributions.png
│   │   ├── ca_potential_true_negatives.csv
│   │   ├── azole_coupling_audit.csv
│   │   └── summary_statistics.csv
│   └── synthesis_prioritization/       <- Synthesis priority outputs
│       ├── goldilocks_compounds.csv
│       ├── synthesis_priority_all_scenario_a.csv
│       └── top_synthesis_candidates.csv
│
└── results/                            <- Visualization outputs
    └── recombination_figures/          <- Fragment recombination mechanism figures
        ├── CMPD_087_recombination.png  <- S. aureus exemplar
        ├── CMPD_162_recombination.png  <- E. coli exemplar
        ├── CMPD_428_recombination.png  <- C. albicans exemplar
        └── reaction_mechanism_exemplar_compounds_explained.txt
```

## Data Files

### Generated Molecules (`data/genmol_input.csv`)

The main dataset containing all 700 generated molecules with fragment metadata:

| Column | Description |
|--------|-------------|
| `COMPOUND_ID` | Unique identifier (CMPD_001 to CMPD_700) |
| `SMILES` | Product structure with atom mapping |
| `source_library` | Origin library (SA, EC, CA, SA_EC, SA_CA, CA_EC, TRIPLE) |
| `route_class` | Synthetic route (A, B, or C) |
| `product_linkage_type` | **amide** or **urea** - the bond formed |
| `acid_fragment_smiles` | SMILES of acid-bearing fragment |
| `amine_fragment_smiles` | SMILES of amine-bearing fragment |
| `acid_handle_origin` | Type of acid handle (native_acid, latent_ester, etc.) |
| `MW`, `LogP`, `TPSA`, etc. | Physicochemical properties |
| `SA_score` | Synthetic accessibility score (Ertl) |

### Prediction Files (`data/predictions/`)

Each prediction CSV contains all 700 generated compounds evaluated against one RGCN model:

| File | Model | Contents |
|------|-------|----------|
| `genmol_all_input_SA_prediction.csv` | S. aureus | Predictions + XAI attributions |
| `genmol_all_input_EC_prediction.csv` | E. coli | Predictions + XAI attributions |
| `genmol_all_input_CA_prediction.csv` | C. albicans | Predictions + XAI attributions |
| `pathogen_selective_exemplar_compounds.csv` | All | 3 pathogen-selective exemplar compounds |

**Key columns in prediction files**:

| Column | Description |
|--------|-------------|
| `COMPOUND_ID` | Unique identifier (CMPD_001 to CMPD_700) |
| `SMILES` | Molecule structure with atom mapping |
| `ensemble_prediction` | Mean probability from 5-fold ensemble |
| `prediction` | Binary classification (0/1) |
| `decision_scenario` | Tier classification (A, B, C, or D) |
| `murcko_substructure_N_smiles` | SMILES of Nth Murcko scaffold |
| `murcko_substructure_N_attribution` | XAI attribution score for Nth scaffold |

**Decision Scenarios**:
- **A (Tier 3)**: High agreement + High reliability - most trustworthy
- **B**: High agreement + Low reliability
- **C**: Low agreement + High reliability
- **D**: Low agreement + Low reliability

## Seven-Library Architecture

| Library | Description | Source Fragments | Purpose |
|---------|-------------|------------------|---------|
| **SA** | S. aureus-specific | Scaffolds + Substituents | Gram-positive only |
| **EC** | E. coli-specific | Scaffolds + Substituents | Gram-negative only |
| **CA** | C. albicans-specific | Scaffolds + Substituents | Antifungal only |
| **SA_EC** | Dual SA+EC active | Scaffolds + Substituents | Antibacterial broad-spectrum |
| **SA_CA** | Dual SA+CA active | Scaffolds + Substituents | Cross-kingdom |
| **CA_EC** | Dual CA+EC active | Scaffolds + Substituents | Cross-kingdom |
| **TRIPLE** | Active against all 3 | Scaffolds + Substituents | Pan-active broad-spectrum |

### Fragment Statistics

A total of **17,442 fragments** were extracted from 68,736 compounds, comprising both positive (activity-associated) and negative (inactivity-associated) fragments.

#### Positive Fragments (12,993 total)

| Library | Scaffolds | Substituents | Total |
|---------|----------:|-------------:|------:|
| SA | 1,986 | 346 | 2,332 |
| EC | 452 | 85 | 537 |
| CA | 1,100 | 134 | 1,234 |
| SA_EC | 4,803 | 847 | 5,650 |
| SA_CA | 255 | 34 | 289 |
| CA_EC | 111 | 37 | 148 |
| TRIPLE | 2,096 | 707 | 2,803 |
| **Total** | **10,803** | **2,190** | **12,993** |

#### Negative Fragments (4,449 total)

| Library | Scaffolds | Substituents | Total |
|---------|----------:|-------------:|------:|
| SA | 333 | 65 | 398 |
| EC | 133 | 28 | 161 |
| CA | 199 | 28 | 227 |
| SA_EC | 849 | 189 | 1,038 |
| SA_CA | 213 | 12 | 225 |
| CA_EC | 280 | 32 | 312 |
| TRIPLE | 2,087 | 1 | 2,088 |
| **Total** | **4,094** | **355** | **4,449** |

#### Positive vs Negative Comparison

| Category | Positive | Negative | Ratio (Pos:Neg) |
|----------|----------|----------|-----------------|
| Single Pathogen | 4,103 | 786 | 5.2:1 |
| Dual Pathogen | 6,087 | 1,575 | 3.9:1 |
| Triple Pathogen | 2,803 | 2,088 | 1.3:1 |
| **Total** | **12,993** | **4,449** | **2.9:1** |

## Installation



# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- RDKit 2023.03+
- pandas
- numpy
- scipy (optional, for statistical tests)

## Quick Start

### 1. Build Safe Libraries

Build fragment libraries from source CSVs:

```bash
# Build a single library
python scripts/safe_library_builder.py --library SA --base-dir .

# Build all 7 libraries
python scripts/safe_library_builder.py --all --base-dir .
```

### 2. Generate Molecules

Generate novel molecules via acyl transfer:

```bash
python scripts/amide_molecule_generator.py \
    --library SA \
    --safe-library libraries/SA_library/safe_library_SA.json \
    --max-products 100 \
    --seed 42
```

### 3. Analyze Tier 3 Predictions

After running RGCN predictions, analyze results:

```bash
# Analyze a single model
python scripts/tier3_prediction_analysis.py --model SA --base-dir .

# Analyze all 3 models
python scripts/tier3_prediction_analysis.py --all --base-dir .
```

### 4. Verify Novelty

Verify that generated molecules are not present in the training data:

```bash
python scripts/verify_novelty.py \
    --base-dir . \
    --training-dir /path/to/training/data
```

The training directory should contain:
- `S_aureus_input.csv`
- `E_coli_input.csv`
- `C_albicans_input.csv`

**Pre-computed results**: 697/700 (99.6%) molecules confirmed novel. The 3 duplicates are simple molecules from the TRIPLE library listed in `data/novelty_duplicates.csv`.

### 5. Similarity Analysis

Analyze structural similarity between generated molecules and training data:

```bash
python scripts/similarity_analysis.py
```

This script performs:
- **Tanimoto similarity** calculation (Morgan fingerprints, radius=2) vs all training sets
- **Azole coupling site audit** to verify pharmacophore preservation
- **CA inactive similarity** analysis for true/false negative identification
- **Fragment size correlation** with product novelty
- **Amide vs urea comparison** for ring-opening effect

**Outputs** saved to `data_analysis/similarity_analysis/`:
- `similarity_to_training.csv` - Full similarity data for all 700 molecules
- `ca_potential_true_negatives.csv` - 47 compounds similar to training inactives
- `azole_coupling_audit.csv` - 168 fragments with coupling site validation
- Distribution and comparison plots (PNG)

### 6. Synthesis Prioritization

Identify "Goldilocks" compounds for synthesis - novel yet reliable:

```bash
python scripts/synthesis_prioritization.py
```

This script:
- **Validates identical molecules** (sim≥0.99) against training labels
- **Extracts Goldilocks compounds** (similarity 0.30-0.45, Scenario A, predicted active)
- **Assigns priority tiers** to all Scenario A actives
- **Generates top 10 candidates** per pathogen

**Outputs** saved to `data_analysis/synthesis_prioritization/`:
- `goldilocks_compounds.csv` - 63 top priority compounds (Tier 1)
- `synthesis_priority_all_scenario_a.csv` - All 102 Scenario A actives with tiers
- `top_synthesis_candidates.csv` - Top 10 per pathogen for immediate synthesis
- `identical_molecules_analysis.csv` - Model validation (100% accuracy)

## Methodology

### Fragment Recombination via Acyl Transfer

Molecules are generated by coupling acid-bearing fragments with amine-bearing fragments using SMIRKS:

```
[C:1](=[O:2])[O:3][H].[N:5;H2,H1;!$(N=[!#6])]>>[C:1](=[O:2])[N:5]
```

**Product Linkage Types**:
- **Amide (N-C(=O)-C)**: 600 molecules (85.7%) - from esters, lactones, anhydrides
- **Urea (N-C(=O)-N)**: 100 molecules (14.3%) - from oxazolidinones (cyclic carbamates)

**Route Classification**:
- **Route A** (Native): Anhydrides, carboxylic acids - mildest conditions (rt-40°C)
- **Route B** (Easy Latent): Methyl/ethyl esters - mild conditions (60-80°C)
- **Route C** (Hard Latent): Lactones, oxazolidinones (urea formation), tert-butyl/benzyl esters - harsh conditions (100°C)

**Why Acyl Transfer?**

Acyl transfer reactions (amide and urea formation) were chosen as the fragment coupling strategy because:
1. **Synthetic accessibility**: Both amide and urea bonds are among the easiest to form in medicinal chemistry, requiring simple reagents and mild conditions
2. **Pharmaceutical relevance**: Amide bonds are the most common linkage in FDA-approved drugs; ureas are prevalent in kinase inhibitors and antimicrobials
3. **Robust methodology**: High-yielding reactions with well-established protocols suitable for parallel synthesis
4. **Validation focus**: Provides a reliable baseline for validating XAI-derived fragments before exploring more complex chemistries

**Future Directions**: Advanced coupling strategies such as click chemistry (CuAAC), Suzuki-Miyaura cross-coupling, or C-H activation could expand the chemical space accessible from XAI-identified fragments.

### SMILES Atom Mapping Notation

The generated molecule SMILES contain **atom mapping numbers** (e.g., `[C:1]`, `[N:5]`) that track atoms through the coupling reaction:

```
Product SMILES:  O=[C:1](NCCN(O)Cc1ccccc1)[NH:5]Cc1nccs1
                     ↑                      ↑
                 Atom map 1             Atom map 5
              (from acid fragment)   (from amine fragment)
```

| Atom Map | Origin | Chemical Role |
|----------|--------|---------------|
| `[C:1]` | Acid fragment | Carbonyl carbon of the newly formed amide/urea |
| `[N:5]` | Amine fragment | Nitrogen of the newly formed amide/urea |

These atom maps are useful for:
- **Attribution mapping**: Tracing which XAI attributions belong to which parent fragment
- **Synthetic clarity**: Showing the retrosynthetic disconnection point
- **Fragment tracking**: Understanding how the molecule was assembled

> **Note**: The atom mapping does not affect molecular structure visualization in RDKit or other tools. To obtain clean SMILES without atom maps:
> ```python
> from rdkit import Chem
> mol = Chem.MolFromSmiles(mapped_smiles)
> for atom in mol.GetAtoms():
>     atom.SetAtomMapNum(0)
> clean_smiles = Chem.MolToSmiles(mol)
> ```

### Tier 3 Internal Consistency

Tier 3 (Scenario A) represents the highest confidence predictions:
1. **Prediction Agreement**: All 5 ensemble models agree on classification
2. **Explanation Reliability**: Fragment attributions show >70% consistency with magnitude >0.1

### Safety Filtering

All generated molecules are filtered for:
- PAINS A/B/C substructures
- Michael acceptor precursors
- Alkyl halides and azides

## Key Findings

### Novelty Verification

99.6% (697/700) of generated molecules are confirmed novel - not present in any of the three model training datasets (68,725 unique structures). The 3 duplicates are simple molecules (acetohydrazide derivatives) from the TRIPLE library.

### Similarity to Training Data

Comprehensive Tanimoto similarity analysis (Morgan fingerprints, radius=2) comparing all 700 generated molecules to training datasets:

| Pathogen | Training Size | Mean Sim | Max Sim | Interpretation |
|----------|---------------|----------|---------|----------------|
| SA       | 54,277        | 0.385    | 1.000   | Some identical to training |
| EC       | 44,920        | 0.384    | 1.000   | Some identical to training |
| CA       | 28,476        | 0.366    | 0.738   | All moderately novel |

**Why this matters for predictions:**
- **High similarity (→1.0)**: Model can **interpolate** - predictions are reliable because similar structures exist in training data
- **Low similarity (<0.7)**: Model must **extrapolate** - predictions less reliable as it ventures into unknown chemical space

**Key observation**: The CA model never sees a generated molecule with >0.738 similarity to its training data, meaning it must extrapolate for every prediction. In contrast, SA/EC models occasionally see near-identical structures (similarity=1.0), enabling confident interpolation. This, combined with CA having the smallest training set (28,476 vs 54,277 for SA), explains why the CA hit rate (5.2%) is lower - the model is being asked to evaluate genuinely novel chemistry outside its training domain.

### Azole Pharmacophore Preservation

Audit of 168 fragments containing aromatic nitrogens confirmed:
- **100% have alternative aliphatic amines** for coupling
- **0 fragments** have only ring nitrogens available
- Aromatic ring nitrogens (essential for CYP51 coordination in antifungals) are **preserved** during coupling

### Amide vs Urea (Ring-Opening) Novelty

Urea products from oxazolidinone ring-opening show significantly lower similarity to training:

| Pathogen | Amide Similarity | Urea Similarity | p-value  |
|----------|------------------|-----------------|----------|
| SA       | 0.390            | 0.358           | 0.0046   |
| EC       | 0.391            | 0.340           | <0.0001  |
| CA       | 0.379            | 0.289           | <0.0001  |

### Model Validation: Identical Molecules

Four generated molecules with similarity ≥0.99 to training compounds were identified. All predictions matched training labels exactly (100% accuracy), confirming model reliability for interpolation:

| Compound | Pathogen | Similarity | Prediction | Training | Match |
|----------|----------|------------|------------|----------|-------|
| CMPD_654 | SA | 1.000 | Inactive | Inactive | Yes |
| CMPD_654 | EC | 1.000 | Inactive | Inactive | Yes |
| CMPD_693 | EC | 1.000 | Inactive | Inactive | Yes |
| CMPD_696 | EC | 1.000 | Inactive | Inactive | Yes |

### Synthesis Prioritization: Goldilocks Compounds

The "Goldilocks zone" (similarity 0.30-0.45) represents compounds that are **novel enough to be interesting** yet **similar enough for reliable predictions**. Combined with Scenario A (Tier 3) classification and active prediction, these are top synthesis priorities:

| Pathogen | Goldilocks Count | Mean Similarity | Mean Probability |
|----------|------------------|-----------------|------------------|
| SA       | 39               | 0.377           | 0.886            |
| EC       | 19               | 0.377           | 0.855            |
| CA       | 5                | 0.361           | 0.760            |
| **Total**| **63**           |                 |                  |

**Priority Tier System:**
- **Tier 1 (Goldilocks, 0.30-0.45)**: 63 compounds - TOP PRIORITY for synthesis
- **Tier 2 (0.45-0.60)**: 23 compounds - Moderately novel, good reliability
- **Tier 3 (0.20-0.30)**: 11 compounds - Very novel, may need extra validation
- **Tier 4 (>0.60)**: 5 compounds - Less novel but highly reliable

Top candidates per pathogen are available in `data_analysis/synthesis_prioritization/top_synthesis_candidates.csv`.

### Pathogen-Selective Exemplar Compounds

Three compounds were selected as pathogen-selective exemplars demonstrating the fragment recombination approach:

| Compound | Target | Predicted Activity | Route | Linkage | Reaction Time |
|----------|--------|-------------------|-------|---------|---------------|
| **CMPD_087** | S. aureus | SA=0.982, EC=0.361, CA=0.199 | C | UREA | 12-24h |
| **CMPD_162** | E. coli | SA=0.271, EC=0.987, CA=0.250 | C | UREA | 12-24h |
| **CMPD_428** | C. albicans | SA=0.013, EC=0.044, CA=0.776 | B | AMIDE | 6h |

**Reaction Conditions:**
- **Route B** (Ester aminolysis): K₂CO₃, MeOH/DMF, 60°C, 6h → AMIDE bond
- **Route C** (Oxazolidinone ring-opening): Neat, 100°C, 12-24h → UREA bond

Fragment recombination figures showing the reaction mechanisms are available in `results/recombination_figures/`.

### Attribution Predicts Transferability

High training attribution (>0.2) predicts cross-pathogen transfer better than high training activity (>90%).

### Route B Superiority

Route B (mild aminolysis) dominates at 57.1% of generated compounds, with significantly better synthetic accessibility (SA_score 3.22) compared to Route C (SA_score 3.66, p<0.0001).

### SA Score Goldilocks Zone

Tier 3 actives cluster at SA scores 3-4 (p<0.0001), indicating optimal synthetic accessibility.

## Citation

```bibtex
@article{xai_fbdd_antimicrobial,
  title={TBD},
  author={Onawole, Abdulmujeeb T; Blaskovich, Mark A.T.; Zuegg, Johannes},
  journal={TBD},
  year={2025},
  note={Paper 3: Fragment recombination validation}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contact

For questions about the methodology or scripts, please open an issue.
