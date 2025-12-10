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
| Paper 2 | Fragment Extraction | 12,993 fragments from 68,736 compounds; SELECT design rules |
| **Paper 3 (This Work)** | Fragment Recombination | 700 molecules via acyl transfer (amide/urea); Tier 3 validation |

### Key Results

| Model | Tier 3 Actives | Hit Rate | Top Library | Top Hit Rate |
|-------|----------------|----------|-------------|--------------|
| **SA** | 68 | 39.8% | SA_CA | 72.2% |
| **EC** | 26 | 17.4% | EC | 70.0% |
| **CA** | 8 | 5.2% | CA | 9.1% |

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
│   ├── source_fragments/               <- 14 source fragment CSVs
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
│   │   └── TRIPLE_positive_substituents.csv
│   │
│   ├── genmol_input.csv                <- 700 generated molecules with fragment metadata
│   │
│   ├── predictions/                    <- RGCN model predictions & XAI attributions
│   │   ├── genmol_SA_predictions.csv   <- 700 compounds evaluated against SA model
│   │   ├── genmol_EC_predictions.csv   <- 700 compounds evaluated against EC model
│   │   └── genmol_CA_predictions.csv   <- 700 compounds evaluated against CA model
│   │
│   └── novelty_duplicates.csv          <- 3 compounds found in training data
│
├── libraries/                          <- Built safe fragment libraries
│   ├── SA_library/
│   │   ├── safe_library_SA.json
│   │   └── SA_builder_diagnostics.txt
│   ├── EC_library/
│   ├── CA_library/
│   ├── SA_EC_library/
│   ├── SA_CA_library/
│   ├── CA_EC_library/
│   └── TRIPLE_library/
│
├── scripts/
│   ├── safe_library_builder.py         <- Build libraries from source_fragments/
│   ├── amide_molecule_generator.py     <- Generate molecules via acyl transfer (amide/urea)
│   ├── tier3_prediction_analysis.py    <- Analyze Tier 3 predictions
│   ├── verify_novelty.py               <- Verify molecules not in training data
│   └── utils/
│       ├── valency_completion.py
│       └── smiles_cleaner.py
│
├── outputs/                            <- Generated molecules
│   └── generated_molecules.csv
│
└── results/                            <- Analysis results
    └── tier3_reports/
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
| `genmol_SA_predictions.csv` | S. aureus | Predictions + XAI attributions |
| `genmol_EC_predictions.csv` | E. coli | Predictions + XAI attributions |
| `genmol_CA_predictions.csv` | C. albicans | Predictions + XAI attributions |

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
