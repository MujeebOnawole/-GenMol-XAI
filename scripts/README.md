# GenMol Scripts

Core scripts for the XAI-guided fragment-based molecule generation pipeline.

## Essential Scripts

| Script | Location | Purpose |
|--------|----------|---------|
| `safe_library_builder.py` | `scripts/` | Build safe fragment libraries from source CSVs |
| `amide_molecule_generator.py` | `scripts/` | Generate molecules via acyl transfer (amide/urea) |
| `tier3_prediction_analysis.py` | `scripts/` | Analyze Tier 3 predictions with generation factors |
| `verify_novelty.py` | `scripts/` | Verify generated molecules are not in training data |

## Utility Scripts (`scripts/utils/`)

| Script | Purpose |
|--------|---------|
| `valency_completion.py` | Fix valency issues in generated molecules |
| `smiles_cleaner.py` | Clean and canonicalize SMILES |

## Quick Start

### 1. Build All 7 Fragment Libraries

```bash
python scripts/safe_library_builder.py --all --base-dir .
```

Or build individually:

```bash
python scripts/safe_library_builder.py --library SA --base-dir .
python scripts/safe_library_builder.py --library EC --base-dir .
# ... etc for CA, SA_EC, SA_CA, CA_EC, TRIPLE
```

### 2. Generate Molecules

```bash
python scripts/amide_molecule_generator.py \
    --library SA \
    --safe-library libraries/SA_library/safe_library_SA.json \
    --max-products 100
```

### 3. Analyze Tier 3 Predictions

After running RGCN model predictions:

```bash
# All models
python scripts/tier3_prediction_analysis.py --all --base-dir .

# Single model
python scripts/tier3_prediction_analysis.py --model SA --base-dir .
```

### 4. Verify Novelty

```bash
python scripts/verify_novelty.py --base-dir . --training-dir /path/to/training/data
```

## Output Files

### From Library Builder
- `libraries/{LIB}_library/safe_library_{LIB}.json` - Fragment library
- `libraries/{LIB}_library/{LIB}_builder_diagnostics.txt` - Build log

### From Molecule Generator
- `outputs/generated_molecules.csv` - All molecules with metadata

### From Tier 3 Analysis
- `results/tier3_reports/{MODEL}_tier3_analysis_report.md` - Report
- `results/tier3_reports/{MODEL}_tier3_actives.csv` - Active compounds
- `results/tier3_reports/{MODEL}_library_stats.csv` - Library performance

### From Novelty Verification
- `results/novelty_verification_results.csv` - Full novelty check results
- `data/novelty_duplicates.csv` - Any duplicates found in training data

## Dependencies

See `requirements.txt` in the repository root.
