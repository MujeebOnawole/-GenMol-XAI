# Fragment Library Analysis Results

Analysis of Tier 3 (Scenario A) hit rates across all 7 fragment libraries.

## Files

| File | Description |
|------|-------------|
| `library_hit_rates_summary.csv` | Compact summary of Tier 3 hit rates for all libraries |
| `library_statistics_detailed.csv` | Extended statistics including physicochemical properties |
| `SA_EC_library_cross_pathogen_activity.csv` | Detailed cross-pathogen breakdown for SA_EC library |

## Key Findings

### Best Performing Libraries (Tier 3 Hit Rate)

| Target | Best Library | Hit Rate |
|--------|--------------|----------|
| S. aureus | SA_CA | 72.2% |
| E. coli | EC | 70.0% |
| C. albicans | CA | 9.1% |

### Single vs Dual-Active Library Performance

Single-pathogen libraries consistently outperform dual/triple-active designs:

- **SA library** achieves 58.0% SA hit rate vs SA_EC's 9.1% (6.4× better)
- **EC library** achieves 70.0% EC hit rate vs SA_EC's 3.3% (21× better)

### SA_EC Library Cross-Pathogen Activity

Despite being designed for dual SA+EC activity:
- Only 4% of compounds show dual activity
- 74% are inactive against both pathogens
- Suggests multi-objective optimization is challenging

## Methodology

- **Hit Rate** = Tier 3 Actives / Tier 3 Total × 100
- **Tier 3 (Scenario A)** = High ensemble agreement + High explanation reliability
- All libraries tested with 100 compounds each (700 total)

## Column Definitions

### library_hit_rates_summary.csv
- `library`: Fragment library identifier
- `total_fragments`: Number of fragments available in library
- `test_compounds`: Compounds evaluated (100 per library)
- `*_tier3_count`: Compounds classified as Tier 3 for that model
- `*_tier3_actives`: Tier 3 compounds predicted active
- `*_hit_rate_pct`: Hit rate percentage (actives/total × 100)

### library_statistics_detailed.csv
- `library_type`: single, dual, or triple pathogen target
- `scaffolds/substituents`: Fragment breakdown
- `avg_MW/LogP/TPSA`: Mean physicochemical properties
- `*_overall_actives`: All-scenario active predictions
- `best_model`: Model with highest hit rate for this library
