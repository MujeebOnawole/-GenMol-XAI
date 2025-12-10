#!/usr/bin/env python3
"""
Tier 3 Prediction Analysis Script

Analyzes RGCN model predictions with focus on Tier 3 (Scenario A) compounds
where prediction and XAI explanation are in agreement (high confidence).

This script merges:
- Input CSV: Contains generation metadata (source_library, fragments, properties)
- Prediction CSV: Contains model predictions and XAI attributions

Usage:
    python tier3_prediction_analysis.py --model SA --input-csv path/to/input.csv --pred-csv path/to/prediction.csv
    python tier3_prediction_analysis.py --model SA  # Uses default paths
    python tier3_prediction_analysis.py --all       # Runs for SA, EC, CA models

Author: Generated for FBDD XAI Pipeline
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
try:
    from scipy import stats
except ImportError:
    stats = None

import pandas as pd
import numpy as np


def load_and_merge_data(input_csv: str, pred_csv: str) -> pd.DataFrame:
    """
    Load input and prediction CSVs and merge on COMPOUND_ID.

    Args:
        input_csv: Path to input CSV with generation metadata
        pred_csv: Path to prediction CSV with model results

    Returns:
        Merged DataFrame
    """
    print(f"Loading input: {input_csv}")
    input_df = pd.read_csv(input_csv)

    print(f"Loading predictions: {pred_csv}")
    pred_df = pd.read_csv(pred_csv)

    # Columns to keep from input (avoid duplicates with prediction)
    input_cols = [
        'COMPOUND_ID', 'source_library', 'reaction_type', 'route_class',
        'acid_fragment_id', 'acid_fragment_smiles', 'acid_role',
        'acid_library_type', 'acid_avg_attribution', 'acid_activity_rate',
        'handle_origin_acid', 'acid_state',
        'amine_fragment_id', 'amine_fragment_smiles', 'amine_role',
        'amine_library_type', 'amine_avg_attribution', 'amine_activity_rate',
        'handle_origin_amine', 'amine_state',
        'novel_vs_pathogen_train', 'novel_vs_union_train',
        'MW', 'LogP', 'TPSA', 'HBA', 'HBD', 'AromRings', 'RotBonds',
        'QED', 'SA_score', 'design_score', 'rank_within_library'
    ]

    # Filter to available columns
    available_input_cols = [c for c in input_cols if c in input_df.columns]

    # Merge
    merged = pred_df.merge(input_df[available_input_cols], on='COMPOUND_ID', how='left')

    print(f"Merged: {len(merged)} compounds")
    return merged


def analyze_overall_predictions(df: pd.DataFrame, model_name: str) -> Dict:
    """Analyze overall prediction distribution."""
    results = {
        'total_compounds': len(df),
        'predicted_active': int((df['prediction'] == 1).sum()),
        'predicted_inactive': int((df['prediction'] == 0).sum()),
        'active_rate': float((df['prediction'] == 1).mean() * 100),
    }

    # Decision scenario distribution
    results['scenario_distribution'] = df['decision_scenario'].value_counts().to_dict()

    return results


def analyze_tier3(df: pd.DataFrame, model_name: str) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Analyze Tier 3 (Scenario A) compounds.

    Returns:
        tier3_active: DataFrame of Tier 3 actives
        tier3_inactive: DataFrame of Tier 3 inactives
        stats: Dictionary of statistics
    """
    tier3 = df[df['decision_scenario'] == 'A'].copy()
    tier3_active = tier3[tier3['prediction'] == 1].copy()
    tier3_inactive = tier3[tier3['prediction'] == 0].copy()

    stats = {
        'tier3_total': len(tier3),
        'tier3_pct': len(tier3) / len(df) * 100,
        'tier3_active': len(tier3_active),
        'tier3_inactive': len(tier3_inactive),
        'tier3_active_rate': len(tier3_active) / len(tier3) * 100 if len(tier3) > 0 else 0,
    }

    return tier3_active, tier3_inactive, stats


def analyze_by_library(df: pd.DataFrame, tier3_active: pd.DataFrame) -> pd.DataFrame:
    """Analyze Tier 3 hit rates by source library."""
    libraries = ['SA', 'EC', 'CA', 'SA_EC', 'SA_CA', 'CA_EC', 'TRIPLE']

    results = []
    for lib in libraries:
        lib_all = df[df['source_library'] == lib]
        lib_tier3 = df[(df['source_library'] == lib) & (df['decision_scenario'] == 'A')]
        lib_tier3_active = tier3_active[tier3_active['source_library'] == lib]

        if len(lib_all) > 0:
            results.append({
                'library': lib,
                'total_screened': len(lib_all),
                'tier3_count': len(lib_tier3),
                'tier3_pct': len(lib_tier3) / len(lib_all) * 100,
                'tier3_active': len(lib_tier3_active),
                'tier3_hit_rate': len(lib_tier3_active) / len(lib_tier3) * 100 if len(lib_tier3) > 0 else 0,
            })

    return pd.DataFrame(results)


def analyze_fragments(tier3_active: pd.DataFrame, top_n: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Analyze top fragments in Tier 3 actives."""

    # Top acid fragments
    acid_counts = tier3_active.groupby('acid_fragment_id').agg({
        'COMPOUND_ID': 'count',
        'acid_fragment_smiles': 'first',
        'acid_avg_attribution': 'first',
        'acid_activity_rate': 'first',
        'acid_library_type': 'first',
    }).reset_index()
    acid_counts.columns = ['fragment_id', 'count', 'smiles', 'avg_attribution', 'activity_rate', 'library_type']
    acid_counts = acid_counts.sort_values('count', ascending=False).head(top_n)

    # Top amine fragments
    amine_counts = tier3_active.groupby('amine_fragment_id').agg({
        'COMPOUND_ID': 'count',
        'amine_fragment_smiles': 'first',
        'amine_avg_attribution': 'first',
        'amine_activity_rate': 'first',
        'amine_library_type': 'first',
    }).reset_index()
    amine_counts.columns = ['fragment_id', 'count', 'smiles', 'avg_attribution', 'activity_rate', 'library_type']
    amine_counts = amine_counts.sort_values('count', ascending=False).head(top_n)

    return acid_counts, amine_counts


def analyze_xai_attributions(tier3_active: pd.DataFrame) -> Dict:
    """Analyze XAI attribution patterns for Tier 3 actives."""

    # Find attribution columns
    attr_cols = [c for c in tier3_active.columns
                 if 'murcko_substructure' in c and '_attribution' in c]

    if not attr_cols:
        return {'error': 'No attribution columns found'}

    # Convert to numeric
    attr_df = tier3_active[attr_cols].apply(pd.to_numeric, errors='coerce')

    # Calculate statistics
    total_attrs = attr_df.notna().sum().sum()
    pos_attrs = (attr_df > 0).sum().sum()
    neg_attrs = (attr_df < 0).sum().sum()

    # Mean attributions per substructure
    mean_attrs = attr_df.mean().dropna().to_dict()

    return {
        'n_attribution_columns': len(attr_cols),
        'total_attributions': int(total_attrs),
        'positive_attributions': int(pos_attrs),
        'negative_attributions': int(neg_attrs),
        'positive_pct': pos_attrs / total_attrs * 100 if total_attrs > 0 else 0,
        'mean_attributions': mean_attrs,
    }


def analyze_properties(tier3_active: pd.DataFrame, tier3_inactive: pd.DataFrame) -> pd.DataFrame:
    """Compare physicochemical properties between actives and inactives."""

    props = ['MW', 'LogP', 'TPSA', 'HBA', 'HBD', 'QED', 'SA_score']
    available_props = [p for p in props if p in tier3_active.columns]

    results = []
    for prop in available_props:
        act_mean = tier3_active[prop].mean()
        act_std = tier3_active[prop].std()
        inact_mean = tier3_inactive[prop].mean()
        inact_std = tier3_inactive[prop].std()

        results.append({
            'property': prop,
            'active_mean': act_mean,
            'active_std': act_std,
            'inactive_mean': inact_mean,
            'inactive_std': inact_std,
            'delta': act_mean - inact_mean,
        })

    return pd.DataFrame(results)


def analyze_generation_factors(tier3_active: pd.DataFrame, tier3_inactive: pd.DataFrame) -> Dict:
    """
    Analyze generation metadata factors distinguishing Tier 3 actives from inactives.

    Examines: route_class, acid_library_type, amine_library_type, handle_origin_acid
    """
    results = {}

    # Route class distribution
    if 'route_class' in tier3_active.columns:
        active_routes = tier3_active['route_class'].value_counts(normalize=True) * 100
        inactive_routes = tier3_inactive['route_class'].value_counts(normalize=True) * 100
        results['route_class'] = {
            'active': active_routes.to_dict(),
            'inactive': inactive_routes.to_dict(),
        }

    # Acid library type distribution
    if 'acid_library_type' in tier3_active.columns:
        active_acid = tier3_active['acid_library_type'].value_counts(normalize=True) * 100
        inactive_acid = tier3_inactive['acid_library_type'].value_counts(normalize=True) * 100
        results['acid_library_type'] = {
            'active': active_acid.to_dict(),
            'inactive': inactive_acid.to_dict(),
        }

    # Amine library type distribution
    if 'amine_library_type' in tier3_active.columns:
        active_amine = tier3_active['amine_library_type'].value_counts(normalize=True) * 100
        inactive_amine = tier3_inactive['amine_library_type'].value_counts(normalize=True) * 100
        results['amine_library_type'] = {
            'active': active_amine.to_dict(),
            'inactive': inactive_amine.to_dict(),
        }

    # Handle origin acid distribution
    if 'handle_origin_acid' in tier3_active.columns:
        active_handle = tier3_active['handle_origin_acid'].value_counts(normalize=True) * 100
        inactive_handle = tier3_inactive['handle_origin_acid'].value_counts(normalize=True) * 100
        results['handle_origin_acid'] = {
            'active': active_handle.to_dict(),
            'inactive': inactive_handle.to_dict(),
        }

    # SA_score statistical comparison
    if 'SA_score' in tier3_active.columns and len(tier3_active) > 0 and len(tier3_inactive) > 0:
        act_sa = tier3_active['SA_score'].dropna()
        inact_sa = tier3_inactive['SA_score'].dropna()
        if len(act_sa) > 0 and len(inact_sa) > 0:
            results['sa_score_comparison'] = {
                'active_mean': float(act_sa.mean()),
                'active_std': float(act_sa.std()),
                'inactive_mean': float(inact_sa.mean()),
                'inactive_std': float(inact_sa.std()),
            }
            # Add statistical test if scipy is available
            if stats is not None:
                stat, pvalue = stats.mannwhitneyu(act_sa, inact_sa, alternative='two-sided')
                results['sa_score_comparison']['mannwhitney_statistic'] = float(stat)
                results['sa_score_comparison']['pvalue'] = float(pvalue)
                results['sa_score_comparison']['significant'] = pvalue < 0.05

    return results


def format_generation_factors_report(gen_factors: Dict, model_name: str) -> List[str]:
    """Format generation factors analysis for markdown report."""
    lines = []
    lines.append(f"\n## Generation Factors Analysis (Tier 3)")
    lines.append(f"\nComparing Tier 3 actives vs inactives by generation metadata.\n")

    # Route class
    if 'route_class' in gen_factors:
        lines.append("### Route Class Distribution")
        lines.append("\n| Route | Actives % | Inactives % |")
        lines.append("|-------|-----------|-------------|")
        all_routes = set(gen_factors['route_class']['active'].keys()) | set(gen_factors['route_class']['inactive'].keys())
        for route in sorted(all_routes):
            act_pct = gen_factors['route_class']['active'].get(route, 0)
            inact_pct = gen_factors['route_class']['inactive'].get(route, 0)
            lines.append(f"| {route} | {act_pct:.1f}% | {inact_pct:.1f}% |")

    # Acid library type
    if 'acid_library_type' in gen_factors:
        lines.append("\n### Acid Fragment Library Source")
        lines.append("\n| Library | Actives % | Inactives % |")
        lines.append("|---------|-----------|-------------|")
        all_libs = set(gen_factors['acid_library_type']['active'].keys()) | set(gen_factors['acid_library_type']['inactive'].keys())
        for lib in sorted(all_libs):
            act_pct = gen_factors['acid_library_type']['active'].get(lib, 0)
            inact_pct = gen_factors['acid_library_type']['inactive'].get(lib, 0)
            lines.append(f"| {lib} | {act_pct:.1f}% | {inact_pct:.1f}% |")

    # SA_score comparison
    if 'sa_score_comparison' in gen_factors:
        sa = gen_factors['sa_score_comparison']
        lines.append("\n### Synthetic Accessibility Score Comparison")
        lines.append(f"\n- **Actives**: {sa['active_mean']:.2f} +/- {sa['active_std']:.2f}")
        lines.append(f"- **Inactives**: {sa['inactive_mean']:.2f} +/- {sa['inactive_std']:.2f}")
        if 'pvalue' in sa:
            lines.append(f"- **Mann-Whitney U p-value**: {sa['pvalue']:.4f}")
            if sa['significant']:
                lines.append(f"- **Significant difference** (p < 0.05): Tier 3 actives have {'higher' if sa['active_mean'] > sa['inactive_mean'] else 'lower'} SA scores")
            else:
                lines.append(f"- No significant difference in SA scores")
        else:
            delta = sa['active_mean'] - sa['inactive_mean']
            lines.append(f"- **Delta**: {delta:+.2f}")

    return lines


def get_top_compounds(tier3_active: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Get top N Tier 3 active compounds by prediction probability."""

    cols = ['COMPOUND_ID', 'SMILES', 'source_library', 'ensemble_prediction',
            'MW', 'LogP', 'TPSA', 'QED', 'SA_score',
            'acid_fragment_id', 'amine_fragment_id']

    available_cols = [c for c in cols if c in tier3_active.columns]

    return tier3_active.nlargest(n, 'ensemble_prediction')[available_cols]


def get_druglike_compounds(tier3_active: pd.DataFrame,
                           qed_min: float = 0.4,
                           mw_max: float = 500) -> pd.DataFrame:
    """Filter for drug-like Tier 3 actives."""

    druglike = tier3_active[
        (tier3_active['QED'] > qed_min) &
        (tier3_active['MW'] < mw_max)
    ].copy()

    return druglike.nlargest(min(20, len(druglike)), 'ensemble_prediction')


def generate_report(model_name: str,
                    df: pd.DataFrame,
                    tier3_active: pd.DataFrame,
                    tier3_inactive: pd.DataFrame,
                    overall_stats: Dict,
                    tier3_stats: Dict,
                    library_stats: pd.DataFrame,
                    acid_fragments: pd.DataFrame,
                    amine_fragments: pd.DataFrame,
                    xai_stats: Dict,
                    property_stats: pd.DataFrame,
                    top_compounds: pd.DataFrame,
                    druglike_compounds: pd.DataFrame,
                    generation_factors: Dict,
                    output_dir: str) -> str:
    """Generate comprehensive markdown report."""

    report_lines = []

    # Header
    report_lines.append(f"# {model_name} Model Tier 3 Analysis Report")
    report_lines.append(f"\nGenerated: {datetime.now().isoformat()}")
    report_lines.append(f"\n## Executive Summary")
    report_lines.append(f"\n- **Total compounds screened:** {overall_stats['total_compounds']}")
    report_lines.append(f"- **Predicted active:** {overall_stats['predicted_active']} ({overall_stats['active_rate']:.1f}%)")
    report_lines.append(f"- **Tier 3 compounds:** {tier3_stats['tier3_total']} ({tier3_stats['tier3_pct']:.1f}%)")
    report_lines.append(f"- **Tier 3 actives:** {tier3_stats['tier3_active']}")
    report_lines.append(f"- **Tier 3 active rate:** {tier3_stats['tier3_active_rate']:.1f}%")

    # Decision Scenario Distribution
    report_lines.append(f"\n## Decision Scenario Distribution")
    report_lines.append(f"\n| Scenario | Count | Description |")
    report_lines.append(f"|----------|-------|-------------|")
    scenario_desc = {
        'A': 'HIGH Agreement + HIGH Reliability (Tier 3)',
        'B': 'HIGH Agreement + LOW Reliability',
        'C': 'LOW Agreement + HIGH Reliability',
        'D': 'LOW Agreement + LOW Reliability',
    }
    for scenario, count in sorted(overall_stats['scenario_distribution'].items()):
        desc = scenario_desc.get(scenario, 'Unknown')
        report_lines.append(f"| {scenario} | {count} | {desc} |")

    # Library Performance
    report_lines.append(f"\n## Library Performance")
    report_lines.append(f"\n| Library | Screened | Tier 3 | Tier 3 % | Tier 3 Actives | Hit Rate |")
    report_lines.append(f"|---------|----------|--------|----------|----------------|----------|")
    for _, row in library_stats.iterrows():
        report_lines.append(
            f"| {row['library']} | {row['total_screened']} | {row['tier3_count']} | "
            f"{row['tier3_pct']:.1f}% | {row['tier3_active']} | {row['tier3_hit_rate']:.1f}% |"
        )

    # Top Fragments
    report_lines.append(f"\n## Top Acid Fragments in Tier 3 Actives")
    report_lines.append(f"\n| Fragment ID | Count | Attribution | Activity Rate | Library | SMILES |")
    report_lines.append(f"|-------------|-------|-------------|---------------|---------|--------|")
    for _, row in acid_fragments.iterrows():
        smiles_short = str(row['smiles'])[:40] + '...' if len(str(row['smiles'])) > 40 else row['smiles']
        attr = f"{row['avg_attribution']:.3f}" if pd.notna(row['avg_attribution']) else 'N/A'
        rate = f"{row['activity_rate']:.1f}%" if pd.notna(row['activity_rate']) else 'N/A'
        report_lines.append(
            f"| {row['fragment_id']} | {row['count']} | {attr} | {rate} | {row['library_type']} | {smiles_short} |"
        )

    report_lines.append(f"\n## Top Amine Fragments in Tier 3 Actives")
    report_lines.append(f"\n| Fragment ID | Count | Attribution | Activity Rate | Library | SMILES |")
    report_lines.append(f"|-------------|-------|-------------|---------------|---------|--------|")
    for _, row in amine_fragments.iterrows():
        smiles_short = str(row['smiles'])[:40] + '...' if len(str(row['smiles'])) > 40 else row['smiles']
        attr = f"{row['avg_attribution']:.3f}" if pd.notna(row['avg_attribution']) else 'N/A'
        rate = f"{row['activity_rate']:.1f}%" if pd.notna(row['activity_rate']) else 'N/A'
        report_lines.append(
            f"| {row['fragment_id']} | {row['count']} | {attr} | {rate} | {row['library_type']} | {smiles_short} |"
        )

    # XAI Attribution Analysis
    report_lines.append(f"\n## XAI Attribution Analysis")
    if 'error' not in xai_stats:
        report_lines.append(f"\n- **Attribution columns:** {xai_stats['n_attribution_columns']}")
        report_lines.append(f"- **Total attributions:** {xai_stats['total_attributions']}")
        report_lines.append(f"- **Positive attributions:** {xai_stats['positive_attributions']} ({xai_stats['positive_pct']:.1f}%)")
        report_lines.append(f"- **Negative attributions:** {xai_stats['negative_attributions']} ({100-xai_stats['positive_pct']:.1f}%)")
        report_lines.append(f"\n**Interpretation:** {xai_stats['positive_pct']:.1f}% positive attributions indicates ")
        report_lines.append(f"XAI correctly identifies activity-driving substructures for predicted actives.")

    # Property Comparison
    report_lines.append(f"\n## Physicochemical Property Comparison (Tier 3)")
    report_lines.append(f"\n| Property | Actives (n={tier3_stats['tier3_active']}) | Inactives (n={tier3_stats['tier3_inactive']}) | Delta |")
    report_lines.append(f"|----------|----------|-----------|-------|")
    for _, row in property_stats.iterrows():
        report_lines.append(
            f"| {row['property']} | {row['active_mean']:.2f} +/- {row['active_std']:.2f} | "
            f"{row['inactive_mean']:.2f} +/- {row['inactive_std']:.2f} | {row['delta']:+.2f} |"
        )

    # Top Compounds
    report_lines.append(f"\n## Top 20 Tier 3 Active Compounds")
    report_lines.append(f"\n| COMPOUND_ID | Source | Probability | MW | LogP | TPSA | QED |")
    report_lines.append(f"|-------------|--------|-------------|-----|------|------|-----|")
    for _, row in top_compounds.iterrows():
        report_lines.append(
            f"| {row['COMPOUND_ID']} | {row['source_library']} | {row['ensemble_prediction']:.3f} | "
            f"{row['MW']:.1f} | {row['LogP']:.2f} | {row['TPSA']:.1f} | {row['QED']:.3f} |"
        )

    # Drug-like Compounds
    report_lines.append(f"\n## Drug-like Tier 3 Actives (QED > 0.4, MW < 500)")
    report_lines.append(f"\n**Count:** {len(druglike_compounds)}")
    if len(druglike_compounds) > 0:
        report_lines.append(f"\n| COMPOUND_ID | Source | Probability | MW | LogP | QED |")
        report_lines.append(f"|-------------|--------|-------------|-----|------|-----|")
        for _, row in druglike_compounds.iterrows():
            report_lines.append(
                f"| {row['COMPOUND_ID']} | {row['source_library']} | {row['ensemble_prediction']:.3f} | "
                f"{row['MW']:.1f} | {row['LogP']:.2f} | {row['QED']:.3f} |"
            )

    # Generation Factors Analysis
    if generation_factors:
        gen_lines = format_generation_factors_report(generation_factors, model_name)
        report_lines.extend(gen_lines)

    report = '\n'.join(report_lines)

    # Save report
    report_path = os.path.join(output_dir, f'{model_name}_tier3_analysis_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"Report saved: {report_path}")
    return report_path


def run_analysis(model_name: str,
                 input_csv: str,
                 pred_csv: str,
                 output_dir: str) -> Dict:
    """
    Run complete Tier 3 analysis for a model.

    Args:
        model_name: Model identifier (SA, EC, CA)
        input_csv: Path to input CSV with generation metadata
        pred_csv: Path to prediction CSV with model results
        output_dir: Directory to save outputs

    Returns:
        Dictionary with analysis results
    """
    print(f"\n{'='*60}")
    print(f"TIER 3 ANALYSIS: {model_name} MODEL")
    print(f"{'='*60}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load and merge data
    df = load_and_merge_data(input_csv, pred_csv)

    # Run analyses
    overall_stats = analyze_overall_predictions(df, model_name)
    print(f"\nOverall: {overall_stats['predicted_active']}/{overall_stats['total_compounds']} active ({overall_stats['active_rate']:.1f}%)")

    tier3_active, tier3_inactive, tier3_stats = analyze_tier3(df, model_name)
    print(f"Tier 3: {tier3_stats['tier3_active']}/{tier3_stats['tier3_total']} active ({tier3_stats['tier3_active_rate']:.1f}%)")

    library_stats = analyze_by_library(df, tier3_active)
    acid_fragments, amine_fragments = analyze_fragments(tier3_active)
    xai_stats = analyze_xai_attributions(tier3_active)
    property_stats = analyze_properties(tier3_active, tier3_inactive)
    top_compounds = get_top_compounds(tier3_active)
    druglike_compounds = get_druglike_compounds(tier3_active)
    generation_factors = analyze_generation_factors(tier3_active, tier3_inactive)

    # Generate report
    report_path = generate_report(
        model_name, df, tier3_active, tier3_inactive,
        overall_stats, tier3_stats, library_stats,
        acid_fragments, amine_fragments, xai_stats,
        property_stats, top_compounds, druglike_compounds,
        generation_factors, output_dir
    )

    # Save Tier 3 actives CSV
    tier3_active_path = os.path.join(output_dir, f'{model_name}_tier3_actives.csv')
    tier3_active.to_csv(tier3_active_path, index=False)
    print(f"Tier 3 actives saved: {tier3_active_path}")

    # Save library stats
    library_stats_path = os.path.join(output_dir, f'{model_name}_library_stats.csv')
    library_stats.to_csv(library_stats_path, index=False)
    print(f"Library stats saved: {library_stats_path}")

    return {
        'model': model_name,
        'overall_stats': overall_stats,
        'tier3_stats': tier3_stats,
        'library_stats': library_stats.to_dict('records'),
        'xai_stats': xai_stats,
        'generation_factors': generation_factors,
        'report_path': report_path,
        'tier3_active_path': tier3_active_path,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Tier 3 Prediction Analysis for FBDD XAI Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze single model
    python tier3_prediction_analysis.py --model SA

    # Analyze all models
    python tier3_prediction_analysis.py --all

    # Custom paths
    python tier3_prediction_analysis.py --model SA --input-csv my_input.csv --pred-csv my_pred.csv
        """
    )

    parser.add_argument('--model', choices=['SA', 'EC', 'CA'],
                        help='Model to analyze')
    parser.add_argument('--all', action='store_true',
                        help='Analyze all three models (SA, EC, CA)')
    parser.add_argument('--input-csv', type=str,
                        help='Path to input CSV with generation metadata')
    parser.add_argument('--pred-csv', type=str,
                        help='Path to prediction CSV with model results')
    parser.add_argument('--output-dir', type=str, default='analysis_outputs',
                        help='Output directory for reports and CSVs')
    parser.add_argument('--base-dir', type=str, default='.',
                        help='Base directory for default file paths')

    args = parser.parse_args()

    if not args.model and not args.all:
        parser.error("Must specify --model or --all")

    # Determine models to analyze
    models = ['SA', 'EC', 'CA'] if args.all else [args.model]

    results = []
    for model in models:
        # Determine file paths
        if args.input_csv and args.pred_csv:
            input_csv = args.input_csv
            pred_csv = args.pred_csv
        else:
            # Default paths
            input_csv = os.path.join(args.base_dir, f'genmol_all_input_{model}.csv')
            pred_csv = os.path.join(args.base_dir, f'genmol_all_input_{model}_prediction.csv')

        # Check files exist
        if not os.path.exists(input_csv):
            print(f"ERROR: Input file not found: {input_csv}")
            continue
        if not os.path.exists(pred_csv):
            print(f"ERROR: Prediction file not found: {pred_csv}")
            continue

        # Run analysis
        result = run_analysis(model, input_csv, pred_csv, args.output_dir)
        results.append(result)

    # Print summary
    if len(results) > 1:
        print(f"\n{'='*60}")
        print("CROSS-MODEL SUMMARY")
        print(f"{'='*60}")
        print(f"\n| Model | Total | Active | Active % | Tier 3 | Tier 3 Active | Tier 3 Hit Rate |")
        print(f"|-------|-------|--------|----------|--------|---------------|-----------------|")
        for r in results:
            o = r['overall_stats']
            t = r['tier3_stats']
            print(f"| {r['model']} | {o['total_compounds']} | {o['predicted_active']} | "
                  f"{o['active_rate']:.1f}% | {t['tier3_total']} | {t['tier3_active']} | "
                  f"{t['tier3_active_rate']:.1f}% |")

    print(f"\nAnalysis complete!")
    return results


if __name__ == '__main__':
    main()
