"""
Library Diversity Visualization Script
Generates a publication-ready 4-panel figure showing physicochemical property
distributions across all 7 fragment libraries.

Author: Generated for GenMol XAI Analysis
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# Set publication-quality defaults
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Define library order and colors (logical grouping)
LIBRARY_ORDER = ['SA', 'EC', 'CA', 'SA_EC', 'SA_CA', 'CA_EC', 'TRIPLE']

# Color scheme: single pathogens = primary colors, duals = blends, triple = gold
LIBRARY_COLORS = {
    'SA': '#DC143C',      # Crimson (Gram-positive)
    'EC': '#1E90FF',      # Dodger Blue (Gram-negative)
    'CA': '#228B22',      # Forest Green (Fungal)
    'SA_EC': '#8B008B',   # Dark Magenta (red+blue blend)
    'SA_CA': '#FF8C00',   # Dark Orange (red+green blend)
    'CA_EC': '#20B2AA',   # Light Sea Green (blue+green blend)
    'TRIPLE': '#DAA520',  # Goldenrod (universal)
}

# Display labels for libraries
LIBRARY_LABELS = {
    'SA': 'SA',
    'EC': 'EC',
    'CA': 'CA',
    'SA_EC': 'SA+EC',
    'SA_CA': 'SA+CA',
    'CA_EC': 'CA+EC',
    'TRIPLE': 'Triple',
}


def load_library(json_path: Path) -> list:
    """Load a single library JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def load_all_libraries(base_dir: Path) -> pd.DataFrame:
    """Load all 7 libraries and combine into a DataFrame."""
    all_fragments = []

    libraries_dir = base_dir / 'libraries'

    for lib_name in LIBRARY_ORDER:
        lib_dir = libraries_dir / f'{lib_name}_library'
        json_file = lib_dir / f'safe_library_{lib_name}.json'

        if not json_file.exists():
            print(f"Warning: {json_file} not found, skipping...")
            continue

        fragments = load_library(json_file)
        print(f"Loaded {len(fragments)} fragments from {lib_name}")

        for frag in fragments:
            props = frag.get('props', {})
            all_fragments.append({
                'library': lib_name,
                'fragment_id': frag.get('fragment_id'),
                'smiles': frag.get('fragment_smiles'),
                'role': frag.get('role'),
                'MW': props.get('MW'),
                'LogP': props.get('LogP'),
                'TPSA': props.get('TPSA'),
                'HBD': props.get('HBD'),
                'HBA': props.get('HBA'),
                'AromRings': props.get('AromRings'),
                'RotBonds': props.get('RotBonds'),
            })

    df = pd.DataFrame(all_fragments)

    # Calculate total rings (AromRings + non-aromatic estimated)
    # Note: The JSON only has AromRings, so we'll use that
    df['Rings'] = df['AromRings']  # Using aromatic rings as proxy

    return df


def calculate_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate summary statistics per library."""
    stats_list = []

    for lib in LIBRARY_ORDER:
        lib_df = df[df['library'] == lib]
        if len(lib_df) == 0:
            continue

        stats_list.append({
            'Library': LIBRARY_LABELS[lib],
            'N': len(lib_df),
            'MW (mean)': f"{lib_df['MW'].mean():.1f}",
            'MW (range)': f"{lib_df['MW'].min():.0f}-{lib_df['MW'].max():.0f}",
            'LogP (mean)': f"{lib_df['LogP'].mean():.2f}",
            'LogP (range)': f"{lib_df['LogP'].min():.2f}-{lib_df['LogP'].max():.2f}",
            'TPSA (mean)': f"{lib_df['TPSA'].mean():.1f}",
            'HBD (mean)': f"{lib_df['HBD'].mean():.2f}",
            'HBA (mean)': f"{lib_df['HBA'].mean():.2f}",
        })

    return pd.DataFrame(stats_list)


def create_diversity_figure(df: pd.DataFrame, output_path: Path,
                           exemplars: dict = None):
    """
    Create a 4-panel figure showing property distributions.

    Parameters:
    -----------
    df : DataFrame with fragment properties
    output_path : Path to save figure
    exemplars : Optional dict of exemplar compounds to overlay
    """

    # Create figure with 2x2 layout
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    # Properties to plot with their labels and units
    properties = [
        ('LogP', 'LogP', None),
        ('TPSA', 'TPSA', 'Å²'),
        ('HBD', 'H-Bond Donors', None),
        ('MW', 'Molecular Weight', 'Da'),
    ]

    # Create color palette in library order
    palette = [LIBRARY_COLORS[lib] for lib in LIBRARY_ORDER]

    for idx, (prop, label, unit) in enumerate(properties):
        ax = axes[idx]

        # Create violin plot
        sns.violinplot(
            data=df,
            x='library',
            y=prop,
            order=LIBRARY_ORDER,
            palette=palette,
            ax=ax,
            inner='box',  # Show box plot inside violin
            linewidth=0.8,
            saturation=0.85,
        )

        # Customize axis labels
        ylabel = f'{label}' if unit is None else f'{label} ({unit})'
        ax.set_ylabel(ylabel)
        ax.set_xlabel('')

        # Update x-tick labels to display labels
        ax.set_xticklabels([LIBRARY_LABELS[lib] for lib in LIBRARY_ORDER],
                          rotation=45, ha='right')

        # Add panel label (a, b, c, d)
        ax.text(-0.12, 1.05, f'({chr(97+idx)})', transform=ax.transAxes,
                fontsize=12, fontweight='bold', va='top')

        # Add subtle grid
        ax.yaxis.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)

    # Adjust layout
    plt.tight_layout()

    # Add a single legend at the bottom
    # Create custom legend handles
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=LIBRARY_COLORS[lib],
                           edgecolor='black',
                           linewidth=0.5,
                           label=LIBRARY_LABELS[lib])
                     for lib in LIBRARY_ORDER]

    fig.legend(handles=legend_handles,
              loc='lower center',
              ncol=7,
              bbox_to_anchor=(0.5, -0.02),
              frameon=False,
              fontsize=9)

    # Adjust bottom margin for legend
    plt.subplots_adjust(bottom=0.12)

    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight',
                facecolor='white', edgecolor='none')

    print(f"Figure saved to {output_path}")
    print(f"PDF saved to {output_path.with_suffix('.pdf')}")

    return fig


def create_boxplot_version(df: pd.DataFrame, output_path: Path):
    """
    Alternative: Create a cleaner boxplot version if violins are too busy.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    properties = [
        ('LogP', 'LogP', None),
        ('TPSA', 'TPSA', 'Å²'),
        ('HBD', 'H-Bond Donors', None),
        ('MW', 'Molecular Weight', 'Da'),
    ]

    palette = [LIBRARY_COLORS[lib] for lib in LIBRARY_ORDER]

    for idx, (prop, label, unit) in enumerate(properties):
        ax = axes[idx]

        sns.boxplot(
            data=df,
            x='library',
            y=prop,
            order=LIBRARY_ORDER,
            palette=palette,
            ax=ax,
            linewidth=0.8,
            fliersize=2,
            width=0.7,
        )

        ylabel = f'{label}' if unit is None else f'{label} ({unit})'
        ax.set_ylabel(ylabel)
        ax.set_xlabel('')
        ax.set_xticklabels([LIBRARY_LABELS[lib] for lib in LIBRARY_ORDER],
                          rotation=45, ha='right')
        ax.text(-0.12, 1.05, f'({chr(97+idx)})', transform=ax.transAxes,
                fontsize=12, fontweight='bold', va='top')
        ax.yaxis.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)

    plt.tight_layout()

    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=LIBRARY_COLORS[lib],
                           edgecolor='black', linewidth=0.5,
                           label=LIBRARY_LABELS[lib])
                     for lib in LIBRARY_ORDER]

    fig.legend(handles=legend_handles, loc='lower center', ncol=7,
              bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=9)
    plt.subplots_adjust(bottom=0.12)

    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight',
                facecolor='white', edgecolor='none')

    print(f"Boxplot figure saved to {output_path}")

    return fig


def run_statistical_tests(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run Kruskal-Wallis tests to confirm library differences.
    """
    results = []
    properties = ['LogP', 'TPSA', 'HBD', 'MW', 'HBA']

    for prop in properties:
        # Get groups
        groups = [df[df['library'] == lib][prop].dropna().values
                  for lib in LIBRARY_ORDER]
        groups = [g for g in groups if len(g) > 0]

        # Kruskal-Wallis test
        stat, p_value = stats.kruskal(*groups)

        results.append({
            'Property': prop,
            'H-statistic': f"{stat:.2f}",
            'p-value': f"{p_value:.2e}" if p_value < 0.001 else f"{p_value:.4f}",
            'Significant': 'Yes' if p_value < 0.05 else 'No'
        })

    return pd.DataFrame(results)


def main():
    """Main execution function."""

    # Set base directory
    base_dir = Path(__file__).parent.parent

    print("="*60)
    print("Library Diversity Analysis")
    print("="*60)

    # Load all libraries
    print("\nLoading fragment libraries...")
    df = load_all_libraries(base_dir)
    print(f"\nTotal fragments loaded: {len(df)}")

    # Print summary per library
    print("\nFragments per library:")
    for lib in LIBRARY_ORDER:
        count = len(df[df['library'] == lib])
        print(f"  {LIBRARY_LABELS[lib]:>8}: {count:>5} fragments")

    # Calculate summary statistics
    print("\n" + "="*60)
    print("Summary Statistics")
    print("="*60)
    summary_df = calculate_summary_stats(df)
    print(summary_df.to_string(index=False))

    # Save summary to CSV
    output_dir = base_dir / 'results'
    output_dir.mkdir(exist_ok=True)
    summary_df.to_csv(output_dir / 'library_diversity_summary.csv', index=False)
    print(f"\nSummary saved to {output_dir / 'library_diversity_summary.csv'}")

    # Run statistical tests
    print("\n" + "="*60)
    print("Statistical Tests (Kruskal-Wallis)")
    print("="*60)
    stats_df = run_statistical_tests(df)
    print(stats_df.to_string(index=False))
    stats_df.to_csv(output_dir / 'library_diversity_statistics.csv', index=False)

    # Create violin plot figure
    print("\n" + "="*60)
    print("Generating Figures")
    print("="*60)

    fig_path = output_dir / 'library_diversity_violin.png'
    create_diversity_figure(df, fig_path)

    # Also create boxplot version
    box_path = output_dir / 'library_diversity_boxplot.png'
    create_boxplot_version(df, box_path)

    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)
    print(f"\nOutput files:")
    print(f"  - {output_dir / 'library_diversity_summary.csv'}")
    print(f"  - {output_dir / 'library_diversity_statistics.csv'}")
    print(f"  - {fig_path}")
    print(f"  - {fig_path.with_suffix('.pdf')}")
    print(f"  - {box_path}")
    print(f"  - {box_path.with_suffix('.pdf')}")

    return df, summary_df


if __name__ == '__main__':
    df, summary = main()
