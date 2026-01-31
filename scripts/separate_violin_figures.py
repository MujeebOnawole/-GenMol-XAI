"""
Separate Violin Plot Figures for Manuscript
Generates three individual figures (LogP, TPSA, HBD) instead of combined panels.

Author: Generated for GenMol XAI Analysis
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.patches import Patch

# Set publication-quality defaults
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Colors for all 7 libraries
ALL_COLORS = {
    'SA': '#DC143C',      # Crimson
    'EC': '#1E90FF',      # Dodger Blue
    'CA': '#228B22',      # Forest Green
    'SA_EC': '#8B008B',   # Dark Magenta
    'SA_CA': '#FF8C00',   # Dark Orange
    'CA_EC': '#20B2AA',   # Light Sea Green
    'TRIPLE': '#DAA520',  # Goldenrod
}

ALL_LABELS = {
    'SA': 'SA',
    'EC': 'EC',
    'CA': 'CA',
    'SA_EC': 'SA+EC',
    'SA_CA': 'SA+CA',
    'CA_EC': 'CA+EC',
    'TRIPLE': 'Triple',
}

LIBRARY_ORDER = ['SA', 'EC', 'CA', 'SA_EC', 'SA_CA', 'CA_EC', 'TRIPLE']


def load_all_libraries(base_dir: Path) -> pd.DataFrame:
    """Load all 7 libraries."""
    all_fragments = []
    libraries_dir = base_dir / 'libraries'

    for lib_name in LIBRARY_ORDER:
        lib_dir = libraries_dir / f'{lib_name}_library'
        json_file = lib_dir / f'safe_library_{lib_name}.json'

        if not json_file.exists():
            print(f"Warning: {json_file} not found, skipping...")
            continue

        with open(json_file, 'r') as f:
            fragments = json.load(f)

        print(f"Loaded {len(fragments)} fragments from {lib_name}")

        for frag in fragments:
            props = frag.get('props', {})
            all_fragments.append({
                'library': lib_name,
                'MW': props.get('MW'),
                'LogP': props.get('LogP'),
                'TPSA': props.get('TPSA'),
                'HBD': props.get('HBD'),
                'HBA': props.get('HBA'),
            })

    return pd.DataFrame(all_fragments)


def create_single_property_violin(df: pd.DataFrame, property_name: str,
                                   ylabel: str, title: str, output_path: Path):
    """
    Create a single violin plot figure for one property.

    Parameters:
    -----------
    df : DataFrame with fragment properties
    property_name : Column name in df (e.g., 'LogP', 'TPSA', 'HBD')
    ylabel : Y-axis label
    title : Figure title
    output_path : Path to save figure
    """

    # Create display labels
    df_plot = df.copy()
    df_plot['Library'] = df_plot['library'].map(ALL_LABELS)

    # Single figure
    fig, ax = plt.subplots(figsize=(8, 6))

    order = ['SA', 'EC', 'CA', 'SA+EC', 'SA+CA', 'CA+EC', 'Triple']
    palette = [ALL_COLORS['SA'], ALL_COLORS['EC'], ALL_COLORS['CA'],
               ALL_COLORS['SA_EC'], ALL_COLORS['SA_CA'], ALL_COLORS['CA_EC'],
               ALL_COLORS['TRIPLE']]

    # Create violin plot
    sns.violinplot(data=df_plot, x='Library', y=property_name, order=order,
                   palette=palette, ax=ax, inner='box', linewidth=0.7, scale='width')

    ax.set_xlabel('')
    ax.set_ylabel(ylabel, fontweight='bold', fontsize=12)
    ax.set_title(title, fontweight='bold', fontsize=14)

    # Set x-tick labels with bold font
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=45, ha='right', fontweight='bold', fontsize=10)

    # Make y-axis tick labels bold
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(10)

    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)

    # Add legend with counts at the bottom - BOLD and BIGGER font
    lib_counts = df_plot.groupby('Library').size()
    legend_text = '  '.join([f"{lib}: n={lib_counts.get(lib, 0)}" for lib in order])
    fig.text(0.5, -0.02, legend_text, ha='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)

    # Save as PNG and PDF
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')

    print(f"Saved: {output_path}")
    print(f"Saved: {output_path.with_suffix('.pdf')}")

    plt.close(fig)
    return fig


def create_combined_violin(df: pd.DataFrame, output_path: Path):
    """
    Create a combined figure with all three violin plots as panels (a), (b), (c).
    """

    # Create display labels
    df_plot = df.copy()
    df_plot['Library'] = df_plot['library'].map(ALL_LABELS)

    # Create figure with 1 row, 3 columns
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    order = ['SA', 'EC', 'CA', 'SA+EC', 'SA+CA', 'CA+EC', 'Triple']
    palette = [ALL_COLORS['SA'], ALL_COLORS['EC'], ALL_COLORS['CA'],
               ALL_COLORS['SA_EC'], ALL_COLORS['SA_CA'], ALL_COLORS['CA_EC'],
               ALL_COLORS['TRIPLE']]

    # Properties to plot
    properties = [
        ('LogP', 'LogP', 'Lipophilicity'),
        ('TPSA', 'TPSA (Å²)', 'Polar Surface Area'),
        ('HBD', 'H-Bond Donors', 'Hydrogen Bonding'),
    ]

    for idx, (prop, ylabel, title) in enumerate(properties):
        ax = axes[idx]

        # Create violin plot
        sns.violinplot(data=df_plot, x='Library', y=prop, order=order,
                       palette=palette, ax=ax, inner='box', linewidth=0.7, scale='width')

        ax.set_xlabel('')
        ax.set_ylabel(ylabel, fontweight='bold', fontsize=12)
        ax.set_title(title, fontweight='bold', fontsize=14)

        # Set x-tick labels with bold font
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=45, ha='right', fontweight='bold', fontsize=10)

        # Make y-axis tick labels bold
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')
            label.set_fontsize(10)

        ax.yaxis.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)

        # Add panel label (a), (b), (c)
        ax.text(-0.12, 1.05, f'({chr(97+idx)})', transform=ax.transAxes,
                fontsize=14, fontweight='bold')

    # Add legend with counts at the bottom - BOLD and BIGGER font
    lib_counts = df_plot.groupby('Library').size()
    legend_text = '  '.join([f"{lib}: n={lib_counts.get(lib, 0)}" for lib in order])
    fig.text(0.5, -0.02, legend_text, ha='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.20)

    # Save as PNG and PDF
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')

    print(f"Saved: {output_path}")
    print(f"Saved: {output_path.with_suffix('.pdf')}")

    plt.close(fig)
    return fig


def main():
    """Generate three separate violin figures and a combined figure."""

    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / 'results'
    output_dir.mkdir(exist_ok=True)

    print("="*60)
    print("Generating Separate Violin Figures")
    print("="*60)

    print("\nLoading all libraries...")
    df = load_all_libraries(base_dir)
    print(f"Total fragments: {len(df)}")

    # Print counts per library
    print("\nFragments per library:")
    for lib in LIBRARY_ORDER:
        count = len(df[df['library'] == lib])
        print(f"  {ALL_LABELS[lib]:>8}: {count:>5} fragments")

    print("\n" + "="*60)
    print("Generating individual figures...")
    print("="*60 + "\n")

    # Figure 1: LogP (Lipophilicity)
    create_single_property_violin(
        df,
        property_name='LogP',
        ylabel='LogP',
        title='Lipophilicity',
        output_path=output_dir / 'figure_violin_LogP.png'
    )

    # Figure 2: TPSA (Polar Surface Area)
    create_single_property_violin(
        df,
        property_name='TPSA',
        ylabel='TPSA (Å²)',
        title='Polar Surface Area',
        output_path=output_dir / 'figure_violin_TPSA.png'
    )

    # Figure 3: HBD (Hydrogen Bond Donors)
    create_single_property_violin(
        df,
        property_name='HBD',
        ylabel='H-Bond Donors',
        title='Hydrogen Bonding',
        output_path=output_dir / 'figure_violin_HBD.png'
    )

    # Figure 4: Combined figure with all three panels
    print("\nGenerating combined figure...")
    create_combined_violin(
        df,
        output_path=output_dir / 'figure_violin_combined.png'
    )

    print("\n" + "="*60)
    print("Complete! Generated 3 separate figures + 1 combined:")
    print("="*60)
    print(f"  1. {output_dir / 'figure_violin_LogP.png'}")
    print(f"  2. {output_dir / 'figure_violin_TPSA.png'}")
    print(f"  3. {output_dir / 'figure_violin_HBD.png'}")
    print(f"  4. {output_dir / 'figure_violin_combined.png'} (all panels)")
    print("\nPDF versions also saved.")


if __name__ == '__main__':
    main()
