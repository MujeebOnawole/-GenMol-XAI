"""
Chemical Space Diversity Visualization
2D scatter plots showing property pairs (LogP vs MW, TPSA vs LogP, etc.)
colored by library - showing where each library occupies chemical space.

Author: Generated for GenMol XAI Analysis
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Define library order and colors
LIBRARY_ORDER = ['SA', 'EC', 'CA', 'SA_EC', 'SA_CA', 'CA_EC', 'TRIPLE']

LIBRARY_COLORS = {
    'SA': '#DC143C',      # Crimson (Gram-positive)
    'EC': '#1E90FF',      # Dodger Blue (Gram-negative)
    'CA': '#228B22',      # Forest Green (Fungal)
    'SA_EC': '#8B008B',   # Dark Magenta
    'SA_CA': '#FF8C00',   # Dark Orange
    'CA_EC': '#20B2AA',   # Light Sea Green
    'TRIPLE': '#DAA520',  # Goldenrod
}

LIBRARY_LABELS = {
    'SA': 'SA',
    'EC': 'EC',
    'CA': 'CA',
    'SA_EC': 'SA+EC',
    'SA_CA': 'SA+CA',
    'CA_EC': 'CA+EC',
    'TRIPLE': 'Triple',
}


def load_all_libraries(base_dir: Path) -> pd.DataFrame:
    """Load all 7 libraries and combine into a DataFrame."""
    all_fragments = []
    libraries_dir = base_dir / 'libraries'

    for lib_name in LIBRARY_ORDER:
        lib_dir = libraries_dir / f'{lib_name}_library'
        json_file = lib_dir / f'safe_library_{lib_name}.json'

        if not json_file.exists():
            continue

        with open(json_file, 'r') as f:
            fragments = json.load(f)

        for frag in fragments:
            props = frag.get('props', {})
            all_fragments.append({
                'library': lib_name,
                'MW': props.get('MW'),
                'LogP': props.get('LogP'),
                'TPSA': props.get('TPSA'),
                'HBD': props.get('HBD'),
                'HBA': props.get('HBA'),
                'AromRings': props.get('AromRings'),
                'RotBonds': props.get('RotBonds'),
            })

    return pd.DataFrame(all_fragments)


def create_chemical_space_figure(df: pd.DataFrame, output_path: Path):
    """
    Create 4-panel chemical space figure with conventional property pairs.

    Panels:
    (a) LogP vs MW - Classic drug-likeness space
    (b) TPSA vs LogP - Membrane permeability space
    (c) HBD vs TPSA - Polar surface/H-bonding (Gram-negative relevance)
    (d) HBA vs MW - Size vs acceptor capacity
    """

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    axes = axes.flatten()

    # Property pairs to plot (x, y, xlabel, ylabel)
    property_pairs = [
        ('MW', 'LogP', 'Molecular Weight (Da)', 'LogP'),
        ('LogP', 'TPSA', 'LogP', 'TPSA (Å²)'),
        ('TPSA', 'HBD', 'TPSA (Å²)', 'H-Bond Donors'),
        ('MW', 'HBA', 'Molecular Weight (Da)', 'H-Bond Acceptors'),
    ]

    # Plot each library with transparency, smaller libraries on top
    # Sort by size (largest first) so smaller libraries are visible on top
    library_sizes = df.groupby('library').size().to_dict()
    sorted_libraries = sorted(LIBRARY_ORDER, key=lambda x: library_sizes.get(x, 0), reverse=True)

    for idx, (x_prop, y_prop, xlabel, ylabel) in enumerate(property_pairs):
        ax = axes[idx]

        # Plot each library
        for lib in sorted_libraries:
            lib_df = df[df['library'] == lib]

            ax.scatter(
                lib_df[x_prop],
                lib_df[y_prop],
                c=LIBRARY_COLORS[lib],
                s=12,
                alpha=0.5,
                edgecolors='none',
                label=LIBRARY_LABELS[lib],
            )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.text(-0.12, 1.05, f'({chr(97+idx)})', transform=ax.transAxes,
                fontsize=12, fontweight='bold', va='top')

        # Add subtle grid
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)

    plt.tight_layout()

    # Single legend at bottom
    legend_handles = [Patch(facecolor=LIBRARY_COLORS[lib], alpha=0.7,
                           edgecolor='none', label=LIBRARY_LABELS[lib])
                     for lib in LIBRARY_ORDER]

    fig.legend(handles=legend_handles, loc='lower center', ncol=7,
              bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=9)
    plt.subplots_adjust(bottom=0.10)

    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight',
                facecolor='white', edgecolor='none')

    print(f"Chemical space figure saved to {output_path}")
    return fig


def create_key_space_figure(df: pd.DataFrame, output_path: Path):
    """
    Create focused 2-panel figure with the two most informative property pairs:
    - LogP vs MW: Classic drug space
    - TPSA vs LogP: Shows Gram+/Gram-/Fungal separation clearly
    """

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Sort libraries by size for plotting order
    library_sizes = df.groupby('library').size().to_dict()
    sorted_libraries = sorted(LIBRARY_ORDER, key=lambda x: library_sizes.get(x, 0), reverse=True)

    # Panel (a): LogP vs MW
    ax = axes[0]
    for lib in sorted_libraries:
        lib_df = df[df['library'] == lib]
        ax.scatter(lib_df['MW'], lib_df['LogP'],
                  c=LIBRARY_COLORS[lib], s=15, alpha=0.5,
                  edgecolors='none', label=LIBRARY_LABELS[lib])

    ax.set_xlabel('Molecular Weight (Da)')
    ax.set_ylabel('LogP')
    ax.text(-0.10, 1.05, '(a)', transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)

    # Panel (b): TPSA vs LogP - Key for showing pathogen separation!
    ax = axes[1]
    for lib in sorted_libraries:
        lib_df = df[df['library'] == lib]
        ax.scatter(lib_df['LogP'], lib_df['TPSA'],
                  c=LIBRARY_COLORS[lib], s=15, alpha=0.5,
                  edgecolors='none', label=LIBRARY_LABELS[lib])

    ax.set_xlabel('LogP')
    ax.set_ylabel('TPSA (Å²)')
    ax.text(-0.10, 1.05, '(b)', transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()

    # Legend
    legend_handles = [Patch(facecolor=LIBRARY_COLORS[lib], alpha=0.7,
                           edgecolor='none', label=LIBRARY_LABELS[lib])
                     for lib in LIBRARY_ORDER]

    fig.legend(handles=legend_handles, loc='lower center', ncol=7,
              bbox_to_anchor=(0.5, -0.05), frameon=False, fontsize=9)
    plt.subplots_adjust(bottom=0.15)

    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight',
                facecolor='white', edgecolor='none')

    print(f"Key space figure saved to {output_path}")
    return fig


def create_single_pathogens_focus(df: pd.DataFrame, output_path: Path):
    """
    Create figure focusing on single-pathogen libraries (SA, EC, CA)
    to clearly show SELECT zone separation.
    """

    # Filter to single-pathogen libraries only
    single_df = df[df['library'].isin(['SA', 'EC', 'CA'])]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot order: largest first so smaller visible on top
    plot_order = ['SA', 'EC', 'CA']  # SA has most, CA in middle

    # Panel (a): LogP vs MW
    ax = axes[0]
    for lib in plot_order:
        lib_df = single_df[single_df['library'] == lib]
        ax.scatter(lib_df['MW'], lib_df['LogP'],
                  c=LIBRARY_COLORS[lib], s=20, alpha=0.6,
                  edgecolors='none', label=LIBRARY_LABELS[lib])

    ax.set_xlabel('Molecular Weight (Da)')
    ax.set_ylabel('LogP')
    ax.text(-0.10, 1.05, '(a)', transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc='upper right', framealpha=0.9)

    # Panel (b): TPSA vs LogP - Shows the separation best!
    ax = axes[1]
    for lib in plot_order:
        lib_df = single_df[single_df['library'] == lib]
        ax.scatter(lib_df['LogP'], lib_df['TPSA'],
                  c=LIBRARY_COLORS[lib], s=20, alpha=0.6,
                  edgecolors='none', label=LIBRARY_LABELS[lib])

    ax.set_xlabel('LogP')
    ax.set_ylabel('TPSA (Å²)')
    ax.text(-0.10, 1.05, '(b)', transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc='upper right', framealpha=0.9)

    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight',
                facecolor='white', edgecolor='none')

    print(f"Single pathogen focus figure saved to {output_path}")
    return fig


def create_comprehensive_figure(df: pd.DataFrame, output_path: Path):
    """
    Create a comprehensive 4-panel figure optimized for SELECT framework validation:
    (a) LogP vs MW - drug space
    (b) TPSA vs LogP - permeability (key for pathogen separation)
    (c) HBD vs LogP - Gram-negative entry requirements
    (d) TPSA vs MW - size-polarity relationship
    """

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    axes = axes.flatten()

    # Property pairs optimized for SELECT framework
    property_pairs = [
        ('MW', 'LogP', 'Molecular Weight (Da)', 'LogP'),
        ('LogP', 'TPSA', 'LogP', 'TPSA (Å²)'),
        ('LogP', 'HBD', 'LogP', 'H-Bond Donors'),
        ('MW', 'TPSA', 'Molecular Weight (Da)', 'TPSA (Å²)'),
    ]

    # Sort libraries by size
    library_sizes = df.groupby('library').size().to_dict()
    sorted_libraries = sorted(LIBRARY_ORDER, key=lambda x: library_sizes.get(x, 0), reverse=True)

    for idx, (x_prop, y_prop, xlabel, ylabel) in enumerate(property_pairs):
        ax = axes[idx]

        for lib in sorted_libraries:
            lib_df = df[df['library'] == lib]
            ax.scatter(
                lib_df[x_prop],
                lib_df[y_prop],
                c=LIBRARY_COLORS[lib],
                s=10,
                alpha=0.45,
                edgecolors='none',
                label=LIBRARY_LABELS[lib],
            )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.text(-0.10, 1.05, f'({chr(97+idx)})', transform=ax.transAxes,
                fontsize=12, fontweight='bold', va='top')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)

    plt.tight_layout()

    # Legend
    legend_handles = [Patch(facecolor=LIBRARY_COLORS[lib], alpha=0.7,
                           edgecolor='none', label=LIBRARY_LABELS[lib])
                     for lib in LIBRARY_ORDER]

    fig.legend(handles=legend_handles, loc='lower center', ncol=7,
              bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=9)
    plt.subplots_adjust(bottom=0.08)

    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight',
                facecolor='white', edgecolor='none')

    print(f"Comprehensive figure saved to {output_path}")
    return fig


def main():
    """Generate chemical space visualizations."""

    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / 'results'
    output_dir.mkdir(exist_ok=True)

    print("Loading fragment libraries...")
    df = load_all_libraries(base_dir)
    print(f"Total fragments: {len(df)}")

    print("\nGenerating chemical space visualizations...")

    # Option 1: 4-panel with different property pairs
    create_chemical_space_figure(df, output_dir / 'chemical_space_4panel.png')

    # Option 2: 2-panel focused on key spaces
    create_key_space_figure(df, output_dir / 'chemical_space_2panel.png')

    # Option 3: Single pathogen focus (SA, EC, CA only)
    create_single_pathogens_focus(df, output_dir / 'chemical_space_single_pathogens.png')

    # Option 4: Comprehensive 4-panel optimized for SELECT
    create_comprehensive_figure(df, output_dir / 'chemical_space_comprehensive.png')

    print("\n" + "="*60)
    print("Chemical space visualizations complete!")
    print("="*60)
    print("\nOutput files:")
    print("  1. chemical_space_4panel.png - 4 property pairs")
    print("  2. chemical_space_2panel.png - Key spaces (LogP-MW, TPSA-LogP)")
    print("  3. chemical_space_single_pathogens.png - SA/EC/CA only (cleaner)")
    print("  4. chemical_space_comprehensive.png - SELECT-optimized panels")


if __name__ == '__main__':
    main()
