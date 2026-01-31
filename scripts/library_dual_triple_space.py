"""
Dual-Active and Triple-Active Chemical Space Visualization
Separated panels for each library type.

Author: Generated for GenMol XAI Analysis
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set publication-quality defaults
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Library configurations
DUAL_LIBRARIES = ['SA_EC', 'SA_CA', 'CA_EC']
TRIPLE_LIBRARY = ['TRIPLE']

LIBRARY_COLORS = {
    'SA_EC': '#8B008B',   # Dark Magenta
    'SA_CA': '#FF8C00',   # Dark Orange
    'CA_EC': '#20B2AA',   # Light Sea Green
    'TRIPLE': '#DAA520',  # Goldenrod
}

LIBRARY_LABELS = {
    'SA_EC': 'S. aureus + E. coli',
    'SA_CA': 'S. aureus + C. albicans',
    'CA_EC': 'C. albicans + E. coli',
    'TRIPLE': 'Triple-active (SA + EC + CA)',
}


def load_libraries(base_dir: Path, library_list: list) -> pd.DataFrame:
    """Load specified libraries."""
    all_fragments = []
    libraries_dir = base_dir / 'libraries'

    for lib_name in library_list:
        lib_dir = libraries_dir / f'{lib_name}_library'
        json_file = lib_dir / f'safe_library_{lib_name}.json'

        if not json_file.exists():
            print(f"Warning: {json_file} not found")
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


def create_dual_active_separated(df: pd.DataFrame, output_path: Path):
    """
    Create separated panels figure for dual-active libraries.
    3 rows (SA+EC, SA+CA, CA+EC) x 2 columns (MW vs LogP, LogP vs TPSA)
    """

    fig, axes = plt.subplots(3, 2, figsize=(11, 12))

    for row_idx, lib in enumerate(DUAL_LIBRARIES):
        lib_df = df[df['library'] == lib]

        # Left column: MW vs LogP
        ax = axes[row_idx, 0]
        ax.scatter(lib_df['MW'], lib_df['LogP'],
                  c=LIBRARY_COLORS[lib], s=20, alpha=0.6, edgecolors='none')
        ax.set_xlabel('Molecular Weight (Da)')
        ax.set_ylabel('LogP')
        ax.set_title(f'{LIBRARY_LABELS[lib]} (n={len(lib_df)})', fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)
        ax.set_xlim(0, 650)
        ax.set_ylim(-6, 12)

        # Right column: LogP vs TPSA
        ax = axes[row_idx, 1]
        ax.scatter(lib_df['LogP'], lib_df['TPSA'],
                  c=LIBRARY_COLORS[lib], s=20, alpha=0.6, edgecolors='none')
        ax.set_xlabel('LogP')
        ax.set_ylabel('TPSA (Å²)')
        ax.set_title(f'{LIBRARY_LABELS[lib]} (n={len(lib_df)})', fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)
        ax.set_xlim(-6, 12)
        ax.set_ylim(0, 225)

    # Add panel labels
    axes[0, 0].text(-0.12, 1.08, '(a)', transform=axes[0, 0].transAxes,
                   fontsize=13, fontweight='bold', va='top')
    axes[0, 1].text(-0.12, 1.08, '(b)', transform=axes[0, 1].transAxes,
                   fontsize=13, fontweight='bold', va='top')

    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight',
                facecolor='white', edgecolor='none')

    print(f"Dual-active separated figure saved to {output_path}")
    return fig


def create_triple_active_figure(df: pd.DataFrame, output_path: Path):
    """
    Create figure for triple-active library.
    Single row with 2 panels.
    """

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    lib = 'TRIPLE'
    lib_df = df[df['library'] == lib]

    # Left: MW vs LogP
    ax = axes[0]
    ax.scatter(lib_df['MW'], lib_df['LogP'],
              c=LIBRARY_COLORS[lib], s=15, alpha=0.5, edgecolors='none')
    ax.set_xlabel('Molecular Weight (Da)')
    ax.set_ylabel('LogP')
    ax.set_title(f'{LIBRARY_LABELS[lib]} (n={len(lib_df)})', fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    ax.text(-0.10, 1.05, '(a)', transform=ax.transAxes,
            fontsize=13, fontweight='bold', va='top')

    # Right: LogP vs TPSA
    ax = axes[1]
    ax.scatter(lib_df['LogP'], lib_df['TPSA'],
              c=LIBRARY_COLORS[lib], s=15, alpha=0.5, edgecolors='none')
    ax.set_xlabel('LogP')
    ax.set_ylabel('TPSA (Å²)')
    ax.set_title(f'{LIBRARY_LABELS[lib]} (n={len(lib_df)})', fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    ax.text(-0.10, 1.05, '(b)', transform=ax.transAxes,
            fontsize=13, fontweight='bold', va='top')

    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight',
                facecolor='white', edgecolor='none')

    print(f"Triple-active figure saved to {output_path}")
    return fig


def create_all_libraries_comparison(base_dir: Path, output_path: Path):
    """
    Create a comprehensive figure comparing all library types:
    - Single pathogens (SA, EC, CA)
    - Dual-active (SA+EC, SA+CA, CA+EC)
    - Triple-active

    Using LogP vs TPSA which shows the clearest separation.
    """

    # Load all libraries
    all_libs = ['SA', 'EC', 'CA', 'SA_EC', 'SA_CA', 'CA_EC', 'TRIPLE']

    all_colors = {
        'SA': '#DC143C',
        'EC': '#1E90FF',
        'CA': '#228B22',
        'SA_EC': '#8B008B',
        'SA_CA': '#FF8C00',
        'CA_EC': '#20B2AA',
        'TRIPLE': '#DAA520',
    }

    all_labels = {
        'SA': 'S. aureus',
        'EC': 'E. coli',
        'CA': 'C. albicans',
        'SA_EC': 'SA + EC',
        'SA_CA': 'SA + CA',
        'CA_EC': 'CA + EC',
        'TRIPLE': 'Triple',
    }

    # Load all data
    all_fragments = []
    libraries_dir = base_dir / 'libraries'

    for lib_name in all_libs:
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
            })

    df = pd.DataFrame(all_fragments)

    # Create 7-row figure (one row per library)
    fig, axes = plt.subplots(7, 2, figsize=(11, 24))

    for row_idx, lib in enumerate(all_libs):
        lib_df = df[df['library'] == lib]

        # Left: MW vs LogP
        ax = axes[row_idx, 0]
        ax.scatter(lib_df['MW'], lib_df['LogP'],
                  c=all_colors[lib], s=15, alpha=0.55, edgecolors='none')
        ax.set_xlabel('Molecular Weight (Da)')
        ax.set_ylabel('LogP')
        ax.set_title(f'{all_labels[lib]} (n={len(lib_df)})', fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)
        ax.set_xlim(0, 650)
        ax.set_ylim(-6, 12)

        # Right: LogP vs TPSA
        ax = axes[row_idx, 1]
        ax.scatter(lib_df['LogP'], lib_df['TPSA'],
                  c=all_colors[lib], s=15, alpha=0.55, edgecolors='none')
        ax.set_xlabel('LogP')
        ax.set_ylabel('TPSA (Å²)')
        ax.set_title(f'{all_labels[lib]} (n={len(lib_df)})', fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)
        ax.set_xlim(-6, 12)
        ax.set_ylim(0, 225)

    # Add panel labels
    axes[0, 0].text(-0.12, 1.08, '(a)', transform=axes[0, 0].transAxes,
                   fontsize=13, fontweight='bold', va='top')
    axes[0, 1].text(-0.12, 1.08, '(b)', transform=axes[0, 1].transAxes,
                   fontsize=13, fontweight='bold', va='top')

    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight',
                facecolor='white', edgecolor='none')

    print(f"All libraries comparison saved to {output_path}")
    return fig


def main():
    """Generate dual and triple active visualizations."""

    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / 'results'
    output_dir.mkdir(exist_ok=True)

    # Load dual-active libraries
    print("="*60)
    print("Loading Dual-Active Libraries")
    print("="*60)
    dual_df = load_libraries(base_dir, DUAL_LIBRARIES)
    print(f"\nTotal dual-active fragments: {len(dual_df)}")

    # Load triple-active library
    print("\n" + "="*60)
    print("Loading Triple-Active Library")
    print("="*60)
    triple_df = load_libraries(base_dir, TRIPLE_LIBRARY)
    print(f"\nTotal triple-active fragments: {len(triple_df)}")

    print("\n" + "="*60)
    print("Generating Figures")
    print("="*60)

    # Create dual-active separated figure
    create_dual_active_separated(dual_df, output_dir / 'dual_active_separated.png')

    # Create triple-active figure
    create_triple_active_figure(triple_df, output_dir / 'triple_active_figure.png')

    # Create comprehensive all-libraries comparison
    print("\nCreating all-libraries comparison...")
    create_all_libraries_comparison(base_dir, output_dir / 'all_libraries_separated.png')

    print("\n" + "="*60)
    print("Complete!")
    print("="*60)


if __name__ == '__main__':
    main()
