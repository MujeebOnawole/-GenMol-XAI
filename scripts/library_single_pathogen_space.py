"""
Single Pathogen Chemical Space Visualization
Focus only on SA, EC, CA libraries for clearer distinction.

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

# Only single pathogen libraries
LIBRARY_ORDER = ['SA', 'EC', 'CA']

LIBRARY_COLORS = {
    'SA': '#DC143C',      # Crimson (Gram-positive)
    'EC': '#1E90FF',      # Dodger Blue (Gram-negative)
    'CA': '#228B22',      # Forest Green (Fungal)
}

LIBRARY_LABELS = {
    'SA': 'S. aureus',
    'EC': 'E. coli',
    'CA': 'C. albicans',
}


def load_single_pathogen_libraries(base_dir: Path) -> pd.DataFrame:
    """Load only SA, EC, CA libraries."""
    all_fragments = []
    libraries_dir = base_dir / 'libraries'

    for lib_name in LIBRARY_ORDER:
        lib_dir = libraries_dir / f'{lib_name}_library'
        json_file = lib_dir / f'safe_library_{lib_name}.json'

        if not json_file.exists():
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


def create_clear_distinction_figure(df: pd.DataFrame, output_path: Path):
    """
    Create 4-panel figure with only SA, EC, CA for maximum clarity.
    Larger dots, better transparency.
    """

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    axes = axes.flatten()

    # Property pairs
    property_pairs = [
        ('MW', 'LogP', 'Molecular Weight (Da)', 'LogP'),
        ('LogP', 'TPSA', 'LogP', 'TPSA (Å²)'),
        ('LogP', 'HBD', 'LogP', 'H-Bond Donors'),
        ('MW', 'TPSA', 'Molecular Weight (Da)', 'TPSA (Å²)'),
    ]

    # Plot order: SA first (most), then CA, then EC (smallest on top)
    plot_order = ['SA', 'CA', 'EC']

    for idx, (x_prop, y_prop, xlabel, ylabel) in enumerate(property_pairs):
        ax = axes[idx]

        for lib in plot_order:
            lib_df = df[df['library'] == lib]
            ax.scatter(
                lib_df[x_prop],
                lib_df[y_prop],
                c=LIBRARY_COLORS[lib],
                s=25,           # Larger dots
                alpha=0.55,     # Better transparency
                edgecolors='none',
                label=LIBRARY_LABELS[lib],
            )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.text(-0.10, 1.05, f'({chr(97+idx)})', transform=ax.transAxes,
                fontsize=13, fontweight='bold', va='top')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)

    plt.tight_layout()

    # Legend
    legend_handles = [Patch(facecolor=LIBRARY_COLORS[lib], alpha=0.7,
                           edgecolor='none', label=LIBRARY_LABELS[lib])
                     for lib in LIBRARY_ORDER]

    fig.legend(handles=legend_handles, loc='lower center', ncol=3,
              bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=11)
    plt.subplots_adjust(bottom=0.08)

    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight',
                facecolor='white', edgecolor='none')

    print(f"Figure saved to {output_path}")
    return fig


def create_2panel_key_figure(df: pd.DataFrame, output_path: Path):
    """
    Create focused 2-panel figure with the two most informative plots.
    """

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    plot_order = ['SA', 'CA', 'EC']

    # Panel (a): LogP vs MW
    ax = axes[0]
    for lib in plot_order:
        lib_df = df[df['library'] == lib]
        ax.scatter(lib_df['MW'], lib_df['LogP'],
                  c=LIBRARY_COLORS[lib], s=30, alpha=0.55,
                  edgecolors='none', label=LIBRARY_LABELS[lib])

    ax.set_xlabel('Molecular Weight (Da)')
    ax.set_ylabel('LogP')
    ax.text(-0.08, 1.05, '(a)', transform=ax.transAxes,
            fontsize=13, fontweight='bold', va='top')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc='upper left', framealpha=0.9, fontsize=10)

    # Panel (b): TPSA vs LogP
    ax = axes[1]
    for lib in plot_order:
        lib_df = df[df['library'] == lib]
        ax.scatter(lib_df['LogP'], lib_df['TPSA'],
                  c=LIBRARY_COLORS[lib], s=30, alpha=0.55,
                  edgecolors='none', label=LIBRARY_LABELS[lib])

    ax.set_xlabel('LogP')
    ax.set_ylabel('TPSA (Å²)')
    ax.text(-0.08, 1.05, '(b)', transform=ax.transAxes,
            fontsize=13, fontweight='bold', va='top')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc='upper right', framealpha=0.9, fontsize=10)

    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight',
                facecolor='white', edgecolor='none')

    print(f"2-panel figure saved to {output_path}")
    return fig


def create_separated_panels(df: pd.DataFrame, output_path: Path):
    """
    Create figure where each pathogen is shown separately in rows
    for maximum clarity - no overlap issues.
    """

    fig, axes = plt.subplots(3, 2, figsize=(11, 12))

    for row_idx, lib in enumerate(LIBRARY_ORDER):
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
        # Set consistent axis limits
        ax.set_xlim(0, 600)
        ax.set_ylim(-3, 10)

        # Right column: LogP vs TPSA
        ax = axes[row_idx, 1]
        ax.scatter(lib_df['LogP'], lib_df['TPSA'],
                  c=LIBRARY_COLORS[lib], s=20, alpha=0.6, edgecolors='none')
        ax.set_xlabel('LogP')
        ax.set_ylabel('TPSA (Å²)')
        ax.set_title(f'{LIBRARY_LABELS[lib]} (n={len(lib_df)})', fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)
        # Set consistent axis limits
        ax.set_xlim(-3, 10)
        ax.set_ylim(0, 175)

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

    print(f"Separated panels figure saved to {output_path}")
    return fig


def create_density_overlay_figure(df: pd.DataFrame, output_path: Path):
    """
    Create figure with density contours to show where each pathogen clusters.
    """
    from scipy.stats import gaussian_kde

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Panel (a): MW vs LogP with density contours
    ax = axes[0]

    for lib in LIBRARY_ORDER:
        lib_df = df[df['library'] == lib].dropna(subset=['MW', 'LogP'])

        # Plot scatter
        ax.scatter(lib_df['MW'], lib_df['LogP'],
                  c=LIBRARY_COLORS[lib], s=15, alpha=0.3,
                  edgecolors='none', label=LIBRARY_LABELS[lib])

        # Add density contour
        try:
            x = lib_df['MW'].values
            y = lib_df['LogP'].values
            xy = np.vstack([x, y])
            kde = gaussian_kde(xy)

            # Create grid
            xmin, xmax = x.min() - 20, x.max() + 20
            ymin, ymax = y.min() - 0.5, y.max() + 0.5
            xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
            positions = np.vstack([xx.ravel(), yy.ravel()])
            z = kde(positions).reshape(xx.shape)

            # Plot contour at 50% density level
            ax.contour(xx, yy, z, levels=[z.max() * 0.2], colors=[LIBRARY_COLORS[lib]],
                      linewidths=2, alpha=0.9)
        except:
            pass

    ax.set_xlabel('Molecular Weight (Da)')
    ax.set_ylabel('LogP')
    ax.text(-0.08, 1.05, '(a)', transform=ax.transAxes,
            fontsize=13, fontweight='bold', va='top')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc='upper left', framealpha=0.9, fontsize=10)

    # Panel (b): LogP vs TPSA with density contours
    ax = axes[1]

    for lib in LIBRARY_ORDER:
        lib_df = df[df['library'] == lib].dropna(subset=['LogP', 'TPSA'])

        # Plot scatter
        ax.scatter(lib_df['LogP'], lib_df['TPSA'],
                  c=LIBRARY_COLORS[lib], s=15, alpha=0.3,
                  edgecolors='none', label=LIBRARY_LABELS[lib])

        # Add density contour
        try:
            x = lib_df['LogP'].values
            y = lib_df['TPSA'].values
            xy = np.vstack([x, y])
            kde = gaussian_kde(xy)

            xmin, xmax = x.min() - 0.5, x.max() + 0.5
            ymin, ymax = max(0, y.min() - 5), y.max() + 5
            xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
            positions = np.vstack([xx.ravel(), yy.ravel()])
            z = kde(positions).reshape(xx.shape)

            ax.contour(xx, yy, z, levels=[z.max() * 0.2], colors=[LIBRARY_COLORS[lib]],
                      linewidths=2, alpha=0.9)
        except:
            pass

    ax.set_xlabel('LogP')
    ax.set_ylabel('TPSA (Å²)')
    ax.text(-0.08, 1.05, '(b)', transform=ax.transAxes,
            fontsize=13, fontweight='bold', va='top')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc='upper right', framealpha=0.9, fontsize=10)

    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight',
                facecolor='white', edgecolor='none')

    print(f"Density overlay figure saved to {output_path}")
    return fig


def main():
    """Generate single pathogen chemical space visualizations."""

    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / 'results'
    output_dir.mkdir(exist_ok=True)

    print("Loading single pathogen libraries (SA, EC, CA only)...")
    df = load_single_pathogen_libraries(base_dir)
    print(f"\nTotal fragments: {len(df)}")
    print(f"  SA: {len(df[df['library']=='SA'])}")
    print(f"  EC: {len(df[df['library']=='EC'])}")
    print(f"  CA: {len(df[df['library']=='CA'])}")

    print("\nGenerating visualizations...")

    # Option 1: 4-panel with larger dots
    create_clear_distinction_figure(df, output_dir / 'single_pathogen_4panel.png')

    # Option 2: 2-panel focused
    create_2panel_key_figure(df, output_dir / 'single_pathogen_2panel.png')

    # Option 3: Separated rows (no overlap)
    create_separated_panels(df, output_dir / 'single_pathogen_separated.png')

    # Option 4: With density contours
    create_density_overlay_figure(df, output_dir / 'single_pathogen_density.png')

    print("\n" + "="*60)
    print("Single pathogen visualizations complete!")
    print("="*60)


if __name__ == '__main__':
    main()
