"""
Library Diversity Visualization - Scatter/Strip Plot Version
More intuitive visualization where each dot = one fragment.

Author: Generated for GenMol XAI Analysis
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

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
    'SA': '#DC143C',      # Crimson
    'EC': '#1E90FF',      # Dodger Blue
    'CA': '#228B22',      # Forest Green
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
            print(f"Warning: {json_file} not found, skipping...")
            continue

        with open(json_file, 'r') as f:
            fragments = json.load(f)

        for frag in fragments:
            props = frag.get('props', {})
            all_fragments.append({
                'library': lib_name,
                'library_label': LIBRARY_LABELS[lib_name],
                'MW': props.get('MW'),
                'LogP': props.get('LogP'),
                'TPSA': props.get('TPSA'),
                'HBD': props.get('HBD'),
                'HBA': props.get('HBA'),
            })

    return pd.DataFrame(all_fragments)


def create_strip_plot(df: pd.DataFrame, output_path: Path):
    """
    Create a 4-panel strip plot where each dot = one fragment.
    Uses jitter and transparency for readability.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    properties = [
        ('LogP', 'LogP', None),
        ('TPSA', 'TPSA', 'Å²'),
        ('HBD', 'H-Bond Donors', None),
        ('MW', 'Molecular Weight', 'Da'),
    ]

    palette = {lib: LIBRARY_COLORS[lib] for lib in LIBRARY_ORDER}

    for idx, (prop, label, unit) in enumerate(properties):
        ax = axes[idx]

        # Create strip plot with jitter
        sns.stripplot(
            data=df,
            x='library',
            y=prop,
            order=LIBRARY_ORDER,
            palette=palette,
            ax=ax,
            size=2.5,           # Small dots
            alpha=0.4,          # Transparency to show density
            jitter=0.35,        # Horizontal spread
            zorder=1,
        )

        # Add median markers (horizontal lines)
        for i, lib in enumerate(LIBRARY_ORDER):
            lib_data = df[df['library'] == lib][prop].dropna()
            median_val = lib_data.median()
            ax.hlines(median_val, i - 0.3, i + 0.3, colors='black',
                     linewidth=2, zorder=2)

        # Customize
        ylabel = f'{label}' if unit is None else f'{label} ({unit})'
        ax.set_ylabel(ylabel)
        ax.set_xlabel('')
        ax.set_xticklabels([LIBRARY_LABELS[lib] for lib in LIBRARY_ORDER],
                          rotation=45, ha='right')
        ax.text(-0.10, 1.05, f'({chr(97+idx)})', transform=ax.transAxes,
                fontsize=12, fontweight='bold', va='top')
        ax.yaxis.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)

    plt.tight_layout()

    # Legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    legend_handles = [Patch(facecolor=LIBRARY_COLORS[lib], alpha=0.7,
                           edgecolor='black', linewidth=0.5,
                           label=LIBRARY_LABELS[lib])
                     for lib in LIBRARY_ORDER]
    legend_handles.append(Line2D([0], [0], color='black', linewidth=2,
                                  label='Median'))

    fig.legend(handles=legend_handles, loc='lower center', ncol=8,
              bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=9)
    plt.subplots_adjust(bottom=0.12)

    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight',
                facecolor='white', edgecolor='none')

    print(f"Strip plot saved to {output_path}")
    return fig


def create_combined_plot(df: pd.DataFrame, output_path: Path):
    """
    Create strip plot with box plot overlay - best of both worlds.
    Dots show raw data, box shows summary statistics.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    properties = [
        ('LogP', 'LogP', None),
        ('TPSA', 'TPSA', 'Å²'),
        ('HBD', 'H-Bond Donors', None),
        ('MW', 'Molecular Weight', 'Da'),
    ]

    palette = {lib: LIBRARY_COLORS[lib] for lib in LIBRARY_ORDER}

    for idx, (prop, label, unit) in enumerate(properties):
        ax = axes[idx]

        # First layer: strip plot (dots)
        sns.stripplot(
            data=df,
            x='library',
            y=prop,
            order=LIBRARY_ORDER,
            palette=palette,
            ax=ax,
            size=2,
            alpha=0.3,
            jitter=0.3,
            zorder=1,
        )

        # Second layer: box plot (summary)
        sns.boxplot(
            data=df,
            x='library',
            y=prop,
            order=LIBRARY_ORDER,
            ax=ax,
            width=0.5,
            showcaps=True,
            boxprops={'facecolor': 'none', 'edgecolor': 'black', 'linewidth': 1.5},
            whiskerprops={'color': 'black', 'linewidth': 1.2},
            capprops={'color': 'black', 'linewidth': 1.2},
            medianprops={'color': 'red', 'linewidth': 2},
            fliersize=0,  # Hide outlier dots (already shown in strip)
            zorder=2,
        )

        ylabel = f'{label}' if unit is None else f'{label} ({unit})'
        ax.set_ylabel(ylabel)
        ax.set_xlabel('')
        ax.set_xticklabels([LIBRARY_LABELS[lib] for lib in LIBRARY_ORDER],
                          rotation=45, ha='right')
        ax.text(-0.10, 1.05, f'({chr(97+idx)})', transform=ax.transAxes,
                fontsize=12, fontweight='bold', va='top')
        ax.yaxis.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)

    plt.tight_layout()

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    legend_handles = [Patch(facecolor=LIBRARY_COLORS[lib], alpha=0.6,
                           edgecolor='black', linewidth=0.5,
                           label=LIBRARY_LABELS[lib])
                     for lib in LIBRARY_ORDER]
    legend_handles.append(Line2D([0], [0], color='red', linewidth=2,
                                  label='Median'))

    fig.legend(handles=legend_handles, loc='lower center', ncol=8,
              bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=9)
    plt.subplots_adjust(bottom=0.12)

    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight',
                facecolor='white', edgecolor='none')

    print(f"Combined plot saved to {output_path}")
    return fig


def create_sina_style_plot(df: pd.DataFrame, output_path: Path):
    """
    Sina-style plot: dots spread by density (like violin shape but with dots).
    Very intuitive - width shows where most fragments cluster.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    properties = [
        ('LogP', 'LogP', None),
        ('TPSA', 'TPSA', 'Å²'),
        ('HBD', 'H-Bond Donors', None),
        ('MW', 'Molecular Weight', 'Da'),
    ]

    for idx, (prop, label, unit) in enumerate(properties):
        ax = axes[idx]

        # Plot each library
        for i, lib in enumerate(LIBRARY_ORDER):
            lib_data = df[df['library'] == lib][prop].dropna().values

            if len(lib_data) == 0:
                continue

            # Create density-based jitter (sina style)
            # More spread where density is higher
            from scipy import stats as scipy_stats

            try:
                kde = scipy_stats.gaussian_kde(lib_data)
                density = kde(lib_data)
                density_normalized = density / density.max() * 0.35  # Max jitter width
            except:
                density_normalized = np.full_like(lib_data, 0.2)

            # Random jitter scaled by density
            jitter = np.random.uniform(-1, 1, len(lib_data)) * density_normalized

            ax.scatter(
                i + jitter,
                lib_data,
                c=LIBRARY_COLORS[lib],
                s=8,
                alpha=0.5,
                edgecolors='none',
            )

            # Add median line
            median_val = np.median(lib_data)
            ax.hlines(median_val, i - 0.3, i + 0.3, colors='black',
                     linewidth=2.5, zorder=3)

        ylabel = f'{label}' if unit is None else f'{label} ({unit})'
        ax.set_ylabel(ylabel)
        ax.set_xlabel('')
        ax.set_xticks(range(len(LIBRARY_ORDER)))
        ax.set_xticklabels([LIBRARY_LABELS[lib] for lib in LIBRARY_ORDER],
                          rotation=45, ha='right')
        ax.text(-0.10, 1.05, f'({chr(97+idx)})', transform=ax.transAxes,
                fontsize=12, fontweight='bold', va='top')
        ax.yaxis.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)

    plt.tight_layout()

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    legend_handles = [Patch(facecolor=LIBRARY_COLORS[lib], alpha=0.7,
                           edgecolor='none',
                           label=LIBRARY_LABELS[lib])
                     for lib in LIBRARY_ORDER]
    legend_handles.append(Line2D([0], [0], color='black', linewidth=2.5,
                                  label='Median'))

    fig.legend(handles=legend_handles, loc='lower center', ncol=8,
              bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=9)
    plt.subplots_adjust(bottom=0.12)

    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight',
                facecolor='white', edgecolor='none')

    print(f"Sina-style plot saved to {output_path}")
    return fig


def main():
    """Generate all scatter-based visualization options."""

    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / 'results'
    output_dir.mkdir(exist_ok=True)

    print("Loading fragment libraries...")
    df = load_all_libraries(base_dir)
    print(f"Total fragments: {len(df)}")

    print("\nGenerating scatter-based visualizations...")

    # Option 1: Simple strip plot
    create_strip_plot(df, output_dir / 'library_diversity_strip.png')

    # Option 2: Strip + box overlay
    create_combined_plot(df, output_dir / 'library_diversity_combined.png')

    # Option 3: Sina-style (density-aware scatter)
    create_sina_style_plot(df, output_dir / 'library_diversity_sina.png')

    print("\n" + "="*60)
    print("All visualizations generated!")
    print("="*60)
    print("\nCompare the three styles:")
    print("  1. library_diversity_strip.png - Simple scatter + median")
    print("  2. library_diversity_combined.png - Scatter + box plot overlay")
    print("  3. library_diversity_sina.png - Density-aware scatter (recommended)")


if __name__ == '__main__':
    main()
