"""
ALMA — UMAP + Per-Class Density plots for V1, V2, V3, V4
Each dataset gets: 1 UMAP plot + 1 per-class density plot = 8 plots total
"""
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

BASE = os.path.dirname(os.path.abspath(__file__))
V2_DIR = os.path.join(os.path.dirname(BASE), 'Data_Processing_v2')

T_V1 = ['Aspirational', 'Familial', 'Social', 'Navigational', 'Resistance',
        'Attainment', 'First_Gen', 'Perseverance', 'Filial_Piety',
        'Community_Consciousness', 'Spiritual']
T_V2 = ['Aspirational', 'Familial_Capital', 'Social', 'Navigational', 'Resistance',
        'Attainment', 'Perseverance', 'Community_Consciousness', 'Spiritual']
T_V4 = ['Aspirational', 'Familial_Capital', 'Social', 'Navigational', 'Resistance',
        'Attainment', 'Perseverance', 'Spiritual']

COLORS = {
    'Class_0': '#999999', 'Aspirational': '#e6194b', 'Familial': '#3cb44b',
    'Social': '#4363d8', 'Navigational': '#f58231', 'Resistance': '#911eb4',
    'Attainment': '#42d4f4', 'First_Gen': '#f032e6', 'Perseverance': '#bfef45',
    'Filial_Piety': '#fabed4', 'Community_Consciousness': '#dcbeff', 'Spiritual': '#ffe119',
    'Familial_Capital': '#3cb44b'
}


def assign_primary(df, themes):
    theme_counts = df[themes].sum()
    labels = []
    for _, row in df.iterrows():
        active = [t for t in themes if row[t] == 1]
        if not active:
            labels.append('Class_0')
        elif len(active) == 1:
            labels.append(active[0])
        else:
            labels.append(min(active, key=lambda t: theme_counts[t]))
    return labels


def get_umap_subset(v1_keys, ds_keys, all_umap):
    mask = np.array([k in ds_keys for k in v1_keys])
    return all_umap[mask]


def plot_umap(df, umap_data, themes, title, filename):
    classes = ['Class_0'] + themes
    fig, ax = plt.subplots(figsize=(16, 12))
    for cls in classes:
        mask = df['primary_label'] == cls
        if mask.sum() == 0:
            continue
        alpha = 0.06 if cls == 'Class_0' else 0.4
        size = 3 if cls == 'Class_0' else 6
        ax.scatter(umap_data[mask.values, 0], umap_data[mask.values, 1],
                   c=COLORS.get(cls, '#333'), s=size, alpha=alpha,
                   label=f'{cls} ({mask.sum():,})')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend(fontsize=9, markerscale=4, loc='upper left', framealpha=0.9)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(BASE, filename), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def plot_density(df, umap_data, themes, title, filename):
    classes = ['Class_0'] + themes
    n_classes = len(classes)
    ncols = 4
    nrows = (n_classes + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(24, 6 * nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)
    axes_flat = axes.flatten()

    for idx, cls in enumerate(classes):
        ax = axes_flat[idx]
        ax.scatter(umap_data[:, 0], umap_data[:, 1], c='lightgray', s=1, alpha=0.03)
        if cls == 'Class_0':
            mask = (df[themes].sum(axis=1) == 0).values
        else:
            mask = (df[cls] == 1).values
        n_pts = mask.sum()
        if n_pts > 0:
            ax.scatter(umap_data[mask, 0], umap_data[mask, 1],
                       c=COLORS.get(cls, '#333'), s=8, alpha=0.4)
            if n_pts > 30:
                try:
                    xy = umap_data[mask]
                    kde = gaussian_kde(xy.T, bw_method=0.3)
                    xmin, xmax = umap_data[:, 0].min() - 0.5, umap_data[:, 0].max() + 0.5
                    ymin, ymax = umap_data[:, 1].min() - 0.5, umap_data[:, 1].max() + 0.5
                    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
                    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
                    ax.contour(xx, yy, zz, levels=5, colors=COLORS.get(cls, '#333'),
                               alpha=0.6, linewidths=1)
                except Exception:
                    pass
        pct = n_pts / len(df) * 100
        ax.set_title(f'{cls}\n{n_pts:,} sentences ({pct:.1f}%)',
                     fontsize=11, fontweight='bold', color=COLORS.get(cls, '#333'))
        ax.set_xticks([])
        ax.set_yticks([])

    for idx in range(len(classes), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(title, fontsize=18, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE, filename), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def main():
    print("Loading all datasets...")
    v1 = pd.read_csv(os.path.join(BASE, 'v1_original_processed.csv'))
    v2 = pd.read_csv(os.path.join(BASE, 'v2_merged_cleaned.csv'))
    v3 = pd.read_csv(os.path.join(BASE, 'v3_boundary_cleaned.csv'))
    v4 = pd.read_csv(os.path.join(BASE, 'v4_no_cc.csv'))

    all_umap = np.load(os.path.join(V2_DIR, 'v2_umap_2d.npy'))
    v1_keys = list(zip(v1['essay_id'], v1['sentence'].str[:50]))

    # Compute UMAP subsets
    umap_v1 = all_umap[:len(v1)]
    umap_v2 = get_umap_subset(v1_keys, set(zip(v2['essay_id'], v2['sentence'].str[:50])), all_umap)
    umap_v3 = get_umap_subset(v1_keys, set(zip(v3['essay_id'], v3['sentence'].str[:50])), all_umap)
    umap_v4 = get_umap_subset(v1_keys, set(zip(v4['essay_id'], v4['sentence'].str[:50])), all_umap)

    # Primary labels
    v1['primary_label'] = assign_primary(v1, T_V1)
    v2['primary_label'] = assign_primary(v2, T_V2)
    v3['primary_label'] = assign_primary(v3, T_V2)
    v4['primary_label'] = assign_primary(v4, T_V4)

    print(f"  V1: {len(v1):,} | V2: {len(v2):,} | V3: {len(v3):,} | V4: {len(v4):,}")

    # V1
    print("\n--- V1: Original Processed (11 themes) ---")
    plot_umap(v1, umap_v1, T_V1,
              f'V1 — Original Processed UMAP (n={len(v1):,}, 11 themes)', 'v1_umap.png')
    plot_density(v1, umap_v1, T_V1,
                 f'V1 — Per-Class Density (n={len(v1):,}, 11 themes + Class_0)', 'v1_density.png')

    # V2
    print("\n--- V2: Merged + Cleaned (9 themes) ---")
    plot_umap(v2, umap_v2, T_V2,
              f'V2 — Merged+Cleaned UMAP (n={len(v2):,}, 9 themes)', 'v2_umap.png')
    plot_density(v2, umap_v2, T_V2,
                 f'V2 — Per-Class Density (n={len(v2):,}, 9 themes + Class_0)', 'v2_density.png')

    # V3
    print("\n--- V3: Boundary Cleaned (9 themes) ---")
    plot_umap(v3, umap_v3, T_V2,
              f'V3 — Boundary Cleaned UMAP (n={len(v3):,}, 9 themes)', 'v3_umap.png')
    plot_density(v3, umap_v3, T_V2,
                 f'V3 — Boundary Cleaned Density (n={len(v3):,}, 9 themes + Class_0)', 'v3_density.png')

    # V4
    print("\n--- V4: No CC (8 themes) ---")
    plot_umap(v4, umap_v4, T_V4,
              f'V4 — No CC UMAP (n={len(v4):,}, 8 themes)', 'v4_umap.png')
    plot_density(v4, umap_v4, T_V4,
                 f'V4 — Per-Class Density (n={len(v4):,}, 8 themes + Class_0)', 'v4_density.png')

    print("\nAll 8 plots saved!")


if __name__ == '__main__':
    main()
