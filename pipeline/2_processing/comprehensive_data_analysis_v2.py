"""
Comprehensive Data Analysis V2 — PROCESSED (cleaned) dataset
=============================================================
Same 11 plots as V1 but on ALMA_processed_master_dataset.csv (18,019 sentences).
Plots saved to Data_Processing/plots_2/
Embeddings/UMAP recomputed for cleaned dataset.
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import gaussian_kde

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns

# ============================================================
# Configuration — points to PROCESSED dataset + plots_2
# ============================================================
BASE_DIR = "/Users/khalidkhan/Desktop/Workspace/Final_Theseis_Folders/Data/Data_Processing"
INPUT_FILE = os.path.join(BASE_DIR, "ALMA_processed_master_dataset.csv")
PLOT_DIR = os.path.join(BASE_DIR, "plots_2")
EMBED_CACHE = os.path.join(BASE_DIR, "processed_embeddings.npy")
UMAP_CACHE = os.path.join(BASE_DIR, "processed_umap_2d.npy")

THEMES = [
    'Attainment', 'First_Gen', 'Aspirational', 'Navigational', 'Resistance',
    'Perseverance', 'Filial_Piety', 'Familial', 'Community_Consciousness',
    'Social', 'Spiritual'
]

THEME_COLORS = {
    'Aspirational': '#e41a1c',
    'Navigational': '#377eb8',
    'Perseverance': '#4daf4a',
    'Social': '#984ea3',
    'Resistance': '#ff7f00',
    'Spiritual': '#a65628',
    'Familial': '#f781bf',
    'Attainment': '#999999',
    'Filial_Piety': '#66c2a5',
    'Community_Consciousness': '#fc8d62',
    'First_Gen': '#8da0cb',
    'Class_0': '#cccccc',
    'Multi_Label': '#000000',
}

os.makedirs(PLOT_DIR, exist_ok=True)


# ============================================================
# Embeddings + UMAP
# ============================================================
def load_and_embed(df):
    print("=" * 70)
    print("STEP 1: GENERATING SENTENCE EMBEDDINGS")
    print("=" * 70)

    if os.path.exists(EMBED_CACHE):
        embeddings = np.load(EMBED_CACHE)
        if len(embeddings) == len(df):
            print(f"  Loaded cached embeddings: {embeddings.shape}")
            return embeddings

    from sentence_transformers import SentenceTransformer
    print("  Loading model: all-MiniLM-L6-v2...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    sentences = df['sentence'].fillna('').tolist()
    print(f"  Encoding {len(sentences)} sentences...")
    embeddings = model.encode(sentences, show_progress_bar=True, batch_size=256)
    embeddings = np.array(embeddings)
    np.save(EMBED_CACHE, embeddings)
    print(f"  Saved: {embeddings.shape}")
    return embeddings


def compute_umap(embeddings):
    print("\n" + "=" * 70)
    print("STEP 2: UMAP DIMENSION REDUCTION")
    print("=" * 70)

    if os.path.exists(UMAP_CACHE):
        umap_2d = np.load(UMAP_CACHE)
        if len(umap_2d) == len(embeddings):
            print(f"  Loaded cached UMAP: {umap_2d.shape}")
            return umap_2d

    import umap
    print("  Running UMAP (n_neighbors=30, min_dist=0.3)...")
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, n_components=2,
                        metric='cosine', random_state=42)
    umap_2d = reducer.fit_transform(embeddings)
    np.save(UMAP_CACHE, umap_2d)
    print(f"  UMAP complete: {umap_2d.shape}")
    return umap_2d


def assign_labels(df):
    labels = []
    for _, row in df.iterrows():
        active = [t for t in THEMES if row[t] == 1]
        if len(active) == 0:
            labels.append('Class_0')
        elif len(active) == 1:
            labels.append(active[0])
        else:
            labels.append('Multi_Label')
    return labels


# ============================================================
# PLOT 1: Full UMAP scatter
# ============================================================
def plot_full_umap(df, umap_2d, labels):
    print("\n  Plotting: Full UMAP scatter...")
    fig, ax = plt.subplots(figsize=(20, 16))

    mask0 = np.array([l == 'Class_0' for l in labels])
    ax.scatter(umap_2d[mask0, 0], umap_2d[mask0, 1],
               c=THEME_COLORS['Class_0'], s=3, alpha=0.15, label='Class_0', zorder=1)

    for theme in THEMES:
        mask = np.array([l == theme for l in labels])
        if mask.sum() > 0:
            ax.scatter(umap_2d[mask, 0], umap_2d[mask, 1],
                       c=THEME_COLORS[theme], s=12, alpha=0.6, label=theme, zorder=2)

    mask_multi = np.array([l == 'Multi_Label' for l in labels])
    if mask_multi.sum() > 0:
        ax.scatter(umap_2d[mask_multi, 0], umap_2d[mask_multi, 1],
                   c=THEME_COLORS['Multi_Label'], s=8, alpha=0.4,
                   marker='x', label='Multi_Label', zorder=3)

    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', markerscale=3, fontsize=10)
    ax.set_title('PROCESSED Dataset — UMAP (18,019 sentences, after cleaning)', fontsize=16)
    ax.set_xlabel('UMAP-1')
    ax.set_ylabel('UMAP-2')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '01_full_umap_scatter.png'), dpi=200)
    plt.close()
    print("    Saved: 01_full_umap_scatter.png")


# ============================================================
# PLOT 2: Theme centroids
# ============================================================
def plot_theme_centroids(df, umap_2d, embeddings):
    print("  Plotting: Theme centroids...")
    fig, ax = plt.subplots(figsize=(20, 16))

    centroids_umap = {}
    for theme in THEMES:
        mask = df[theme] == 1
        if mask.sum() > 0:
            centroids_umap[theme] = umap_2d[mask].mean(axis=0)
    mask0 = df[THEMES].sum(axis=1) == 0
    centroids_umap['Class_0'] = umap_2d[mask0].mean(axis=0)

    ax.scatter(umap_2d[:, 0], umap_2d[:, 1], c='#e0e0e0', s=2, alpha=0.1, zorder=1)
    for theme in THEMES:
        mask = df[theme] == 1
        if mask.sum() > 0:
            ax.scatter(umap_2d[mask, 0], umap_2d[mask, 1],
                       c=THEME_COLORS[theme], s=8, alpha=0.3, zorder=2)

    for theme, centroid in centroids_umap.items():
        color = THEME_COLORS.get(theme, '#000000')
        ax.scatter(centroid[0], centroid[1], c=color, s=500, marker='*',
                   edgecolors='black', linewidths=1.5, zorder=10)
        ax.annotate(theme, (centroid[0], centroid[1]),
                    fontsize=9, fontweight='bold',
                    xytext=(8, 8), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3))

    ax.set_title('PROCESSED — Theme Centroids with Data Points', fontsize=16)
    ax.set_xlabel('UMAP-1')
    ax.set_ylabel('UMAP-2')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '02_theme_centroids.png'), dpi=200)
    plt.close()
    print("    Saved: 02_theme_centroids.png")


# ============================================================
# PLOT 3: Per-theme density
# ============================================================
def plot_per_theme_density(df, umap_2d):
    print("  Plotting: Per-theme density overlaps...")
    fig, axes = plt.subplots(3, 4, figsize=(28, 21))
    axes = axes.flatten()

    mask0 = df[THEMES].sum(axis=1) == 0
    x0, y0 = umap_2d[mask0, 0], umap_2d[mask0, 1]

    for idx, theme in enumerate(THEMES):
        ax = axes[idx]
        mask_t = df[theme] == 1
        ax.scatter(x0, y0, c='#cccccc', s=1, alpha=0.08, zorder=1)
        xt, yt = umap_2d[mask_t, 0], umap_2d[mask_t, 1]
        ax.scatter(xt, yt, c=THEME_COLORS[theme], s=6, alpha=0.5, zorder=3)

        if mask_t.sum() > 20:
            try:
                xy_t = np.vstack([xt, yt])
                kde_t = gaussian_kde(xy_t, bw_method=0.3)
                xmin, xmax = umap_2d[:, 0].min() - 1, umap_2d[:, 0].max() + 1
                ymin, ymax = umap_2d[:, 1].min() - 1, umap_2d[:, 1].max() + 1
                xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
                positions = np.vstack([xx.ravel(), yy.ravel()])
                z_t = np.reshape(kde_t(positions), xx.shape)
                ax.contour(xx, yy, z_t, levels=5, colors=[THEME_COLORS[theme]],
                           alpha=0.6, linewidths=1)
                ax.contourf(xx, yy, z_t, levels=5, colors=[THEME_COLORS[theme]],
                            alpha=0.15)
            except Exception:
                pass

        ax.set_title(f'{theme} (n={mask_t.sum()})', fontsize=12, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])

    axes[11].axis('off')
    axes[11].text(0.1, 0.5, "PROCESSED DATASET\nDensity Overlap Analysis\n\n"
                  "Shaded = theme concentration\nGray = Class_0\n\n"
                  "After removing 1,705\nconfused/outlier sentences",
                  fontsize=12, va='center', transform=axes[11].transAxes)

    plt.suptitle('PROCESSED — Per-Theme Density (vs Class_0)', fontsize=18, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '03_per_theme_density.png'), dpi=200)
    plt.close()
    print("    Saved: 03_per_theme_density.png")


# ============================================================
# PLOT 4: Correlation heatmap
# ============================================================
def plot_correlation_analysis(df):
    print("  Plotting: Theme correlation heatmap...")
    corr_matrix = df[THEMES].corr()

    fig, axes = plt.subplots(1, 2, figsize=(24, 10))

    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdYlBu_r',
                vmin=-0.2, vmax=0.6, ax=axes[0], square=True, linewidths=0.5)
    axes[0].set_title('PROCESSED — Pearson Correlation', fontsize=14)

    jaccard = np.zeros((len(THEMES), len(THEMES)))
    for i, t1 in enumerate(THEMES):
        for j, t2 in enumerate(THEMES):
            a = set(df[df[t1] == 1].index)
            b = set(df[df[t2] == 1].index)
            if len(a | b) > 0:
                jaccard[i, j] = len(a & b) / len(a | b)

    jaccard_df = pd.DataFrame(jaccard, index=THEMES, columns=THEMES)
    sns.heatmap(jaccard_df, annot=True, fmt='.3f', cmap='YlOrRd',
                vmin=0, vmax=0.5, ax=axes[1], square=True, linewidths=0.5)
    axes[1].set_title('PROCESSED — Jaccard Similarity', fontsize=14)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '04_correlation_heatmap.png'), dpi=200)
    plt.close()
    print("    Saved: 04_correlation_heatmap.png")
    return corr_matrix


# ============================================================
# PLOT 5: Class imbalance
# ============================================================
def plot_imbalance(df):
    print("  Plotting: Class imbalance...")
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    counts = {t: (df[t] == 1).sum() for t in THEMES}
    counts['Class_0'] = (df[THEMES].sum(axis=1) == 0).sum()
    sorted_themes = sorted(counts.keys(), key=lambda x: -counts[x])
    colors = [THEME_COLORS.get(t, '#999999') for t in sorted_themes]

    bars = axes[0].barh(sorted_themes, [counts[t] for t in sorted_themes], color=colors)
    for bar, t in zip(bars, sorted_themes):
        axes[0].text(bar.get_width() + 50, bar.get_y() + bar.get_height() / 2,
                     f'{counts[t]:,} ({counts[t] / len(df) * 100:.1f}%)',
                     va='center', fontsize=9)
    axes[0].set_title('PROCESSED — Sentence Count per Theme', fontsize=14)
    axes[0].set_xlabel('Count')
    axes[0].invert_yaxis()

    n_labels = df[THEMES].sum(axis=1)
    label_counts = n_labels.value_counts().sort_index()
    axes[1].bar(label_counts.index, label_counts.values, color='steelblue')
    for x, y in zip(label_counts.index, label_counts.values):
        axes[1].text(x, y + 50, f'{y}\n({y / len(df) * 100:.1f}%)', ha='center', fontsize=9)
    axes[1].set_title('Multi-Label Distribution', fontsize=14)
    axes[1].set_xlabel('Number of themes per sentence')
    axes[1].set_ylabel('Count')

    ratios = {}
    for t in THEMES:
        pos = (df[t] == 1).sum()
        neg = (df[t] == 0).sum()
        ratios[t] = neg / max(pos, 1)
    sorted_r = sorted(ratios.keys(), key=lambda x: -ratios[x])
    colors_r = [THEME_COLORS[t] for t in sorted_r]
    axes[2].barh(sorted_r, [ratios[t] for t in sorted_r], color=colors_r)
    for i, t in enumerate(sorted_r):
        axes[2].text(ratios[t] + 1, i, f'{ratios[t]:.0f}:1', va='center', fontsize=9)
    axes[2].set_title('Class Imbalance Ratio (Neg:Pos)', fontsize=14)
    axes[2].set_xlabel('Ratio')
    axes[2].axvline(x=10, color='red', linestyle='--', alpha=0.5, label='10:1 threshold')
    axes[2].legend()
    axes[2].invert_yaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '05_class_imbalance.png'), dpi=200)
    plt.close()
    print("    Saved: 05_class_imbalance.png")


# ============================================================
# PLOT 6: Decision boundaries
# ============================================================
def plot_decision_boundaries(df, umap_2d):
    print("  Plotting: Decision boundaries...")

    primary_labels = []
    for _, row in df.iterrows():
        active = [t for t in THEMES if row[t] == 1]
        if len(active) == 0:
            primary_labels.append('Class_0')
        elif len(active) == 1:
            primary_labels.append(active[0])
        else:
            theme_counts = {t: (df[t] == 1).sum() for t in active}
            primary_labels.append(min(theme_counts, key=theme_counts.get))
    primary_labels = np.array(primary_labels)

    unique_labels = list(set(primary_labels))
    label_to_int = {l: i for i, l in enumerate(unique_labels)}
    y = np.array([label_to_int[l] for l in primary_labels])

    knn = KNeighborsClassifier(n_neighbors=15, weights='distance')
    knn.fit(umap_2d, y)

    x_min, x_max = umap_2d[:, 0].min() - 1, umap_2d[:, 0].max() + 1
    y_min, y_max = umap_2d[:, 1].min() - 1, umap_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    mesh_points = np.c_[xx.ravel(), yy.ravel()]

    probs = knn.predict_proba(mesh_points)
    max_probs = probs.max(axis=1).reshape(xx.shape)
    predictions = knn.predict(mesh_points).reshape(xx.shape)

    fig, axes = plt.subplots(1, 2, figsize=(28, 12))

    cmap_colors = [THEME_COLORS.get(l, '#999999') for l in unique_labels]
    cmap = ListedColormap(cmap_colors)
    axes[0].contourf(xx, yy, predictions, alpha=0.15, cmap=cmap, levels=len(unique_labels))
    mask0 = primary_labels == 'Class_0'
    axes[0].scatter(umap_2d[mask0, 0], umap_2d[mask0, 1], c='#cccccc', s=2, alpha=0.1, zorder=1)
    for theme in THEMES:
        mask = primary_labels == theme
        if mask.sum() > 0:
            axes[0].scatter(umap_2d[mask, 0], umap_2d[mask, 1],
                            c=THEME_COLORS[theme], s=8, alpha=0.4, label=theme, zorder=2)
    axes[0].set_title('PROCESSED — KNN Decision Regions (k=15)', fontsize=14)
    axes[0].legend(bbox_to_anchor=(1.02, 1), loc='upper left', markerscale=3, fontsize=8)

    confusion = axes[1].contourf(xx, yy, max_probs, levels=[0, 0.3, 0.5, 0.7, 1.0],
                                  colors=['#ff0000', '#ff9900', '#ffff00', '#00ff00'], alpha=0.25)
    axes[1].scatter(umap_2d[:, 0], umap_2d[:, 1], c='#333333', s=1, alpha=0.1)
    axes[1].contour(xx, yy, max_probs, levels=[0.3, 0.5],
                    colors=['red', 'orange'], linewidths=2, linestyles='--')
    cbar = plt.colorbar(confusion, ax=axes[1])
    cbar.set_label('KNN Confidence')
    axes[1].set_title('PROCESSED — Overlap/Confusion Zones\n(Red=confusion, Green=clear)', fontsize=14)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '06_decision_boundaries.png'), dpi=200)
    plt.close()
    print("    Saved: 06_decision_boundaries.png")


# ============================================================
# PLOT 7: Pairwise overlaps
# ============================================================
def plot_pairwise_overlaps(df, umap_2d):
    print("  Plotting: Pairwise theme overlaps...")
    pairs = []
    for i, t1 in enumerate(THEMES):
        for j, t2 in enumerate(THEMES):
            if i < j:
                mask1, mask2 = df[t1] == 1, df[t2] == 1
                both = (mask1 & mask2).sum()
                if both > 0:
                    pairs.append((t1, t2, both / (mask1 | mask2).sum(), both))
    pairs.sort(key=lambda x: -x[2])
    top_pairs = pairs[:6]

    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    axes = axes.flatten()

    for idx, (t1, t2, jacc, both) in enumerate(top_pairs):
        ax = axes[idx]
        mask1, mask2 = df[t1] == 1, df[t2] == 1
        both_mask = mask1 & mask2
        only1 = mask1 & ~mask2
        only2 = mask2 & ~mask1
        neither = ~mask1 & ~mask2

        ax.scatter(umap_2d[neither, 0], umap_2d[neither, 1], c='#eeeeee', s=1, alpha=0.05)
        ax.scatter(umap_2d[only1, 0], umap_2d[only1, 1],
                   c=THEME_COLORS[t1], s=10, alpha=0.5, label=f'{t1} only ({only1.sum()})')
        ax.scatter(umap_2d[only2, 0], umap_2d[only2, 1],
                   c=THEME_COLORS[t2], s=10, alpha=0.5, label=f'{t2} only ({only2.sum()})')
        ax.scatter(umap_2d[both_mask, 0], umap_2d[both_mask, 1],
                   c='black', s=15, alpha=0.7, marker='D', label=f'BOTH ({both})')
        ax.set_title(f'{t1} vs {t2}\nJaccard={jacc:.3f}, Co-occur={both}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='best')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle('PROCESSED — Top 6 Most Overlapping Theme Pairs', fontsize=16, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '07_pairwise_overlaps.png'), dpi=200)
    plt.close()
    print("    Saved: 07_pairwise_overlaps.png")


# ============================================================
# PLOT 8: Similarity distributions
# ============================================================
def plot_similarity_distributions(df, embeddings):
    print("  Plotting: Similarity distributions...")

    centroids = {}
    for theme in THEMES:
        mask = df[theme] == 1
        if mask.sum() > 0:
            centroids[theme] = embeddings[mask].mean(axis=0)
    mask0 = df[THEMES].sum(axis=1) == 0
    centroids['Class_0'] = embeddings[mask0].mean(axis=0)

    fig, axes = plt.subplots(3, 4, figsize=(28, 21))
    axes = axes.flatten()

    for idx, theme in enumerate(THEMES):
        ax = axes[idx]
        mask_t = df[theme] == 1
        if mask_t.sum() < 5:
            ax.set_title(f'{theme} (too few)')
            continue

        centroid = centroids[theme].reshape(1, -1)
        theme_sims = cosine_similarity(embeddings[mask_t], centroid).flatten()
        class0_sims = cosine_similarity(embeddings[mask0], centroid).flatten()

        ax.hist(theme_sims, bins=50, alpha=0.7, color=THEME_COLORS[theme],
                label=f'{theme} (n={mask_t.sum()})', density=True)
        ax.hist(class0_sims, bins=50, alpha=0.4, color='gray',
                label=f'Class_0 (n={mask0.sum()})', density=True)

        overlap_low = max(theme_sims.min(), np.percentile(class0_sims, 75))
        overlap_high = min(theme_sims.max(), np.percentile(theme_sims, 25))
        if overlap_low < overlap_high:
            ax.axvspan(overlap_low, overlap_high, color='red', alpha=0.15, label='Overlap zone')

        ax.set_title(f'{theme}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=7)
        ax.set_xlabel('Cosine similarity to centroid')

    axes[11].axis('off')
    axes[11].text(0.1, 0.5,
                  "PROCESSED DATASET\nSimilarity Distributions\n\n"
                  "Colored = theme sentences\nGray = Class_0\n"
                  "Red = overlap zone\n\n"
                  "Compare with V1 to see\nimproved separation",
                  fontsize=12, va='center', transform=axes[11].transAxes)

    plt.suptitle('PROCESSED — Cosine Similarity to Theme Centroids', fontsize=16, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '08_similarity_distributions.png'), dpi=200)
    plt.close()
    print("    Saved: 08_similarity_distributions.png")


# ============================================================
# PLOT 9: K-Means clusters vs actual
# ============================================================
def plot_kmeans_clusters(df, umap_2d, embeddings):
    print("  Plotting: K-Means clusters vs actual labels...")

    kmeans = KMeans(n_clusters=12, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)

    fig, axes = plt.subplots(1, 2, figsize=(24, 10))

    scatter1 = axes[0].scatter(umap_2d[:, 0], umap_2d[:, 1],
                                c=cluster_labels, cmap='Set3', s=3, alpha=0.3)
    axes[0].set_title('PROCESSED — K-Means Clusters (k=12)', fontsize=14)
    plt.colorbar(scatter1, ax=axes[0], label='Cluster')

    primary = []
    for _, row in df.iterrows():
        active = [t for t in THEMES if row[t] == 1]
        if len(active) == 0:
            primary.append(0)
        else:
            primary.append(THEMES.index(active[0]) + 1)

    scatter2 = axes[1].scatter(umap_2d[:, 0], umap_2d[:, 1],
                                c=primary, cmap='Set3', s=3, alpha=0.3)
    axes[1].set_title('PROCESSED — Actual Theme Labels', fontsize=14)
    plt.colorbar(scatter2, ax=axes[1], label='Theme')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '09_kmeans_vs_actual.png'), dpi=200)
    plt.close()
    print("    Saved: 09_kmeans_vs_actual.png")

    print("\n  CLUSTER PURITY ANALYSIS:")
    for c in range(12):
        mask = cluster_labels == c
        n = mask.sum()
        theme_counts = {}
        for t in THEMES:
            tc = (df[t][mask] == 1).sum()
            if tc > 0:
                theme_counts[t] = tc
        class0 = (df[THEMES][mask].sum(axis=1) == 0).sum()
        if class0 > 0:
            theme_counts['Class_0'] = class0
        dominant = max(theme_counts, key=theme_counts.get) if theme_counts else 'empty'
        purity = theme_counts.get(dominant, 0) / n if n > 0 else 0
        top3 = dict(sorted(theme_counts.items(), key=lambda x: -x[1])[:3])
        print(f"    Cluster {c}: n={n}, dominant={dominant} ({purity:.1%}), themes={top3}")


# ============================================================
# PLOT 10: Theme vs Class_0 overlap regions
# ============================================================
def plot_theme_vs_class0_overlap(df, umap_2d):
    print("  Plotting: Theme vs Class_0 overlap regions...")

    mask0 = df[THEMES].sum(axis=1) == 0
    theme_sizes = {t: (df[t] == 1).sum() for t in THEMES}
    top_themes = sorted(theme_sizes, key=lambda x: -theme_sizes[x])[:6]

    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    axes = axes.flatten()

    for idx, theme in enumerate(top_themes):
        ax = axes[idx]
        mask_t = df[theme] == 1
        xt, yt = umap_2d[mask_t, 0], umap_2d[mask_t, 1]
        x0, y0 = umap_2d[mask0, 0], umap_2d[mask0, 1]

        ax.scatter(umap_2d[:, 0], umap_2d[:, 1], c='#f5f5f5', s=1, alpha=0.05)

        try:
            xy_t = np.vstack([xt, yt])
            kde_t = gaussian_kde(xy_t, bw_method=0.3)
            xmin, xmax = umap_2d[:, 0].min() - 1, umap_2d[:, 0].max() + 1
            ymin, ymax = umap_2d[:, 1].min() - 1, umap_2d[:, 1].max() + 1
            xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
            positions = np.vstack([xx.ravel(), yy.ravel()])
            z_t = np.reshape(kde_t(positions), xx.shape)

            c0_sample = np.random.choice(mask0.sum(), min(3000, mask0.sum()), replace=False)
            xy_0 = np.vstack([x0.values[c0_sample], y0.values[c0_sample]])
            kde_0 = gaussian_kde(xy_0, bw_method=0.3)
            z_0 = np.reshape(kde_0(positions), xx.shape)

            z_t_norm = z_t / (z_t.max() + 1e-10)
            z_0_norm = z_0 / (z_0.max() + 1e-10)
            overlap = np.minimum(z_t_norm, z_0_norm)

            ax.contourf(xx, yy, z_t_norm, levels=5, colors=[THEME_COLORS[theme]], alpha=0.2)
            ax.contour(xx, yy, z_t_norm, levels=3, colors=[THEME_COLORS[theme]], alpha=0.5, linewidths=1)
            ax.contourf(xx, yy, overlap, levels=[0.2, 1.0], colors=['red'], alpha=0.3)
            ax.contour(xx, yy, overlap, levels=[0.2], colors=['red'], linewidths=2, linestyles='--')
        except Exception:
            pass

        ax.scatter(x0, y0, c='gray', s=1, alpha=0.05)
        ax.scatter(xt, yt, c=THEME_COLORS[theme], s=6, alpha=0.4)
        ax.set_title(f'{theme} (n={mask_t.sum()}) — Red = Overlap', fontsize=11, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle('PROCESSED — Theme vs Class_0 Overlap (Red = confusion zones)', fontsize=16, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '10_theme_class0_overlap.png'), dpi=200)
    plt.close()
    print("    Saved: 10_theme_class0_overlap.png")


# ============================================================
# PLOT 11: Before vs After comparison
# ============================================================
def plot_before_after_comparison(df):
    """Side-by-side bar chart comparing original vs processed distributions."""
    print("  Plotting: Before vs After comparison...")

    # Original counts
    orig = {'Attainment': 456, 'First_Gen': 33, 'Aspirational': 2792,
            'Navigational': 4449, 'Resistance': 840, 'Perseverance': 1542,
            'Filial_Piety': 241, 'Familial': 646, 'Community_Consciousness': 119,
            'Social': 998, 'Spiritual': 718, 'Class_0': 10337}
    proc = {t: (df[t] == 1).sum() for t in THEMES}
    proc['Class_0'] = (df[THEMES].sum(axis=1) == 0).sum()

    sorted_keys = sorted(orig.keys(), key=lambda x: -orig[x])

    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(sorted_keys))
    w = 0.35
    bars1 = ax.bar(x - w / 2, [orig[k] for k in sorted_keys], w, label='Before (19,724)',
                   color='#aaaaaa', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + w / 2, [proc[k] for k in sorted_keys], w, label='After (18,019)',
                   color=[THEME_COLORS.get(k, '#cccccc') for k in sorted_keys],
                   edgecolor='black', linewidth=0.5)

    for bar, k in zip(bars1, sorted_keys):
        diff = proc[k] - orig[k]
        if diff != 0:
            ax.text(bar.get_x() + w, bar.get_height() + 50,
                    f'{diff:+d}', ha='center', fontsize=8, color='red')

    ax.set_xticks(x)
    ax.set_xticklabels(sorted_keys, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Sentence Count')
    ax.set_title('Before vs After Cleaning — Distribution Comparison', fontsize=16)
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '11_before_after_comparison.png'), dpi=200)
    plt.close()
    print("    Saved: 11_before_after_comparison.png")


# ============================================================
# MAIN
# ============================================================
def main():
    print("Loading PROCESSED dataset...")
    df = pd.read_csv(INPUT_FILE)
    print(f"  {len(df)} sentences loaded")

    embeddings = load_and_embed(df)
    umap_2d = compute_umap(embeddings)
    labels = assign_labels(df)

    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS (PROCESSED DATASET)")
    print("=" * 70)

    plot_full_umap(df, umap_2d, labels)
    plot_theme_centroids(df, umap_2d, embeddings)
    plot_per_theme_density(df, umap_2d)
    corr = plot_correlation_analysis(df)
    plot_imbalance(df)
    plot_decision_boundaries(df, umap_2d)
    plot_pairwise_overlaps(df, umap_2d)
    plot_similarity_distributions(df, embeddings)
    plot_kmeans_clusters(df, umap_2d, embeddings)
    plot_theme_vs_class0_overlap(df, umap_2d)
    plot_before_after_comparison(df)

    print(f"\n{'=' * 70}")
    print(f"ALL PLOTS SAVED TO: {PLOT_DIR}/")
    print(f"{'=' * 70}")
    for f in sorted(os.listdir(PLOT_DIR)):
        print(f"  {f}")


if __name__ == '__main__':
    main()
