"""
Comprehensive Data Analysis for ALMA Sentence-Level Dataset
============================================================
1. Sentence embeddings (sentence-transformers) + UMAP visualization
2. Theme centroids as centers with colored data points
3. Correlation, overlap, imbalance analysis
4. Decision boundary analysis with shaded overlap regions
5. Suspicious data point identification by ID
6. Similarity clusters and removal recommendations

All plots saved to Data_Processing/plots/
Flagged data points saved to Data_Processing/flagged_datapoints.csv
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import seaborn as sns

# ============================================================
# Configuration
# ============================================================
BASE_DIR = "/Users/khalidkhan/Desktop/Workspace/Final_Theseis_Folders/Data/Data_Processing"
INPUT_FILE = os.path.join(BASE_DIR, "ALMA_sentence_level_dataset.csv")
PLOT_DIR = os.path.join(BASE_DIR, "plots")
EMBED_CACHE = os.path.join(BASE_DIR, "sentence_embeddings.npy")

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
# 1. Load data and generate embeddings
# ============================================================
def load_and_embed(df):
    """Generate sentence embeddings using sentence-transformers."""
    print("=" * 70)
    print("STEP 1: GENERATING SENTENCE EMBEDDINGS")
    print("=" * 70)

    if os.path.exists(EMBED_CACHE):
        print(f"  Loading cached embeddings from {EMBED_CACHE}")
        embeddings = np.load(EMBED_CACHE)
        if len(embeddings) == len(df):
            print(f"  Loaded: {embeddings.shape}")
            return embeddings
        print("  Cache size mismatch, regenerating...")

    from sentence_transformers import SentenceTransformer
    print("  Loading model: all-MiniLM-L6-v2...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    sentences = df['sentence'].fillna('').tolist()
    print(f"  Encoding {len(sentences)} sentences...")
    embeddings = model.encode(sentences, show_progress_bar=True, batch_size=256)
    embeddings = np.array(embeddings)

    np.save(EMBED_CACHE, embeddings)
    print(f"  Saved embeddings: {embeddings.shape} to cache")
    return embeddings


# ============================================================
# 2. UMAP dimension reduction
# ============================================================
def compute_umap(embeddings):
    """Reduce embeddings to 2D using UMAP."""
    print("\n" + "=" * 70)
    print("STEP 2: UMAP DIMENSION REDUCTION")
    print("=" * 70)

    umap_cache = os.path.join(BASE_DIR, "umap_2d.npy")
    if os.path.exists(umap_cache):
        umap_2d = np.load(umap_cache)
        if len(umap_2d) == len(embeddings):
            print(f"  Loaded cached UMAP: {umap_2d.shape}")
            return umap_2d

    import umap
    print("  Running UMAP (n_neighbors=30, min_dist=0.3)...")
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, n_components=2,
                        metric='cosine', random_state=42)
    umap_2d = reducer.fit_transform(embeddings)

    np.save(umap_cache, umap_2d)
    print(f"  UMAP complete: {umap_2d.shape}")
    return umap_2d


# ============================================================
# 3. Assign primary label for visualization
# ============================================================
def assign_labels(df):
    """Assign primary label: single theme, multi-label, or Class_0."""
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
# PLOT 1: Full UMAP scatter — all data points colored by theme
# ============================================================
def plot_full_umap(df, umap_2d, labels):
    """Full UMAP scatter plot with all data points."""
    print("\n  Plotting: Full UMAP scatter...")

    fig, ax = plt.subplots(figsize=(20, 16))

    # Plot Class_0 first (background)
    mask0 = np.array([l == 'Class_0' for l in labels])
    ax.scatter(umap_2d[mask0, 0], umap_2d[mask0, 1],
               c=THEME_COLORS['Class_0'], s=3, alpha=0.15, label='Class_0', zorder=1)

    # Plot themed points on top
    for theme in THEMES:
        mask = np.array([l == theme for l in labels])
        if mask.sum() > 0:
            ax.scatter(umap_2d[mask, 0], umap_2d[mask, 1],
                       c=THEME_COLORS[theme], s=12, alpha=0.6, label=theme, zorder=2)

    # Multi-label
    mask_multi = np.array([l == 'Multi_Label' for l in labels])
    if mask_multi.sum() > 0:
        ax.scatter(umap_2d[mask_multi, 0], umap_2d[mask_multi, 1],
                   c=THEME_COLORS['Multi_Label'], s=8, alpha=0.4,
                   marker='x', label='Multi_Label', zorder=3)

    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', markerscale=3, fontsize=10)
    ax.set_title('ALMA Sentence Embeddings — UMAP (all 19,724 sentences)', fontsize=16)
    ax.set_xlabel('UMAP-1')
    ax.set_ylabel('UMAP-2')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '01_full_umap_scatter.png'), dpi=200)
    plt.close()
    print("    Saved: 01_full_umap_scatter.png")


# ============================================================
# PLOT 2: Theme centroids with data points
# ============================================================
def plot_theme_centroids(df, umap_2d, embeddings):
    """Theme centroids as large markers with data points around them."""
    print("  Plotting: Theme centroids...")

    fig, ax = plt.subplots(figsize=(20, 16))

    # Compute centroids in embedding space, then project
    centroids_umap = {}
    for theme in THEMES:
        mask = df[theme] == 1
        if mask.sum() > 0:
            centroid_umap = umap_2d[mask].mean(axis=0)
            centroids_umap[theme] = centroid_umap

    # Class_0 centroid
    mask0 = df[THEMES].sum(axis=1) == 0
    centroids_umap['Class_0'] = umap_2d[mask0].mean(axis=0)

    # Plot all points faintly
    ax.scatter(umap_2d[:, 0], umap_2d[:, 1], c='#e0e0e0', s=2, alpha=0.1, zorder=1)

    # Plot themed points
    for theme in THEMES:
        mask = df[theme] == 1
        if mask.sum() > 0:
            ax.scatter(umap_2d[mask, 0], umap_2d[mask, 1],
                       c=THEME_COLORS[theme], s=8, alpha=0.3, zorder=2)

    # Plot centroids as large stars
    for theme, centroid in centroids_umap.items():
        color = THEME_COLORS.get(theme, '#000000')
        ax.scatter(centroid[0], centroid[1], c=color, s=500, marker='*',
                   edgecolors='black', linewidths=1.5, zorder=10)
        ax.annotate(theme, (centroid[0], centroid[1]),
                    fontsize=9, fontweight='bold',
                    xytext=(8, 8), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3))

    # Draw lines between close centroids
    centroid_list = list(centroids_umap.items())
    for i in range(len(centroid_list)):
        for j in range(i + 1, len(centroid_list)):
            name_i, pos_i = centroid_list[i]
            name_j, pos_j = centroid_list[j]
            dist = np.linalg.norm(pos_i - pos_j)
            if dist < np.percentile([np.linalg.norm(a[1] - b[1])
                                      for a in centroid_list for b in centroid_list
                                      if a[0] != b[0]], 30):
                ax.plot([pos_i[0], pos_j[0]], [pos_i[1], pos_j[1]],
                        'k--', alpha=0.2, linewidth=1)

    ax.set_title('Theme Centroids (★) with Data Points — UMAP Space', fontsize=16)
    ax.set_xlabel('UMAP-1')
    ax.set_ylabel('UMAP-2')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '02_theme_centroids.png'), dpi=200)
    plt.close()
    print("    Saved: 02_theme_centroids.png")

    return centroids_umap


# ============================================================
# PLOT 3: Per-theme density plots with overlap shading
# ============================================================
def plot_per_theme_density(df, umap_2d):
    """Per-theme KDE density + Class_0 density to show overlap regions."""
    print("  Plotting: Per-theme density overlaps...")

    fig, axes = plt.subplots(3, 4, figsize=(28, 21))
    axes = axes.flatten()

    mask0 = df[THEMES].sum(axis=1) == 0
    x0, y0 = umap_2d[mask0, 0], umap_2d[mask0, 1]

    for idx, theme in enumerate(THEMES):
        ax = axes[idx]
        mask_t = df[theme] == 1

        # Plot Class_0 density
        ax.scatter(x0, y0, c='#cccccc', s=1, alpha=0.08, zorder=1)

        # Plot theme density
        xt, yt = umap_2d[mask_t, 0], umap_2d[mask_t, 1]
        ax.scatter(xt, yt, c=THEME_COLORS[theme], s=6, alpha=0.5, zorder=3)

        # KDE contour for theme
        if mask_t.sum() > 20:
            try:
                from scipy.stats import gaussian_kde
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

        n_pos = mask_t.sum()
        ax.set_title(f'{theme} (n={n_pos})', fontsize=12, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])

    # Last subplot: summary
    axes[11].axis('off')
    summary_text = "DENSITY OVERLAP ANALYSIS\n\n"
    summary_text += "Shaded regions show where\ntheme sentences concentrate.\n\n"
    summary_text += "Gray dots = Class_0\n"
    summary_text += "Colored = Theme positive\n\n"
    summary_text += "Overlap with Class_0 gray\n= potential confusion zones"
    axes[11].text(0.1, 0.5, summary_text, fontsize=12, va='center',
                  transform=axes[11].transAxes)

    plt.suptitle('Per-Theme Density in UMAP Space (vs Class_0 gray)', fontsize=18, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '03_per_theme_density.png'), dpi=200)
    plt.close()
    print("    Saved: 03_per_theme_density.png")


# ============================================================
# PLOT 4: Correlation heatmap between themes
# ============================================================
def plot_correlation_analysis(df):
    """Theme co-occurrence correlation matrix."""
    print("  Plotting: Theme correlation heatmap...")

    # Binary correlation
    corr_matrix = df[THEMES].corr()

    fig, axes = plt.subplots(1, 2, figsize=(24, 10))

    # Pearson correlation
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdYlBu_r',
                vmin=-0.2, vmax=0.6, ax=axes[0], square=True,
                linewidths=0.5)
    axes[0].set_title('Pearson Correlation Between Themes', fontsize=14)

    # Jaccard similarity (co-occurrence)
    jaccard = np.zeros((len(THEMES), len(THEMES)))
    for i, t1 in enumerate(THEMES):
        for j, t2 in enumerate(THEMES):
            a = set(df[df[t1] == 1].index)
            b = set(df[df[t2] == 1].index)
            if len(a | b) > 0:
                jaccard[i, j] = len(a & b) / len(a | b)
            else:
                jaccard[i, j] = 0

    jaccard_df = pd.DataFrame(jaccard, index=THEMES, columns=THEMES)
    sns.heatmap(jaccard_df, annot=True, fmt='.3f', cmap='YlOrRd',
                vmin=0, vmax=0.5, ax=axes[1], square=True,
                linewidths=0.5)
    axes[1].set_title('Jaccard Similarity (Co-occurrence)', fontsize=14)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '04_correlation_heatmap.png'), dpi=200)
    plt.close()
    print("    Saved: 04_correlation_heatmap.png")

    # Print top co-occurring pairs
    print("\n  TOP CO-OCCURRING THEME PAIRS (Jaccard):")
    pairs = []
    for i in range(len(THEMES)):
        for j in range(i + 1, len(THEMES)):
            pairs.append((THEMES[i], THEMES[j], jaccard[i, j]))
    pairs.sort(key=lambda x: -x[2])
    for t1, t2, j in pairs[:10]:
        a = set(df[df[t1] == 1].index)
        b = set(df[df[t2] == 1].index)
        print(f"    {t1} ↔ {t2}: Jaccard={j:.3f}, co-occur={len(a & b)}")

    return corr_matrix, jaccard_df


# ============================================================
# PLOT 5: Class imbalance visualization
# ============================================================
def plot_imbalance(df):
    """Theme distribution bar chart and pie chart."""
    print("  Plotting: Class imbalance...")

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    # Bar chart
    counts = {t: (df[t] == 1).sum() for t in THEMES}
    counts['Class_0'] = (df[THEMES].sum(axis=1) == 0).sum()
    sorted_themes = sorted(counts.keys(), key=lambda x: -counts[x])
    colors = [THEME_COLORS.get(t, '#999999') for t in sorted_themes]

    bars = axes[0].barh(sorted_themes, [counts[t] for t in sorted_themes], color=colors)
    for bar, t in zip(bars, sorted_themes):
        axes[0].text(bar.get_width() + 50, bar.get_y() + bar.get_height() / 2,
                     f'{counts[t]:,} ({counts[t] / len(df) * 100:.1f}%)',
                     va='center', fontsize=9)
    axes[0].set_title('Sentence Count per Theme', fontsize=14)
    axes[0].set_xlabel('Count')
    axes[0].invert_yaxis()

    # Multi-label distribution
    n_labels = df[THEMES].sum(axis=1)
    label_counts = n_labels.value_counts().sort_index()
    axes[1].bar(label_counts.index, label_counts.values, color='steelblue')
    for i, (x, y) in enumerate(zip(label_counts.index, label_counts.values)):
        axes[1].text(x, y + 50, f'{y}\n({y / len(df) * 100:.1f}%)',
                     ha='center', fontsize=9)
    axes[1].set_title('Multi-Label Distribution', fontsize=14)
    axes[1].set_xlabel('Number of themes per sentence')
    axes[1].set_ylabel('Count')

    # Imbalance ratio
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
# PLOT 6: Decision boundary analysis with overlap regions
# ============================================================
def plot_decision_boundaries(df, umap_2d, embeddings):
    """KNN decision boundary with overlap/confusion regions shaded."""
    print("  Plotting: Decision boundaries with overlap regions...")

    # Create a simplified label for KNN: primary theme or Class_0
    primary_labels = []
    for _, row in df.iterrows():
        active = [t for t in THEMES if row[t] == 1]
        if len(active) == 0:
            primary_labels.append('Class_0')
        elif len(active) == 1:
            primary_labels.append(active[0])
        else:
            # For multi-label, use the rarest theme (most informative)
            theme_counts = {t: (df[t] == 1).sum() for t in active}
            primary_labels.append(min(theme_counts, key=theme_counts.get))
    primary_labels = np.array(primary_labels)

    # Encode labels
    unique_labels = list(set(primary_labels))
    label_to_int = {l: i for i, l in enumerate(unique_labels)}
    y = np.array([label_to_int[l] for l in primary_labels])

    # Fit KNN on UMAP coordinates
    knn = KNeighborsClassifier(n_neighbors=15, weights='distance')
    knn.fit(umap_2d, y)

    # Create mesh grid
    x_min, x_max = umap_2d[:, 0].min() - 1, umap_2d[:, 0].max() + 1
    y_min, y_max = umap_2d[:, 1].min() - 1, umap_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    mesh_points = np.c_[xx.ravel(), yy.ravel()]

    # Get probabilities for confusion detection
    probs = knn.predict_proba(mesh_points)
    max_probs = probs.max(axis=1).reshape(xx.shape)
    predictions = knn.predict(mesh_points).reshape(xx.shape)

    fig, axes = plt.subplots(1, 2, figsize=(28, 12))

    # Plot 1: Decision regions
    # Build colormap from unique_labels
    cmap_colors = [THEME_COLORS.get(l, '#999999') for l in unique_labels]
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(cmap_colors)

    axes[0].contourf(xx, yy, predictions, alpha=0.15, cmap=cmap, levels=len(unique_labels))
    # Overlay actual points
    mask0 = primary_labels == 'Class_0'
    axes[0].scatter(umap_2d[mask0, 0], umap_2d[mask0, 1],
                    c='#cccccc', s=2, alpha=0.1, zorder=1)
    for theme in THEMES:
        mask = primary_labels == theme
        if mask.sum() > 0:
            axes[0].scatter(umap_2d[mask, 0], umap_2d[mask, 1],
                            c=THEME_COLORS[theme], s=8, alpha=0.4, label=theme, zorder=2)
    axes[0].set_title('KNN Decision Regions (k=15)', fontsize=14)
    axes[0].legend(bbox_to_anchor=(1.02, 1), loc='upper left', markerscale=3, fontsize=8)

    # Plot 2: Confusion/overlap zones (low confidence = overlap)
    # Shade regions where max probability < 0.5 (uncertain)
    confusion = axes[1].contourf(xx, yy, max_probs, levels=[0, 0.3, 0.5, 0.7, 1.0],
                                  colors=['#ff0000', '#ff9900', '#ffff00', '#00ff00'],
                                  alpha=0.25)
    axes[1].scatter(umap_2d[:, 0], umap_2d[:, 1], c='#333333', s=1, alpha=0.1)

    # Highlight overlap zones with contour lines
    axes[1].contour(xx, yy, max_probs, levels=[0.3, 0.5],
                    colors=['red', 'orange'], linewidths=2, linestyles='--')

    cbar = plt.colorbar(confusion, ax=axes[1])
    cbar.set_label('KNN Confidence (max probability)')
    axes[1].set_title('Overlap/Confusion Zones\n(Red=High confusion, Green=Clear)', fontsize=14)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '06_decision_boundaries.png'), dpi=200)
    plt.close()
    print("    Saved: 06_decision_boundaries.png")

    return knn, label_to_int, unique_labels


# ============================================================
# PLOT 7: Pairwise theme overlap in embedding space
# ============================================================
def plot_pairwise_overlaps(df, umap_2d):
    """Show the most overlapping theme pairs side by side."""
    print("  Plotting: Pairwise theme overlaps...")

    # Find top 6 most overlapping pairs
    pairs = []
    for i, t1 in enumerate(THEMES):
        for j, t2 in enumerate(THEMES):
            if i < j:
                mask1 = df[t1] == 1
                mask2 = df[t2] == 1
                both = (mask1 & mask2).sum()
                if both > 0:
                    jaccard = both / (mask1 | mask2).sum()
                    pairs.append((t1, t2, jaccard, both))
    pairs.sort(key=lambda x: -x[2])
    top_pairs = pairs[:6]

    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    axes = axes.flatten()

    for idx, (t1, t2, jacc, both) in enumerate(top_pairs):
        ax = axes[idx]
        mask1 = df[t1] == 1
        mask2 = df[t2] == 1
        both_mask = mask1 & mask2
        only1 = mask1 & ~mask2
        only2 = mask2 & ~mask1
        neither = ~mask1 & ~mask2

        # Background
        ax.scatter(umap_2d[neither, 0], umap_2d[neither, 1],
                   c='#eeeeee', s=1, alpha=0.05)
        # Only theme 1
        ax.scatter(umap_2d[only1, 0], umap_2d[only1, 1],
                   c=THEME_COLORS[t1], s=10, alpha=0.5, label=f'{t1} only ({only1.sum()})')
        # Only theme 2
        ax.scatter(umap_2d[only2, 0], umap_2d[only2, 1],
                   c=THEME_COLORS[t2], s=10, alpha=0.5, label=f'{t2} only ({only2.sum()})')
        # Both
        ax.scatter(umap_2d[both_mask, 0], umap_2d[both_mask, 1],
                   c='black', s=15, alpha=0.7, marker='D', label=f'BOTH ({both})')

        ax.set_title(f'{t1} vs {t2}\nJaccard={jacc:.3f}, Co-occur={both}',
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='best')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle('Top 6 Most Overlapping Theme Pairs in Embedding Space', fontsize=16, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '07_pairwise_overlaps.png'), dpi=200)
    plt.close()
    print("    Saved: 07_pairwise_overlaps.png")


# ============================================================
# ANALYSIS: Identify suspicious data points
# ============================================================
def identify_suspicious_datapoints(df, embeddings, umap_2d):
    """
    Flag suspicious data points:
    1. Class_0 sentences close to theme centroids (potential false negatives)
    2. Theme sentences close to Class_0 centroid (potential false positives)
    3. Theme sentences far from their theme centroid (outliers)
    4. Sentences in high-overlap/confusion zones
    """
    print("\n" + "=" * 70)
    print("STEP 3: IDENTIFYING SUSPICIOUS DATA POINTS")
    print("=" * 70)

    flagged = []

    # Compute theme centroids in embedding space
    centroids_embed = {}
    for theme in THEMES:
        mask = df[theme] == 1
        if mask.sum() > 0:
            centroids_embed[theme] = embeddings[mask].mean(axis=0)

    # Class_0 centroid
    mask0 = df[THEMES].sum(axis=1) == 0
    centroids_embed['Class_0'] = embeddings[mask0].mean(axis=0)

    # ---- FLAG TYPE 1: Class_0 near theme centroids ----
    print("\n  [FLAG 1] Class_0 sentences near theme centroids...")
    class0_indices = np.where(mask0)[0]
    class0_embeds = embeddings[class0_indices]

    for theme in THEMES:
        centroid = centroids_embed[theme].reshape(1, -1)
        sims = cosine_similarity(class0_embeds, centroid).flatten()

        # Flag those with similarity > 75th percentile of actual theme members
        theme_mask = df[theme] == 1
        theme_embeds = embeddings[theme_mask]
        theme_sims = cosine_similarity(theme_embeds, centroid).flatten()
        threshold = np.percentile(theme_sims, 25)  # 25th percentile of real members

        suspicious = sims >= threshold
        n_suspicious = suspicious.sum()

        if n_suspicious > 0 and n_suspicious < 2000:  # sanity check
            for local_idx in np.where(suspicious)[0]:
                global_idx = class0_indices[local_idx]
                flagged.append({
                    'datapoint_index': int(global_idx),
                    'essay_id': int(df.iloc[global_idx]['essay_id']),
                    'sentence_id': int(df.iloc[global_idx]['sentence_id']),
                    'sentence': df.iloc[global_idx]['sentence'][:150],
                    'flag_type': 'Class0_near_theme',
                    'target_theme': theme,
                    'similarity': float(sims[local_idx]),
                    'threshold': float(threshold),
                    'recommendation': 'REVIEW: Class_0 but semantically close to theme'
                })

        print(f"    {theme}: {n_suspicious} Class_0 sentences above threshold ({threshold:.3f})")

    # ---- FLAG TYPE 2: Theme sentences far from their centroid ----
    print("\n  [FLAG 2] Theme sentences far from own centroid (outliers)...")
    for theme in THEMES:
        theme_mask = df[theme] == 1
        if theme_mask.sum() < 5:
            continue
        theme_indices = np.where(theme_mask)[0]
        theme_embeds = embeddings[theme_indices]
        centroid = centroids_embed[theme].reshape(1, -1)
        sims = cosine_similarity(theme_embeds, centroid).flatten()

        # Flag bottom 5% (furthest from centroid)
        threshold = np.percentile(sims, 5)
        outliers = sims <= threshold

        for local_idx in np.where(outliers)[0]:
            global_idx = theme_indices[local_idx]
            flagged.append({
                'datapoint_index': int(global_idx),
                'essay_id': int(df.iloc[global_idx]['essay_id']),
                'sentence_id': int(df.iloc[global_idx]['sentence_id']),
                'sentence': df.iloc[global_idx]['sentence'][:150],
                'flag_type': 'theme_outlier',
                'target_theme': theme,
                'similarity': float(sims[local_idx]),
                'threshold': float(threshold),
                'recommendation': f'REVIEW: Labeled {theme} but far from centroid'
            })

        n_out = outliers.sum()
        print(f"    {theme}: {n_out} outliers (sim < {threshold:.3f})")

    # ---- FLAG TYPE 3: Class_0 sentences in high-confusion UMAP zones ----
    print("\n  [FLAG 3] Class_0 in high-confusion UMAP zones...")
    # Use KNN to find Class_0 sentences surrounded by themed neighbors
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=10, metric='cosine')
    nn.fit(embeddings)

    class0_dists, class0_neighbors = nn.kneighbors(class0_embeds)

    # For each Class_0 sentence, check if majority of neighbors are themed
    theme_any = df[THEMES].sum(axis=1) > 0
    confused_count = 0
    for local_idx in range(len(class0_indices)):
        global_idx = class0_indices[local_idx]
        neighbor_indices = class0_neighbors[local_idx]
        n_themed_neighbors = sum(1 for ni in neighbor_indices if theme_any.iloc[ni])

        if n_themed_neighbors >= 7:  # 7+ of 10 neighbors are themed
            # Find which theme dominates
            theme_votes = {}
            for ni in neighbor_indices:
                for t in THEMES:
                    if df.iloc[ni][t] == 1:
                        theme_votes[t] = theme_votes.get(t, 0) + 1
            if theme_votes:
                dominant_theme = max(theme_votes, key=theme_votes.get)
                flagged.append({
                    'datapoint_index': int(global_idx),
                    'essay_id': int(df.iloc[global_idx]['essay_id']),
                    'sentence_id': int(df.iloc[global_idx]['sentence_id']),
                    'sentence': df.iloc[global_idx]['sentence'][:150],
                    'flag_type': 'Class0_in_confusion_zone',
                    'target_theme': dominant_theme,
                    'similarity': float(n_themed_neighbors / 10),
                    'threshold': 0.7,
                    'recommendation': f'HIGH PRIORITY: Class_0 surrounded by {dominant_theme} neighbors'
                })
                confused_count += 1

    print(f"    {confused_count} Class_0 sentences with 7+ themed neighbors")

    # Deduplicate flags (same datapoint might get flagged multiple times)
    seen = set()
    unique_flagged = []
    for f in flagged:
        key = (f['datapoint_index'], f['flag_type'], f.get('target_theme', ''))
        if key not in seen:
            seen.add(key)
            unique_flagged.append(f)

    flagged_df = pd.DataFrame(unique_flagged)
    print(f"\n  TOTAL UNIQUE FLAGS: {len(flagged_df)}")

    if len(flagged_df) > 0:
        print("\n  FLAGS BY TYPE:")
        for ft, group in flagged_df.groupby('flag_type'):
            print(f"    {ft}: {len(group)}")
            if 'target_theme' in group.columns:
                for theme, tg in group.groupby('target_theme'):
                    print(f"      → {theme}: {len(tg)}")

    return flagged_df


# ============================================================
# PLOT 8: Flagged data points highlighted on UMAP
# ============================================================
def plot_flagged_points(df, umap_2d, flagged_df):
    """Highlight flagged suspicious points on UMAP."""
    print("  Plotting: Flagged data points on UMAP...")

    if len(flagged_df) == 0:
        print("    No flagged points to plot.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(30, 10))

    flag_types = ['Class0_near_theme', 'theme_outlier', 'Class0_in_confusion_zone']
    titles = [
        'Class_0 Near Theme Centroids\n(Potential False Negatives)',
        'Theme Outliers\n(Potential False Positives)',
        'Class_0 in Confusion Zones\n(High Priority Review)'
    ]

    for idx, (ft, title) in enumerate(zip(flag_types, titles)):
        ax = axes[idx]

        # Background: all points gray
        ax.scatter(umap_2d[:, 0], umap_2d[:, 1], c='#e8e8e8', s=1, alpha=0.1)

        # Flagged points
        ft_flags = flagged_df[flagged_df['flag_type'] == ft]
        if len(ft_flags) > 0:
            flagged_indices = ft_flags['datapoint_index'].values
            # Color by target theme
            for theme in THEMES:
                theme_flags = ft_flags[ft_flags['target_theme'] == theme]
                if len(theme_flags) > 0:
                    fi = theme_flags['datapoint_index'].values
                    ax.scatter(umap_2d[fi, 0], umap_2d[fi, 1],
                               c=THEME_COLORS[theme], s=20, alpha=0.7,
                               edgecolors='red', linewidths=0.5,
                               label=f'{theme} ({len(theme_flags)})')

        ax.set_title(f'{title}\n({len(ft_flags)} points)', fontsize=12)
        ax.legend(fontsize=7, loc='best', ncol=2)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle('Suspicious Data Points Flagged for Review', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '08_flagged_datapoints.png'), dpi=200)
    plt.close()
    print("    Saved: 08_flagged_datapoints.png")


# ============================================================
# PLOT 9: Cosine similarity distributions
# ============================================================
def plot_similarity_distributions(df, embeddings):
    """Distribution of cosine similarities within and between themes."""
    print("  Plotting: Similarity distributions...")

    fig, axes = plt.subplots(3, 4, figsize=(28, 21))
    axes = axes.flatten()

    centroids_embed = {}
    for theme in THEMES:
        mask = df[theme] == 1
        if mask.sum() > 0:
            centroids_embed[theme] = embeddings[mask].mean(axis=0)
    mask0 = df[THEMES].sum(axis=1) == 0
    centroids_embed['Class_0'] = embeddings[mask0].mean(axis=0)

    for idx, theme in enumerate(THEMES):
        ax = axes[idx]
        mask_t = df[theme] == 1
        if mask_t.sum() < 5:
            ax.set_title(f'{theme} (too few)')
            continue

        centroid = centroids_embed[theme].reshape(1, -1)

        # Similarity of theme members to theme centroid
        theme_sims = cosine_similarity(embeddings[mask_t], centroid).flatten()
        # Similarity of Class_0 to theme centroid
        class0_sims = cosine_similarity(embeddings[mask0], centroid).flatten()

        ax.hist(theme_sims, bins=50, alpha=0.7, color=THEME_COLORS[theme],
                label=f'{theme} (n={mask_t.sum()})', density=True)
        ax.hist(class0_sims, bins=50, alpha=0.4, color='gray',
                label=f'Class_0 (n={mask0.sum()})', density=True)

        # Mark overlap zone
        overlap_low = max(theme_sims.min(), np.percentile(class0_sims, 75))
        overlap_high = min(theme_sims.max(), np.percentile(theme_sims, 25))
        if overlap_low < overlap_high:
            ax.axvspan(overlap_low, overlap_high, color='red', alpha=0.15,
                       label='Overlap zone')

        ax.set_title(f'{theme}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=7)
        ax.set_xlabel('Cosine similarity to centroid')

    axes[11].axis('off')
    axes[11].text(0.1, 0.5,
                  "SIMILARITY DISTRIBUTIONS\n\n"
                  "Colored = theme sentences\n"
                  "Gray = Class_0 sentences\n\n"
                  "Red shading = overlap zone\n"
                  "where Class_0 looks like theme\n\n"
                  "More overlap = harder to\nseparate during training",
                  fontsize=12, va='center', transform=axes[11].transAxes)

    plt.suptitle('Cosine Similarity to Theme Centroids: Theme vs Class_0', fontsize=16, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '09_similarity_distributions.png'), dpi=200)
    plt.close()
    print("    Saved: 09_similarity_distributions.png")


# ============================================================
# PLOT 10: Embedding clusters (K-Means)
# ============================================================
def plot_kmeans_clusters(df, umap_2d, embeddings):
    """K-Means clustering vs actual labels."""
    print("  Plotting: K-Means clusters vs actual labels...")

    # Try k=12 (11 themes + Class_0)
    kmeans = KMeans(n_clusters=12, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)

    fig, axes = plt.subplots(1, 2, figsize=(24, 10))

    # Plot 1: K-Means clusters
    scatter1 = axes[0].scatter(umap_2d[:, 0], umap_2d[:, 1],
                                c=cluster_labels, cmap='Set3', s=3, alpha=0.3)
    axes[0].set_title('K-Means Clusters (k=12) in UMAP Space', fontsize=14)
    plt.colorbar(scatter1, ax=axes[0], label='Cluster')

    # Plot 2: Actual labels
    primary = []
    for _, row in df.iterrows():
        active = [t for t in THEMES if row[t] == 1]
        if len(active) == 0:
            primary.append(0)
        else:
            primary.append(THEMES.index(active[0]) + 1)
    primary = np.array(primary)

    scatter2 = axes[1].scatter(umap_2d[:, 0], umap_2d[:, 1],
                                c=primary, cmap='Set3', s=3, alpha=0.3)
    axes[1].set_title('Actual Theme Labels in UMAP Space', fontsize=14)
    plt.colorbar(scatter2, ax=axes[1], label='Theme')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '10_kmeans_vs_actual.png'), dpi=200)
    plt.close()
    print("    Saved: 10_kmeans_vs_actual.png")

    # Cluster purity analysis
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
        print(f"    Cluster {c}: n={n}, dominant={dominant} ({purity:.1%}), "
              f"themes={dict(sorted(theme_counts.items(), key=lambda x:-x[1])[:3])}")


# ============================================================
# PLOT 11: Theme-specific embedding space with Class_0 overlap
# ============================================================
def plot_theme_vs_class0_overlap(df, umap_2d, embeddings):
    """For each theme, shade the UMAP region where Class_0 overlaps."""
    print("  Plotting: Theme vs Class_0 overlap regions...")

    mask0 = df[THEMES].sum(axis=1) == 0

    # Top 6 themes by size
    theme_sizes = {t: (df[t] == 1).sum() for t in THEMES}
    top_themes = sorted(theme_sizes, key=lambda x: -theme_sizes[x])[:6]

    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    axes = axes.flatten()

    for idx, theme in enumerate(top_themes):
        ax = axes[idx]
        mask_t = df[theme] == 1

        xt, yt = umap_2d[mask_t, 0], umap_2d[mask_t, 1]
        x0, y0 = umap_2d[mask0, 0], umap_2d[mask0, 1]

        # Background
        ax.scatter(umap_2d[:, 0], umap_2d[:, 1], c='#f5f5f5', s=1, alpha=0.05)

        # Theme KDE
        from scipy.stats import gaussian_kde
        try:
            xy_t = np.vstack([xt, yt])
            kde_t = gaussian_kde(xy_t, bw_method=0.3)

            xmin, xmax = umap_2d[:, 0].min() - 1, umap_2d[:, 0].max() + 1
            ymin, ymax = umap_2d[:, 1].min() - 1, umap_2d[:, 1].max() + 1
            xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
            positions = np.vstack([xx.ravel(), yy.ravel()])

            z_t = np.reshape(kde_t(positions), xx.shape)

            # Class_0 KDE (subsample for speed)
            c0_sample = np.random.choice(mask0.sum(), min(3000, mask0.sum()), replace=False)
            xy_0 = np.vstack([x0[c0_sample], y0[c0_sample]])
            kde_0 = gaussian_kde(xy_0, bw_method=0.3)
            z_0 = np.reshape(kde_0(positions), xx.shape)

            # Normalize
            z_t_norm = z_t / (z_t.max() + 1e-10)
            z_0_norm = z_0 / (z_0.max() + 1e-10)

            # Overlap = minimum of two densities
            overlap = np.minimum(z_t_norm, z_0_norm)

            # Plot theme density (blue)
            ax.contourf(xx, yy, z_t_norm, levels=5,
                        colors=[THEME_COLORS[theme]], alpha=0.2)
            ax.contour(xx, yy, z_t_norm, levels=3,
                       colors=[THEME_COLORS[theme]], alpha=0.5, linewidths=1)

            # Plot overlap region (RED shading)
            overlap_threshold = 0.2
            ax.contourf(xx, yy, overlap, levels=[overlap_threshold, 1.0],
                        colors=['red'], alpha=0.3)
            ax.contour(xx, yy, overlap, levels=[overlap_threshold],
                       colors=['red'], linewidths=2, linestyles='--')

        except Exception as e:
            pass

        # Plot actual points
        ax.scatter(x0, y0, c='gray', s=1, alpha=0.05)
        ax.scatter(xt, yt, c=THEME_COLORS[theme], s=6, alpha=0.4)

        ax.set_title(f'{theme} (n={mask_t.sum()}) — Red = Overlap with Class_0',
                     fontsize=11, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle('Theme Density vs Class_0 — Red Shaded = OVERLAP REGIONS (confusion zones)',
                 fontsize=16, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '11_theme_class0_overlap.png'), dpi=200)
    plt.close()
    print("    Saved: 11_theme_class0_overlap.png")


# ============================================================
# Summary report
# ============================================================
def generate_summary(df, flagged_df, corr_matrix):
    """Generate text summary of all findings."""
    print("\n" + "=" * 70)
    print("COMPREHENSIVE ANALYSIS SUMMARY")
    print("=" * 70)

    print(f"\n  Dataset: {len(df)} sentences, {df['essay_id'].nunique()} essays")
    themed = (df[THEMES].sum(axis=1) > 0).sum()
    class0 = (df[THEMES].sum(axis=1) == 0).sum()
    print(f"  Themed: {themed} ({themed / len(df) * 100:.1f}%)")
    print(f"  Class_0: {class0} ({class0 / len(df) * 100:.1f}%)")

    print(f"\n  IMBALANCE SUMMARY:")
    for t in THEMES:
        pos = (df[t] == 1).sum()
        neg = (df[t] == 0).sum()
        ratio = neg / max(pos, 1)
        severity = "CRITICAL" if ratio > 100 else "SEVERE" if ratio > 20 else "MODERATE" if ratio > 10 else "OK"
        print(f"    {t:<30} {pos:>5} pos, ratio {ratio:>6.0f}:1 [{severity}]")

    if len(flagged_df) > 0:
        print(f"\n  FLAGGED DATA POINTS: {len(flagged_df)} total")
        for ft, group in flagged_df.groupby('flag_type'):
            unique_pts = group['datapoint_index'].nunique()
            print(f"    {ft}: {len(group)} flags ({unique_pts} unique sentences)")

        # Count unique datapoints across all flags
        all_flagged = flagged_df['datapoint_index'].unique()
        print(f"\n  UNIQUE SENTENCES FLAGGED: {len(all_flagged)}")
        print(f"  As % of dataset: {len(all_flagged) / len(df) * 100:.1f}%")

        # High priority flags
        hp = flagged_df[flagged_df['flag_type'] == 'Class0_in_confusion_zone']
        print(f"\n  HIGH PRIORITY (Class_0 in confusion zones): {len(hp)}")
        if len(hp) > 0:
            print(f"  Top themes confused with Class_0:")
            for theme, tg in hp.groupby('target_theme'):
                print(f"    {theme}: {len(tg)} sentences")

    print(f"\n  HIGHLY CORRELATED THEME PAIRS (Pearson > 0.2):")
    for i in range(len(THEMES)):
        for j in range(i + 1, len(THEMES)):
            if corr_matrix.iloc[i, j] > 0.2:
                print(f"    {THEMES[i]} ↔ {THEMES[j]}: r={corr_matrix.iloc[i, j]:.3f}")


# ============================================================
# MAIN
# ============================================================
def main():
    print("Loading dataset...")
    df = pd.read_csv(INPUT_FILE)
    print(f"  {len(df)} sentences loaded")

    # Step 1: Embeddings
    embeddings = load_and_embed(df)

    # Step 2: UMAP
    umap_2d = compute_umap(embeddings)

    # Assign display labels
    labels = assign_labels(df)

    print("\n" + "=" * 70)
    print("STEP 3: GENERATING VISUALIZATIONS")
    print("=" * 70)

    # Plot 1: Full UMAP scatter
    plot_full_umap(df, umap_2d, labels)

    # Plot 2: Theme centroids
    centroids_umap = plot_theme_centroids(df, umap_2d, embeddings)

    # Plot 3: Per-theme density
    plot_per_theme_density(df, umap_2d)

    # Plot 4: Correlation
    corr_matrix, jaccard_df = plot_correlation_analysis(df)

    # Plot 5: Imbalance
    plot_imbalance(df)

    # Plot 6: Decision boundaries
    knn, label_to_int, unique_labels = plot_decision_boundaries(df, umap_2d, embeddings)

    # Plot 7: Pairwise overlaps
    plot_pairwise_overlaps(df, umap_2d)

    # Identify suspicious data points
    flagged_df = identify_suspicious_datapoints(df, embeddings, umap_2d)

    # Plot 8: Flagged points
    plot_flagged_points(df, umap_2d, flagged_df)

    # Plot 9: Similarity distributions
    plot_similarity_distributions(df, embeddings)

    # Plot 10: K-Means vs actual
    plot_kmeans_clusters(df, umap_2d, embeddings)

    # Plot 11: Overlap regions
    plot_theme_vs_class0_overlap(df, umap_2d, embeddings)

    # Save flagged data points
    if len(flagged_df) > 0:
        flagged_path = os.path.join(BASE_DIR, "flagged_datapoints.csv")
        flagged_df.to_csv(flagged_path, index=False)
        print(f"\n  Saved flagged datapoints: {flagged_path}")
        print(f"  Total flags: {len(flagged_df)}")

    # Summary
    generate_summary(df, flagged_df, corr_matrix)

    print(f"\n{'=' * 70}")
    print(f"ALL PLOTS SAVED TO: {PLOT_DIR}/")
    print(f"{'=' * 70}")
    print("Files:")
    for f in sorted(os.listdir(PLOT_DIR)):
        print(f"  {f}")


if __name__ == '__main__':
    main()
