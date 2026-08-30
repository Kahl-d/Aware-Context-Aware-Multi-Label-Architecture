"""
ALMA Round 2 — Single-Label-Only Analysis
Shows ONLY sentences with exactly 1 theme (or Class_0).
Removes multi-label noise to see "pure" class structure.
Reuses cached embeddings/UMAP from comprehensive_analysis_v2.py.
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import gaussian_kde
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# ─── Config ──────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, 'ALMA_processed_master_dataset.csv')
PLOT_DIR = os.path.join(BASE_DIR, 'plots_single_label')
os.makedirs(PLOT_DIR, exist_ok=True)

ALL_THEMES = ['Aspirational', 'Familial', 'Social', 'Navigational', 'Resistance',
              'Attainment', 'First_Gen', 'Perseverance', 'Filial_Piety',
              'Community_Consciousness', 'Spiritual']
ALL_CLASSES = ['Class_0'] + ALL_THEMES

CLASS_COLORS = {
    'Class_0': '#999999',
    'Aspirational': '#e6194b',
    'Familial': '#3cb44b',
    'Social': '#4363d8',
    'Navigational': '#f58231',
    'Resistance': '#911eb4',
    'Attainment': '#42d4f4',
    'First_Gen': '#f032e6',
    'Perseverance': '#bfef45',
    'Filial_Piety': '#fabed4',
    'Community_Consciousness': '#dcbeff',
    'Spiritual': '#ffe119'
}

def load_and_filter():
    """Load data, filter to single-label + Class_0 only."""
    print("Loading data...")
    df = pd.read_csv(DATA_FILE)
    if 'Class_0' not in df.columns:
        df['Class_0'] = (df[ALL_THEMES].sum(axis=1) == 0).astype(int)

    theme_counts = df[ALL_THEMES].sum(axis=1)
    # Single-label: exactly 1 theme OR Class_0 (0 themes)
    single_mask = (theme_counts <= 1)
    df_single = df[single_mask].copy().reset_index(drop=True)

    # Assign the single label
    labels = []
    for _, row in df_single.iterrows():
        active = [t for t in ALL_THEMES if row[t] == 1]
        labels.append(active[0] if active else 'Class_0')
    df_single['label'] = labels

    # Store original indices for ID tracking
    df_single['original_index'] = df[single_mask].index.values

    print(f"  Total: {len(df)} → Single-label: {len(df_single)} ({len(df_single)/len(df)*100:.1f}%)")
    print(f"  Multi-label excluded: {(~single_mask).sum()}")
    print(f"\n  Single-label distribution:")
    for cls in ALL_CLASSES:
        n = (df_single['label'] == cls).sum()
        print(f"    {cls:30s}: {n:6d} ({n/len(df_single)*100:5.1f}%)")

    return df, df_single, single_mask

def load_cached_embeddings(single_mask):
    """Load cached embeddings and UMAP, filter to single-label."""
    emb_path = os.path.join(BASE_DIR, 'v2_embeddings.npy')
    umap2d_path = os.path.join(BASE_DIR, 'v2_umap_2d.npy')
    umap3d_path = os.path.join(BASE_DIR, 'v2_umap_3d.npy')

    embeddings_full = np.load(emb_path)
    umap_2d_full = np.load(umap2d_path)
    umap_3d_full = np.load(umap3d_path)

    embeddings = embeddings_full[single_mask]
    umap_2d = umap_2d_full[single_mask]
    umap_3d = umap_3d_full[single_mask]

    print(f"  Embeddings: {embeddings.shape}")
    return embeddings, umap_2d, umap_3d

def compute_centroids(df_single, embeddings):
    """Centroids from single-label sentences only — purest representations."""
    centroids = {}
    for cls in ALL_CLASSES:
        mask = df_single['label'] == cls
        if mask.sum() > 0:
            centroids[cls] = embeddings[mask.values].mean(axis=0)
    return centroids

# ═══════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════

def plot_19_single_umap(df_single, umap_2d):
    """UMAP scatter — single-label sentences only."""
    print("Plot 19: Single-label UMAP scatter...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(28, 12))

    # Left: all single-label
    mask_c0 = df_single['label'] == 'Class_0'
    ax1.scatter(umap_2d[mask_c0, 0], umap_2d[mask_c0, 1], c=CLASS_COLORS['Class_0'],
                s=3, alpha=0.1, label=f'Class_0 ({mask_c0.sum():,})')
    for theme in ALL_THEMES:
        mask = df_single['label'] == theme
        if mask.sum() > 0:
            ax1.scatter(umap_2d[mask, 0], umap_2d[mask, 1], c=CLASS_COLORS[theme],
                        s=10, alpha=0.6, label=f'{theme} ({mask.sum():,})')
    ax1.set_title('UMAP — Single-Label Only (No Multi-Label Noise)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=8, markerscale=3, loc='upper left')

    # Right: without Class_0 — themes only
    for theme in ALL_THEMES:
        mask = df_single['label'] == theme
        if mask.sum() > 0:
            ax2.scatter(umap_2d[mask, 0], umap_2d[mask, 1], c=CLASS_COLORS[theme],
                        s=12, alpha=0.6, label=f'{theme} ({mask.sum():,})')
    ax2.set_title('UMAP — Single-Label THEMES Only (No Class_0)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=8, markerscale=3, loc='upper left')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '19_single_label_umap.png'), dpi=200)
    plt.close()

def plot_20_single_density(df_single, umap_2d):
    """Per-class KDE density — single-label only."""
    print("Plot 20: Single-label per-class density...")
    fig, axes = plt.subplots(3, 4, figsize=(24, 18))

    for idx, cls in enumerate(ALL_CLASSES):
        ax = axes[idx // 4][idx % 4]
        mask = df_single['label'] == cls
        n = mask.sum()

        ax.scatter(umap_2d[:, 0], umap_2d[:, 1], c='lightgray', s=1, alpha=0.05)

        if n > 10:
            pts = umap_2d[mask]
            ax.scatter(pts[:, 0], pts[:, 1], c=CLASS_COLORS[cls], s=5, alpha=0.4)
            try:
                kde = gaussian_kde(pts.T, bw_method=0.3)
                xmin, xmax = umap_2d[:, 0].min()-1, umap_2d[:, 0].max()+1
                ymin, ymax = umap_2d[:, 1].min()-1, umap_2d[:, 1].max()+1
                xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
                zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
                ax.contour(xx, yy, zz, levels=5, colors=[CLASS_COLORS[cls]], alpha=0.6)
            except Exception:
                pass

        ax.set_title(f'{cls} (n={n:,})', fontsize=10, fontweight='bold', color=CLASS_COLORS[cls])
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle('Per-Class Density — SINGLE-LABEL ONLY (Pure Class Representation)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(PLOT_DIR, '20_single_label_density.png'), dpi=200)
    plt.close()

def plot_21_single_centroids(df_single, umap_2d, embeddings, centroids):
    """Centroids from single-label data only — purest centroids."""
    print("Plot 21: Single-label centroids...")
    fig, ax = plt.subplots(figsize=(16, 12))

    ax.scatter(umap_2d[:, 0], umap_2d[:, 1], c='lightgray', s=1, alpha=0.1)

    for cls in ALL_CLASSES:
        if cls not in centroids:
            continue
        mask = df_single['label'] == cls
        cent = centroids[cls]
        dists = np.linalg.norm(embeddings - cent, axis=1)
        nearest_idx = np.argmin(dists)
        cx, cy = umap_2d[nearest_idx]

        ax.scatter(umap_2d[mask, 0], umap_2d[mask, 1], c=CLASS_COLORS[cls], s=4, alpha=0.2)
        ax.scatter(cx, cy, c=CLASS_COLORS[cls], s=400, marker='*', edgecolors='black',
                   linewidths=1.5, zorder=10)
        ax.annotate(cls, (cx, cy), fontsize=9, fontweight='bold',
                    xytext=(5, 5), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=CLASS_COLORS[cls], alpha=0.6))

    ax.set_title('Single-Label Centroids — Pure Class Representations (★)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '21_single_label_centroids.png'), dpi=200)
    plt.close()

def plot_22_single_vs_all_centroids(df_single, df_full, embeddings_single, embeddings_full_path,
                                     centroids_single, umap_2d):
    """Compare single-label centroids vs all-data centroids — how much does multi-label shift them."""
    print("Plot 22: Single-label vs all-data centroid comparison...")
    embeddings_full = np.load(embeddings_full_path)
    centroids_all = {}
    for cls in ALL_CLASSES:
        if cls == 'Class_0':
            mask = (df_full[ALL_THEMES].sum(axis=1) == 0)
        else:
            mask = df_full[cls] == 1
        if mask.sum() > 0:
            centroids_all[cls] = embeddings_full[mask.values].mean(axis=0)

    # Compute similarity between single-label and all-data centroids
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))

    classes = [c for c in ALL_CLASSES if c in centroids_single and c in centroids_all]
    sims = []
    dists = []
    for cls in classes:
        v1 = centroids_single[cls]
        v2 = centroids_all[cls]
        sim = (v1 @ v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        dist = np.linalg.norm(v1 - v2)
        sims.append(sim)
        dists.append(dist)

    colors = [CLASS_COLORS[c] for c in classes]

    ax1.barh(range(len(classes)), sims, color=colors)
    ax1.set_yticks(range(len(classes)))
    ax1.set_yticklabels(classes)
    for i, s in enumerate(sims):
        ax1.text(s + 0.001, i, f'{s:.4f}', va='center', fontsize=9)
    ax1.set_title('Cosine Similarity: Single-Label vs All-Data Centroids', fontweight='bold')
    ax1.set_xlabel('Cosine Similarity')
    ax1.set_xlim(0.95, 1.001)
    ax1.invert_yaxis()

    ax2.barh(range(len(classes)), dists, color=colors)
    ax2.set_yticks(range(len(classes)))
    ax2.set_yticklabels(classes)
    for i, d in enumerate(dists):
        ax2.text(d + 0.01, i, f'{d:.4f}', va='center', fontsize=9)
    ax2.set_title('Euclidean Distance: Single-Label vs All-Data Centroids\n(How much multi-label shifts the centroid)',
                  fontweight='bold')
    ax2.set_xlabel('Euclidean Distance')
    ax2.invert_yaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '22_single_vs_all_centroids.png'), dpi=200)
    plt.close()

def plot_23_single_similarity(df_single, embeddings, centroids):
    """Similarity distributions — single-label only (cleanest signal)."""
    print("Plot 23: Single-label similarity distributions...")
    fig, axes = plt.subplots(3, 4, figsize=(24, 18))

    for idx, cls in enumerate(ALL_CLASSES):
        ax = axes[idx // 4][idx % 4]
        if cls not in centroids:
            ax.set_title(f'{cls} — No centroid')
            continue

        cent = centroids[cls]
        norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(cent)
        sims = (embeddings @ cent) / (norms + 1e-10)

        member_mask = df_single['label'] == cls
        member_sims = sims[member_mask]
        nonmember_sims = sims[~member_mask]

        ax.hist(nonmember_sims, bins=80, alpha=0.4, color='gray', density=True, label='Non-members')
        ax.hist(member_sims, bins=80, alpha=0.6, color=CLASS_COLORS[cls], density=True, label='Members')
        ax.axvline(np.mean(member_sims), color=CLASS_COLORS[cls], linestyle='--', linewidth=2)
        ax.axvline(np.mean(nonmember_sims), color='gray', linestyle='--', linewidth=2)

        # Separation metric: difference of means / pooled std
        if len(member_sims) > 1 and len(nonmember_sims) > 1:
            sep = (np.mean(member_sims) - np.mean(nonmember_sims)) / (
                np.sqrt((np.std(member_sims)**2 + np.std(nonmember_sims)**2) / 2) + 1e-10)
        else:
            sep = 0
        ax.set_title(f'{cls} (d\'={sep:.2f})', fontsize=10, fontweight='bold', color=CLASS_COLORS[cls])
        ax.legend(fontsize=7)

    fig.suptitle('Similarity to Centroid — SINGLE-LABEL ONLY\n(d\' = separability: higher = easier to classify)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(os.path.join(PLOT_DIR, '23_single_label_similarity.png'), dpi=200)
    plt.close()

def plot_24_single_decision_boundaries(df_single, umap_2d):
    """Decision boundaries — single-label only (cleaner boundaries)."""
    print("Plot 24: Single-label decision boundaries (KNN k=15)...")
    le = LabelEncoder()
    labels_enc = le.fit_transform(df_single['label'].values)

    knn = KNeighborsClassifier(n_neighbors=15, metric='euclidean', weights='distance')
    knn.fit(umap_2d, labels_enc)

    xmin, xmax = umap_2d[:, 0].min()-1, umap_2d[:, 0].max()+1
    ymin, ymax = umap_2d[:, 1].min()-1, umap_2d[:, 1].max()+1
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 300), np.linspace(ymin, ymax, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]

    probs = knn.predict_proba(grid)
    max_prob = probs.max(axis=1).reshape(xx.shape)
    pred = knn.predict(grid).reshape(xx.shape)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(28, 12))

    color_list = [CLASS_COLORS.get(c, '#333333') for c in le.classes_]
    cmap = ListedColormap(color_list)
    ax1.contourf(xx, yy, pred, cmap=cmap, alpha=0.2)
    for cls in ALL_CLASSES:
        mask = df_single['label'] == cls
        if mask.sum() > 0:
            ax1.scatter(umap_2d[mask, 0], umap_2d[mask, 1], c=CLASS_COLORS[cls],
                        s=5, alpha=0.4, label=cls)
    ax1.set_title('Decision Regions — SINGLE-LABEL ONLY (k=15)', fontweight='bold')
    ax1.legend(fontsize=7, markerscale=3, loc='upper left')

    conf = ax2.contourf(xx, yy, max_prob, levels=20, cmap='RdYlGn', alpha=0.8)
    plt.colorbar(conf, ax=ax2, label='KNN Confidence')
    ax2.contour(xx, yy, max_prob, levels=[0.3, 0.5], colors=['red', 'orange'], linewidths=2)
    ax2.scatter(umap_2d[:, 0], umap_2d[:, 1], c='black', s=1, alpha=0.05)
    ax2.set_title('Confidence Map — SINGLE-LABEL ONLY\n(Cleaner boundaries without multi-label noise)', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '24_single_label_boundaries.png'), dpi=200)
    plt.close()

def plot_25_single_pairwise_similarity(centroids):
    """Pairwise similarity between single-label centroids."""
    print("Plot 25: Single-label pairwise centroid similarity...")
    classes = [c for c in ALL_CLASSES if c in centroids]
    n = len(classes)
    sim_matrix = np.zeros((n, n))
    for i, c1 in enumerate(classes):
        for j, c2 in enumerate(classes):
            v1 = centroids[c1]
            v2 = centroids[c2]
            sim_matrix[i, j] = (v1 @ v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(sim_matrix, xticklabels=classes, yticklabels=classes,
                annot=True, fmt='.3f', cmap='RdYlBu_r', ax=ax,
                vmin=0.5, vmax=1.0, square=True)
    ax.set_title('Pairwise Similarity — SINGLE-LABEL Centroids\n(Purest class representations)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '25_single_label_pairwise.png'), dpi=200)
    plt.close()

def plot_26_single_3d_umap(df_single, umap_3d):
    """3D UMAP — single-label only, multiple angles."""
    print("Plot 26: Single-label 3D UMAP...")
    fig = plt.figure(figsize=(24, 18))

    angles = [(30, 45), (30, 135), (60, 45), (10, 90)]
    for i, (elev, azim) in enumerate(angles):
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        for cls in ALL_CLASSES:
            mask = df_single['label'] == cls
            if mask.sum() > 0:
                alpha = 0.06 if cls == 'Class_0' else 0.5
                size = 2 if cls == 'Class_0' else 8
                ax.scatter(umap_3d[mask, 0], umap_3d[mask, 1], umap_3d[mask, 2],
                           c=CLASS_COLORS[cls], s=size, alpha=alpha, label=cls)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f'elev={elev}, azim={azim}', fontsize=10)
        if i == 0:
            ax.legend(fontsize=6, markerscale=3, loc='upper left')

    fig.suptitle('3D UMAP — SINGLE-LABEL ONLY (Multiple Angles)', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(PLOT_DIR, '26_single_label_3d_umap.png'), dpi=200)
    plt.close()

def plot_27_single_kmeans_optimal(embeddings):
    """Optimal K for single-label data — different from full data?"""
    print("Plot 27: Single-label optimal K...")
    from sklearn.decomposition import PCA
    pca = PCA(n_components=50, random_state=42)
    pca_50 = pca.fit_transform(embeddings)

    K_range = range(2, 21)
    silhouettes = []
    inertias = []
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(pca_50)
        silhouettes.append(silhouette_score(pca_50, labels, sample_size=5000, random_state=42))
        inertias.append(km.inertia_)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.plot(K_range, inertias, 'bo-')
    ax1.set_title('Elbow (Single-Label Only)', fontweight='bold')
    ax1.set_xlabel('K')
    ax1.set_ylabel('Inertia')

    ax2.plot(K_range, silhouettes, 'go-')
    best_k = list(K_range)[np.argmax(silhouettes)]
    ax2.axvline(best_k, color='r', linestyle='--', label=f'Best K={best_k}')
    ax2.set_title('Silhouette (Single-Label Only)', fontweight='bold')
    ax2.set_xlabel('K')
    ax2.set_ylabel('Silhouette Score')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '27_single_label_optimal_k.png'), dpi=200)
    plt.close()
    return best_k, pca_50

def plot_28_single_kmeans_composition(df_single, umap_2d, pca_50, best_k):
    """K-Means on single-label data — cluster composition."""
    print(f"Plot 28: Single-label K-Means (K={best_k}) composition...")
    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    cluster_labels = km.fit_predict(pca_50)

    fig, axes = plt.subplots(2, 2, figsize=(24, 20))

    # Top-left: clusters on UMAP
    scatter = axes[0, 0].scatter(umap_2d[:, 0], umap_2d[:, 1], c=cluster_labels,
                                  cmap='tab20', s=3, alpha=0.4)
    axes[0, 0].set_title(f'K-Means Clusters (K={best_k}) — Single-Label', fontweight='bold')
    plt.colorbar(scatter, ax=axes[0, 0])

    # Top-right: actual labels
    for cls in ALL_CLASSES:
        mask = df_single['label'] == cls
        if mask.sum() > 0:
            axes[0, 1].scatter(umap_2d[mask, 0], umap_2d[mask, 1], c=CLASS_COLORS[cls],
                               s=3, alpha=0.3, label=cls)
    axes[0, 1].set_title('Actual Labels — Single-Label', fontweight='bold')
    axes[0, 1].legend(fontsize=7, markerscale=3, loc='upper left')

    # Bottom: Cluster composition
    comp = pd.DataFrame({'cluster': cluster_labels, 'label': df_single['label'].values})
    ct = pd.crosstab(comp['cluster'], comp['label'])
    # Reorder columns
    ct = ct[[c for c in ALL_CLASSES if c in ct.columns]]

    ct.plot(kind='bar', stacked=True, ax=axes[1, 0],
            color=[CLASS_COLORS[c] for c in ct.columns])
    axes[1, 0].set_title('Cluster Composition (Absolute)', fontweight='bold')
    axes[1, 0].legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc='upper left')

    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
    ct_pct.plot(kind='bar', stacked=True, ax=axes[1, 1],
                color=[CLASS_COLORS[c] for c in ct_pct.columns])
    axes[1, 1].set_title('Cluster Composition (Percentage)', fontweight='bold')
    axes[1, 1].set_ylabel('%')
    axes[1, 1].legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '28_single_label_kmeans_composition.png'), dpi=200)
    plt.close()

    # Print dominant class per cluster
    print(f"\n  Cluster composition (single-label K={best_k}):")
    for c in range(best_k):
        row = ct.loc[c]
        total = row.sum()
        dominant = row.idxmax()
        dom_pct = row.max() / total * 100
        print(f"    Cluster {c}: {total:5.0f} sentences — dominant: {dominant} ({dom_pct:.0f}%)")

def plot_29_single_vs_multi_comparison(df_full, df_single, single_mask):
    """Side-by-side: single-label vs multi-label class distributions."""
    print("Plot 29: Single-label vs multi-label comparison...")
    theme_counts = df_full[ALL_THEMES].sum(axis=1)
    multi_mask = theme_counts > 1

    fig, axes = plt.subplots(1, 3, figsize=(24, 10))

    # Count per theme: single vs multi
    single_counts = {}
    multi_counts = {}
    for theme in ALL_THEMES:
        single_counts[theme] = df_full.loc[single_mask & (df_full[theme] == 1), theme].sum()
        multi_counts[theme] = df_full.loc[multi_mask & (df_full[theme] == 1), theme].sum()

    x = np.arange(len(ALL_THEMES))
    width = 0.35
    s_vals = [single_counts[t] for t in ALL_THEMES]
    m_vals = [multi_counts[t] for t in ALL_THEMES]

    axes[0].barh(x - width/2, s_vals, width, label='Single-label', color='steelblue')
    axes[0].barh(x + width/2, m_vals, width, label='Multi-label', color='coral')
    axes[0].set_yticks(x)
    axes[0].set_yticklabels(ALL_THEMES, fontsize=9)
    axes[0].set_title('Sentence Counts: Single vs Multi-Label', fontweight='bold')
    axes[0].legend()
    axes[0].invert_yaxis()

    # % that are multi-label per theme
    multi_pct = {}
    for theme in ALL_THEMES:
        total = df_full[theme].sum()
        ml = df_full.loc[multi_mask, theme].sum()
        multi_pct[theme] = ml / total * 100 if total > 0 else 0

    sorted_themes = sorted(multi_pct.items(), key=lambda x: x[1], reverse=True)
    names = [s[0] for s in sorted_themes]
    pcts = [s[1] for s in sorted_themes]
    colors = [CLASS_COLORS[n] for n in names]

    axes[1].barh(names, pcts, color=colors)
    for i, p in enumerate(pcts):
        axes[1].text(p + 0.5, i, f'{p:.1f}%', va='center', fontsize=9)
    axes[1].set_title('% of Each Theme That Is Multi-Label', fontweight='bold')
    axes[1].set_xlabel('% Multi-Label')
    axes[1].invert_yaxis()

    # Pie: overall split
    n_c0 = (theme_counts == 0).sum()
    n_single = (theme_counts == 1).sum()
    n_multi = (theme_counts > 1).sum()
    axes[2].pie([n_c0, n_single, n_multi],
                labels=[f'Class_0\n{n_c0:,}', f'Single\n{n_single:,}', f'Multi\n{n_multi:,}'],
                colors=['#999999', 'steelblue', 'coral'],
                autopct='%1.1f%%', startangle=90, textprops={'fontsize': 12})
    axes[2].set_title('Overall Distribution', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '29_single_vs_multi_comparison.png'), dpi=200)
    plt.close()

def plot_30_separability_summary(df_single, embeddings, centroids):
    """Summary chart: how separable is each class (d-prime from centroid analysis)."""
    print("Plot 30: Separability summary...")
    results = []
    for cls in ALL_CLASSES:
        if cls not in centroids:
            continue
        cent = centroids[cls]
        norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(cent)
        sims = (embeddings @ cent) / (norms + 1e-10)

        mask = df_single['label'] == cls
        m_sims = sims[mask]
        nm_sims = sims[~mask]

        if len(m_sims) > 1 and len(nm_sims) > 1:
            dprime = (np.mean(m_sims) - np.mean(nm_sims)) / (
                np.sqrt((np.std(m_sims)**2 + np.std(nm_sims)**2) / 2) + 1e-10)
            mean_sep = np.mean(m_sims) - np.mean(nm_sims)
        else:
            dprime = 0
            mean_sep = 0

        results.append({
            'class': cls,
            'n_samples': mask.sum(),
            'dprime': dprime,
            'mean_separation': mean_sep,
            'member_mean_sim': np.mean(m_sims) if len(m_sims) > 0 else 0,
            'nonmember_mean_sim': np.mean(nm_sims) if len(nm_sims) > 0 else 0
        })

    res_df = pd.DataFrame(results).sort_values('dprime', ascending=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    colors = [CLASS_COLORS[c] for c in res_df['class']]

    ax1.barh(range(len(res_df)), res_df['dprime'].values, color=colors)
    ax1.set_yticks(range(len(res_df)))
    ax1.set_yticklabels([f"{r['class']} (n={r['n_samples']})" for _, r in res_df.iterrows()], fontsize=10)
    for i, (_, r) in enumerate(res_df.iterrows()):
        ax1.text(r['dprime'] + 0.02, i, f"d'={r['dprime']:.2f}", va='center', fontsize=9)
    ax1.set_title("Separability (d') — Single-Label Centroids\n(Higher = Easier to Classify)",
                  fontweight='bold', fontsize=13)
    ax1.set_xlabel("d' (Cohen's d-like separability)")
    ax1.axvline(1.0, color='green', linestyle='--', alpha=0.5, label="d'=1.0 (good)")
    ax1.axvline(0.5, color='orange', linestyle='--', alpha=0.5, label="d'=0.5 (fair)")
    ax1.legend()

    # Difficulty tier assignment
    tiers = []
    for _, r in res_df.iterrows():
        if r['dprime'] >= 1.0:
            tiers.append(('Easy', 'green'))
        elif r['dprime'] >= 0.5:
            tiers.append(('Moderate', 'orange'))
        elif r['dprime'] >= 0.2:
            tiers.append(('Hard', 'red'))
        else:
            tiers.append(('Very Hard', 'darkred'))

    ax2.barh(range(len(res_df)), res_df['n_samples'].values, color=colors)
    ax2.set_yticks(range(len(res_df)))
    ax2.set_yticklabels([f"{r['class']} [{t[0]}]" for (_, r), t in zip(res_df.iterrows(), tiers)], fontsize=10)
    for i, ((_, r), t) in enumerate(zip(res_df.iterrows(), tiers)):
        ax2.text(r['n_samples'] + 20, i, f"n={r['n_samples']:,}", va='center', fontsize=9,
                 color=t[1], fontweight='bold')
    ax2.set_title('Sample Count by Difficulty Tier\n(Size + Separability = Training Difficulty)',
                  fontweight='bold', fontsize=13)
    ax2.set_xlabel('Single-Label Sentence Count')
    ax2.set_xscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '30_separability_summary.png'), dpi=200)
    plt.close()

    # Print summary table
    print("\n  Separability Summary (Single-Label):")
    print(f"  {'Class':30s} {'N':>6s} {'d-prime':>8s} {'Tier':>10s}")
    print("  " + "-" * 56)
    for _, r in res_df.sort_values('dprime', ascending=False).iterrows():
        if r['dprime'] >= 1.0: tier = 'Easy'
        elif r['dprime'] >= 0.5: tier = 'Moderate'
        elif r['dprime'] >= 0.2: tier = 'Hard'
        else: tier = 'Very Hard'
        print(f"  {r['class']:30s} {r['n_samples']:6.0f} {r['dprime']:8.3f} {tier:>10s}")

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    df_full, df_single, single_mask = load_and_filter()
    embeddings, umap_2d, umap_3d = load_cached_embeddings(single_mask)
    centroids = compute_centroids(df_single, embeddings)

    plot_19_single_umap(df_single, umap_2d)
    plot_20_single_density(df_single, umap_2d)
    plot_21_single_centroids(df_single, umap_2d, embeddings, centroids)
    plot_22_single_vs_all_centroids(df_single, df_full, embeddings,
                                     os.path.join(BASE_DIR, 'v2_embeddings.npy'),
                                     centroids, umap_2d)
    plot_23_single_similarity(df_single, embeddings, centroids)
    plot_24_single_decision_boundaries(df_single, umap_2d)
    plot_25_single_pairwise_similarity(centroids)
    plot_26_single_3d_umap(df_single, umap_3d)
    plot_27_single_kmeans_optimal(embeddings)
    best_k, pca_50 = plot_27_single_kmeans_optimal(embeddings)
    plot_28_single_kmeans_composition(df_single, umap_2d, pca_50, best_k)
    plot_29_single_vs_multi_comparison(df_full, df_single, single_mask)
    plot_30_separability_summary(df_single, embeddings, centroids)

    print(f"\n{'='*60}")
    print(f"All single-label plots saved to: {PLOT_DIR}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
