"""
ALMA Round 2 Comprehensive Data Analysis
- Class_0 treated as a FULL CLASS (12 classes total)
- UMAP 2D + 3D, PCA 2D + 3D
- Optimal K-Means (elbow + silhouette), Hierarchical clustering
- Cluster composition analysis (actual labels in each cluster)
- Theme centroids, per-theme density, decision boundaries
- Similarity distributions, pairwise overlaps, overlap with Class_0
- Outlier flagging with data point IDs
- Deep analysis output
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import normalize
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import cosine as cosine_dist
from scipy.stats import gaussian_kde
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# ─── Configuration ───────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, 'ALMA_processed_master_dataset.csv')
PLOT_DIR = os.path.join(BASE_DIR, 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

ALL_THEMES = ['Aspirational', 'Familial', 'Social', 'Navigational', 'Resistance',
              'Attainment', 'First_Gen', 'Perseverance', 'Filial_Piety',
              'Community_Consciousness', 'Spiritual']

# 12 classes: Class_0 + 11 themes
ALL_CLASSES = ['Class_0'] + ALL_THEMES

# Distinct colors for 12 classes
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

# ─── Step 1: Load data ──────────────────────────────────────
def load_data():
    print("Loading data...")
    df = pd.read_csv(DATA_FILE)
    # Create Class_0 column if not present
    if 'Class_0' not in df.columns:
        df['Class_0'] = (df[ALL_THEMES].sum(axis=1) == 0).astype(int)
    print(f"  Loaded {len(df)} sentences")
    return df

# ─── Step 2: Assign primary label (for single-label analyses) ──
def assign_primary_label(df):
    """Assign each sentence a primary label for visualization.
    For multi-label: pick the rarest theme (most informative).
    """
    theme_counts = df[ALL_THEMES].sum()
    labels = []
    for _, row in df.iterrows():
        active = [t for t in ALL_THEMES if row[t] == 1]
        if not active:
            labels.append('Class_0')
        elif len(active) == 1:
            labels.append(active[0])
        else:
            # Pick rarest theme
            rarest = min(active, key=lambda t: theme_counts[t])
            labels.append(rarest)
    df['primary_label'] = labels
    return df

# ─── Step 3: Embeddings ─────────────────────────────────────
def compute_embeddings(df):
    emb_path = os.path.join(BASE_DIR, 'v2_embeddings.npy')
    if os.path.exists(emb_path):
        print("Loading cached embeddings...")
        embeddings = np.load(emb_path)
        if len(embeddings) == len(df):
            return embeddings
    print("Computing sentence embeddings (this takes a few minutes)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(df['sentence'].tolist(), show_progress_bar=True, batch_size=256)
    np.save(emb_path, embeddings)
    print(f"  Embeddings shape: {embeddings.shape}")
    return embeddings

# ─── Step 4: UMAP 2D + 3D ───────────────────────────────────
def compute_umap(embeddings):
    import umap
    umap2d_path = os.path.join(BASE_DIR, 'v2_umap_2d.npy')
    umap3d_path = os.path.join(BASE_DIR, 'v2_umap_3d.npy')

    if os.path.exists(umap2d_path):
        print("Loading cached UMAP 2D...")
        umap_2d = np.load(umap2d_path)
    else:
        print("Computing UMAP 2D...")
        reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.3, metric='cosine', random_state=42)
        umap_2d = reducer.fit_transform(embeddings)
        np.save(umap2d_path, umap_2d)

    if os.path.exists(umap3d_path):
        print("Loading cached UMAP 3D...")
        umap_3d = np.load(umap3d_path)
    else:
        print("Computing UMAP 3D...")
        reducer3d = umap.UMAP(n_components=3, n_neighbors=30, min_dist=0.3, metric='cosine', random_state=42)
        umap_3d = reducer3d.fit_transform(embeddings)
        np.save(umap3d_path, umap_3d)

    return umap_2d, umap_3d

# ─── Step 5: PCA 2D + 3D ────────────────────────────────────
def compute_pca(embeddings):
    print("Computing PCA...")
    pca_full = PCA(n_components=50, random_state=42)
    pca_50 = pca_full.fit_transform(embeddings)
    explained_var = pca_full.explained_variance_ratio_

    pca_2d = pca_50[:, :2]
    pca_3d = pca_50[:, :3]

    return pca_2d, pca_3d, pca_50, explained_var

# ─── Step 6: Compute centroids (in full embedding space) ─────
def compute_centroids(df, embeddings):
    """Compute centroid for each of the 12 classes in embedding space."""
    centroids = {}
    for cls in ALL_CLASSES:
        if cls == 'Class_0':
            mask = df['Class_0'] == 1
        else:
            mask = df[cls] == 1
        if mask.sum() > 0:
            centroids[cls] = embeddings[mask].mean(axis=0)
    return centroids

# ═══════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════

def plot_01_umap_scatter(df, umap_2d):
    """Full UMAP 2D scatter — all 12 classes."""
    print("Plot 01: UMAP 2D scatter (12 classes)...")
    fig, ax = plt.subplots(figsize=(16, 12))

    # Plot Class_0 first (background)
    mask = df['primary_label'] == 'Class_0'
    ax.scatter(umap_2d[mask, 0], umap_2d[mask, 1], c=CLASS_COLORS['Class_0'],
               s=3, alpha=0.15, label=f'Class_0 ({mask.sum():,})')

    # Then each theme
    for theme in ALL_THEMES:
        mask = df['primary_label'] == theme
        if mask.sum() > 0:
            ax.scatter(umap_2d[mask, 0], umap_2d[mask, 1], c=CLASS_COLORS[theme],
                       s=8, alpha=0.5, label=f'{theme} ({mask.sum():,})')

    ax.set_title('UMAP 2D — All 12 Classes (Class_0 as Full Class)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8, markerscale=3, framealpha=0.9)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '01_umap_2d_all_classes.png'), dpi=200)
    plt.close()

def plot_02_pca_scatter(df, pca_2d):
    """PCA 2D scatter — all 12 classes."""
    print("Plot 02: PCA 2D scatter (12 classes)...")
    fig, ax = plt.subplots(figsize=(16, 12))

    mask = df['primary_label'] == 'Class_0'
    ax.scatter(pca_2d[mask, 0], pca_2d[mask, 1], c=CLASS_COLORS['Class_0'],
               s=3, alpha=0.15, label=f'Class_0 ({mask.sum():,})')

    for theme in ALL_THEMES:
        mask = df['primary_label'] == theme
        if mask.sum() > 0:
            ax.scatter(pca_2d[mask, 0], pca_2d[mask, 1], c=CLASS_COLORS[theme],
                       s=8, alpha=0.5, label=f'{theme} ({mask.sum():,})')

    ax.set_title('PCA 2D — All 12 Classes', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8, markerscale=3, framealpha=0.9)
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '02_pca_2d_all_classes.png'), dpi=200)
    plt.close()

def plot_03_pca_variance(explained_var):
    """PCA explained variance — how many dimensions needed."""
    print("Plot 03: PCA explained variance...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    cumvar = np.cumsum(explained_var)

    ax1.bar(range(1, len(explained_var)+1), explained_var, color='steelblue', alpha=0.7)
    ax1.set_title('Individual Explained Variance by Component', fontweight='bold')
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Explained Variance Ratio')
    ax1.set_xlim(0, 51)

    ax2.plot(range(1, len(cumvar)+1), cumvar, 'b-o', markersize=3)
    ax2.axhline(y=0.80, color='r', linestyle='--', label='80% threshold')
    ax2.axhline(y=0.90, color='orange', linestyle='--', label='90% threshold')
    ax2.axhline(y=0.95, color='green', linestyle='--', label='95% threshold')
    n80 = np.argmax(cumvar >= 0.80) + 1
    n90 = np.argmax(cumvar >= 0.90) + 1
    n95 = np.argmax(cumvar >= 0.95) + 1
    ax2.axvline(x=n80, color='r', linestyle=':', alpha=0.5)
    ax2.axvline(x=n90, color='orange', linestyle=':', alpha=0.5)
    ax2.axvline(x=n95, color='green', linestyle=':', alpha=0.5)
    ax2.set_title(f'Cumulative Variance (80%@{n80}, 90%@{n90}, 95%@{n95} PCs)', fontweight='bold')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Explained Variance')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '03_pca_explained_variance.png'), dpi=200)
    plt.close()
    return n80, n90, n95

def plot_04_centroids(df, umap_2d, embeddings, centroids):
    """Theme centroids on UMAP — all 12 classes including Class_0."""
    print("Plot 04: Theme centroids (12 classes)...")
    import umap as umap_lib

    fig, ax = plt.subplots(figsize=(16, 12))

    # Background scatter
    ax.scatter(umap_2d[:, 0], umap_2d[:, 1], c='lightgray', s=1, alpha=0.1)

    # Project centroids into UMAP space (approximate via nearest neighbor)
    for cls in ALL_CLASSES:
        if cls not in centroids:
            continue
        cent = centroids[cls]
        # Find nearest data point to centroid
        dists = np.linalg.norm(embeddings - cent, axis=1)
        nearest_idx = np.argmin(dists)
        cx, cy = umap_2d[nearest_idx]

        color = CLASS_COLORS[cls]
        # Plot theme members lightly
        if cls == 'Class_0':
            mask = df['Class_0'] == 1
        else:
            mask = df[cls] == 1
        ax.scatter(umap_2d[mask, 0], umap_2d[mask, 1], c=color, s=4, alpha=0.15)

        # Plot centroid
        ax.scatter(cx, cy, c=color, s=300, marker='*', edgecolors='black',
                   linewidths=1.5, zorder=10)
        ax.annotate(cls, (cx, cy), fontsize=8, fontweight='bold',
                    xytext=(5, 5), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.6))

    ax.set_title('UMAP — All 12 Class Centroids (★ = centroid)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '04_centroids_12_classes.png'), dpi=200)
    plt.close()

def plot_05_per_theme_density(df, umap_2d):
    """Per-theme KDE density — all 12 classes in grid."""
    print("Plot 05: Per-theme density (12 classes)...")
    fig, axes = plt.subplots(3, 4, figsize=(24, 18))

    for idx, cls in enumerate(ALL_CLASSES):
        ax = axes[idx // 4][idx % 4]
        if cls == 'Class_0':
            mask = df['Class_0'] == 1
        else:
            mask = df[cls] == 1
        n = mask.sum()

        # Background
        ax.scatter(umap_2d[:, 0], umap_2d[:, 1], c='lightgray', s=1, alpha=0.05)

        if n > 10:
            pts = umap_2d[mask]
            ax.scatter(pts[:, 0], pts[:, 1], c=CLASS_COLORS[cls], s=3, alpha=0.3)

            # KDE contour
            try:
                kde = gaussian_kde(pts.T, bw_method=0.3)
                xmin, xmax = umap_2d[:, 0].min()-1, umap_2d[:, 0].max()+1
                ymin, ymax = umap_2d[:, 1].min()-1, umap_2d[:, 1].max()+1
                xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
                zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
                ax.contour(xx, yy, zz, levels=5, colors=[CLASS_COLORS[cls]], alpha=0.6, linewidths=1)
            except Exception:
                pass

        ax.set_title(f'{cls} (n={n:,})', fontsize=10, fontweight='bold',
                     color=CLASS_COLORS[cls])
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle('Per-Class Density (KDE on UMAP) — All 12 Classes', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(PLOT_DIR, '05_per_class_density.png'), dpi=200)
    plt.close()

def plot_06_similarity_distributions(df, embeddings, centroids):
    """Cosine similarity of each sentence to ALL 12 class centroids."""
    print("Plot 06: Similarity distributions to centroids...")
    fig, axes = plt.subplots(3, 4, figsize=(24, 18))

    for idx, cls in enumerate(ALL_CLASSES):
        ax = axes[idx // 4][idx % 4]
        if cls not in centroids:
            ax.set_title(f'{cls} — No centroid')
            continue

        cent = centroids[cls]
        # Cosine similarity to this centroid for all points
        norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(cent)
        sims = (embeddings @ cent) / (norms + 1e-10)

        # Split: members vs non-members
        if cls == 'Class_0':
            member_mask = df['Class_0'] == 1
        else:
            member_mask = df[cls] == 1

        member_sims = sims[member_mask]
        nonmember_sims = sims[~member_mask]

        ax.hist(nonmember_sims, bins=80, alpha=0.4, color='gray', density=True, label='Non-members')
        ax.hist(member_sims, bins=80, alpha=0.6, color=CLASS_COLORS[cls], density=True, label='Members')
        ax.axvline(np.mean(member_sims), color=CLASS_COLORS[cls], linestyle='--', linewidth=2)
        ax.axvline(np.mean(nonmember_sims), color='gray', linestyle='--', linewidth=2)

        overlap = np.minimum(
            np.histogram(member_sims, bins=100, density=True, range=(0, 1))[0],
            np.histogram(nonmember_sims, bins=100, density=True, range=(0, 1))[0]
        ).sum() / 100
        ax.set_title(f'{cls} (overlap={overlap:.2f})', fontsize=10, fontweight='bold',
                     color=CLASS_COLORS[cls])
        ax.legend(fontsize=7)

    fig.suptitle('Similarity to Each Class Centroid — Members vs Non-Members', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(PLOT_DIR, '06_similarity_distributions.png'), dpi=200)
    plt.close()

def plot_07_decision_boundaries(df, umap_2d):
    """KNN decision boundaries with confusion zones — 12 classes."""
    print("Plot 07: Decision boundaries (KNN k=15)...")
    # Use primary labels
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    labels_enc = le.fit_transform(df['primary_label'].values)

    knn = KNeighborsClassifier(n_neighbors=15, metric='euclidean', weights='distance')
    knn.fit(umap_2d, labels_enc)

    # Create grid
    xmin, xmax = umap_2d[:, 0].min()-1, umap_2d[:, 0].max()+1
    ymin, ymax = umap_2d[:, 1].min()-1, umap_2d[:, 1].max()+1
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 300), np.linspace(ymin, ymax, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Predict probabilities
    probs = knn.predict_proba(grid)
    max_prob = probs.max(axis=1).reshape(xx.shape)
    pred = knn.predict(grid).reshape(xx.shape)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(28, 12))

    # Left: Decision regions
    color_list = [CLASS_COLORS[c] for c in le.classes_]
    cmap = ListedColormap(color_list)
    ax1.contourf(xx, yy, pred, cmap=cmap, alpha=0.2)
    for cls in ALL_CLASSES:
        mask = df['primary_label'] == cls
        if mask.sum() > 0:
            ax1.scatter(umap_2d[mask, 0], umap_2d[mask, 1], c=CLASS_COLORS[cls],
                        s=3, alpha=0.3, label=cls)
    ax1.set_title('KNN Decision Regions (k=15) — 12 Classes', fontweight='bold')
    ax1.legend(fontsize=7, markerscale=3, loc='upper left')

    # Right: Confidence/confusion
    conf = ax2.contourf(xx, yy, max_prob, levels=20, cmap='RdYlGn', alpha=0.8)
    plt.colorbar(conf, ax=ax2, label='KNN Confidence')
    # Highlight confusion zones (confidence < 0.5)
    ax2.contour(xx, yy, max_prob, levels=[0.3, 0.5], colors=['red', 'orange'], linewidths=2)
    ax2.scatter(umap_2d[:, 0], umap_2d[:, 1], c='black', s=1, alpha=0.05)
    ax2.set_title('Confidence Map (red < 0.3, orange < 0.5 = confusion zones)', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '07_decision_boundaries.png'), dpi=200)
    plt.close()
    return knn, le

def plot_08_overlap_with_class0(df, umap_2d):
    """Overlap between each theme and Class_0 in UMAP space."""
    print("Plot 08: Theme vs Class_0 overlap...")
    class0_mask = df['Class_0'] == 1
    class0_pts = umap_2d[class0_mask]

    fig, axes = plt.subplots(3, 4, figsize=(24, 18))
    axes_flat = axes.flatten()

    # First subplot: overall distribution
    ax = axes_flat[0]
    ax.scatter(umap_2d[class0_mask, 0], umap_2d[class0_mask, 1], c='gray', s=2, alpha=0.1, label='Class_0')
    ax.scatter(umap_2d[~class0_mask, 0], umap_2d[~class0_mask, 1], c='red', s=2, alpha=0.1, label='Themed')
    ax.set_title('Class_0 vs All Themes', fontweight='bold', fontsize=10)
    ax.legend(fontsize=7, markerscale=3)
    ax.set_xticks([])
    ax.set_yticks([])

    for idx, theme in enumerate(ALL_THEMES):
        ax = axes_flat[idx + 1]
        theme_mask = df[theme] == 1
        n_theme = theme_mask.sum()

        ax.scatter(class0_pts[:, 0], class0_pts[:, 1], c='gray', s=2, alpha=0.1)
        ax.scatter(umap_2d[theme_mask, 0], umap_2d[theme_mask, 1],
                   c=CLASS_COLORS[theme], s=5, alpha=0.4)

        # KDE overlap estimation
        try:
            if n_theme > 30:
                theme_kde = gaussian_kde(umap_2d[theme_mask].T, bw_method=0.3)
                class0_kde = gaussian_kde(class0_pts.T, bw_method=0.3)
                xmin, xmax = umap_2d[:, 0].min(), umap_2d[:, 0].max()
                ymin, ymax = umap_2d[:, 1].min(), umap_2d[:, 1].max()
                xx, yy = np.mgrid[xmin:xmax:80j, ymin:ymax:80j]
                grid_pts = np.vstack([xx.ravel(), yy.ravel()])
                t_dens = theme_kde(grid_pts).reshape(xx.shape)
                c_dens = class0_kde(grid_pts).reshape(xx.shape)
                overlap = np.minimum(t_dens, c_dens)
                ax.contourf(xx, yy, overlap, levels=5, cmap='Reds', alpha=0.3)
        except Exception:
            pass

        ax.set_title(f'{theme} vs Class_0 (n={n_theme:,})', fontsize=9, fontweight='bold',
                     color=CLASS_COLORS[theme])
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle('Each Theme vs Class_0 Overlap Regions', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(PLOT_DIR, '08_theme_vs_class0_overlap.png'), dpi=200)
    plt.close()

def plot_09_kmeans_optimal_k(pca_50):
    """Find optimal K for K-Means — elbow + silhouette + calinski-harabasz."""
    print("Plot 09: Optimal K analysis (K=2..25)...")
    K_range = range(2, 26)
    inertias = []
    silhouettes = []
    calinski = []
    davies = []

    # Use PCA-50 for speed
    for k in K_range:
        print(f"  K={k}...")
        km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        labels = km.fit_predict(pca_50)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(pca_50, labels, sample_size=5000, random_state=42))
        calinski.append(calinski_harabasz_score(pca_50, labels))
        davies.append(davies_bouldin_score(pca_50, labels))

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    axes[0, 0].plot(K_range, inertias, 'bo-')
    axes[0, 0].set_title('Elbow Method (Inertia)', fontweight='bold')
    axes[0, 0].set_xlabel('K')
    axes[0, 0].set_ylabel('Inertia')

    axes[0, 1].plot(K_range, silhouettes, 'go-')
    axes[0, 1].set_title('Silhouette Score (higher = better)', fontweight='bold')
    axes[0, 1].set_xlabel('K')
    axes[0, 1].set_ylabel('Silhouette')
    best_k_sil = list(K_range)[np.argmax(silhouettes)]
    axes[0, 1].axvline(best_k_sil, color='r', linestyle='--', label=f'Best K={best_k_sil}')
    axes[0, 1].legend()

    axes[1, 0].plot(K_range, calinski, 'ro-')
    axes[1, 0].set_title('Calinski-Harabasz (higher = better)', fontweight='bold')
    axes[1, 0].set_xlabel('K')
    axes[1, 0].set_ylabel('CH Index')

    axes[1, 1].plot(K_range, davies, 'mo-')
    axes[1, 1].set_title('Davies-Bouldin (lower = better)', fontweight='bold')
    axes[1, 1].set_xlabel('K')
    axes[1, 1].set_ylabel('DB Index')
    best_k_db = list(K_range)[np.argmin(davies)]
    axes[1, 1].axvline(best_k_db, color='r', linestyle='--', label=f'Best K={best_k_db}')
    axes[1, 1].legend()

    fig.suptitle('Optimal K Analysis for K-Means Clustering', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(PLOT_DIR, '09_optimal_k_analysis.png'), dpi=200)
    plt.close()

    return best_k_sil, best_k_db, silhouettes, davies

def plot_10_kmeans_clusters(df, umap_2d, pca_50, best_k):
    """K-Means with optimal K — clusters vs actual labels + composition."""
    print(f"Plot 10: K-Means (K={best_k}) clusters vs actual labels...")
    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    cluster_labels = km.fit_predict(pca_50)
    df['kmeans_cluster'] = cluster_labels

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(28, 12))

    # Left: K-Means clusters
    scatter = ax1.scatter(umap_2d[:, 0], umap_2d[:, 1], c=cluster_labels,
                          cmap='tab20', s=3, alpha=0.4)
    ax1.set_title(f'K-Means Clusters (K={best_k})', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, ax=ax1, label='Cluster')

    # Right: Actual labels
    for cls in ALL_CLASSES:
        mask = df['primary_label'] == cls
        if mask.sum() > 0:
            ax2.scatter(umap_2d[mask, 0], umap_2d[mask, 1], c=CLASS_COLORS[cls],
                        s=3, alpha=0.3, label=cls)
    ax2.set_title('Actual Labels (12 Classes)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=7, markerscale=3, loc='upper left')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '10_kmeans_vs_actual.png'), dpi=200)
    plt.close()

    return cluster_labels

def plot_11_cluster_composition(df, cluster_labels, best_k, method='KMeans'):
    """Stacked bar chart: composition of each cluster by actual labels."""
    print(f"Plot 11: {method} cluster composition...")
    fig, axes = plt.subplots(2, 1, figsize=(20, 16))

    # Top: Absolute counts
    comp = pd.DataFrame()
    for cls in ALL_CLASSES:
        if cls == 'Class_0':
            vals = df['Class_0'].values
        else:
            vals = df[cls].values
        comp[cls] = vals
    comp['cluster'] = cluster_labels

    cluster_theme_counts = comp.groupby('cluster')[ALL_CLASSES].sum()

    cluster_theme_counts.plot(kind='bar', stacked=True, ax=axes[0],
                               color=[CLASS_COLORS[c] for c in ALL_CLASSES])
    axes[0].set_title(f'{method} Cluster Composition (Absolute Counts)', fontweight='bold')
    axes[0].set_xlabel('Cluster')
    axes[0].set_ylabel('Sentence Count')
    axes[0].legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc='upper left')

    # Bottom: Percentages
    cluster_sizes = comp.groupby('cluster').size()
    cluster_theme_pct = cluster_theme_counts.div(cluster_sizes, axis=0) * 100

    cluster_theme_pct.plot(kind='bar', stacked=True, ax=axes[1],
                            color=[CLASS_COLORS[c] for c in ALL_CLASSES])
    axes[1].set_title(f'{method} Cluster Composition (Percentage)', fontweight='bold')
    axes[1].set_xlabel('Cluster')
    axes[1].set_ylabel('Percentage')
    axes[1].legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f'11_{method.lower()}_cluster_composition.png'), dpi=200)
    plt.close()

    return cluster_theme_counts, cluster_theme_pct

def plot_12_hierarchical_clustering(df, umap_2d, pca_50):
    """Agglomerative hierarchical clustering — dendrogram + cluster results."""
    print("Plot 12: Hierarchical clustering...")

    # Subsample for dendrogram (too slow on 18K)
    np.random.seed(42)
    sample_idx = np.random.choice(len(pca_50), size=2000, replace=False)
    sample_pca = pca_50[sample_idx]
    sample_labels = df['primary_label'].values[sample_idx]

    # Linkage
    print("  Computing linkage (Ward)...")
    Z = linkage(sample_pca, method='ward', metric='euclidean')

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 16))

    # Dendrogram
    dendrogram(Z, ax=ax1, truncate_mode='lastp', p=30, leaf_font_size=8,
               color_threshold=0.7 * max(Z[:, 2]))
    ax1.set_title('Hierarchical Clustering Dendrogram (Ward, 2K sample)', fontweight='bold')
    ax1.set_ylabel('Distance')

    # Full hierarchical clustering with different k values
    best_ks = [6, 8, 12, 16]
    for i, k in enumerate(best_ks):
        hc = AgglomerativeClustering(n_clusters=k, linkage='ward')
        hc_labels = hc.fit_predict(pca_50)

        if k == 12:
            df['hierarchical_cluster'] = hc_labels

    # Plot K=12 hierarchical
    hc12 = AgglomerativeClustering(n_clusters=12, linkage='ward')
    hc12_labels = hc12.fit_predict(pca_50)
    scatter = ax2.scatter(umap_2d[:, 0], umap_2d[:, 1], c=hc12_labels,
                          cmap='tab20', s=3, alpha=0.4)
    ax2.set_title('Hierarchical Clustering (K=12, Ward) on UMAP', fontweight='bold')
    plt.colorbar(scatter, ax=ax2, label='Cluster')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '12_hierarchical_clustering.png'), dpi=200)
    plt.close()

    return hc12_labels

def plot_13_3d_umap(df, umap_3d):
    """3D UMAP visualization — multiple angles."""
    print("Plot 13: 3D UMAP (multiple angles)...")
    fig = plt.figure(figsize=(24, 18))

    angles = [(30, 45), (30, 135), (60, 45), (10, 90)]
    for i, (elev, azim) in enumerate(angles):
        ax = fig.add_subplot(2, 2, i+1, projection='3d')

        # Plot each class
        for cls in ALL_CLASSES:
            mask = df['primary_label'] == cls
            if mask.sum() > 0:
                alpha = 0.08 if cls == 'Class_0' else 0.4
                size = 2 if cls == 'Class_0' else 5
                ax.scatter(umap_3d[mask, 0], umap_3d[mask, 1], umap_3d[mask, 2],
                           c=CLASS_COLORS[cls], s=size, alpha=alpha, label=cls)

        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f'Angle: elev={elev}, azim={azim}', fontsize=10)
        if i == 0:
            ax.legend(fontsize=6, markerscale=3, loc='upper left')

    fig.suptitle('3D UMAP — All 12 Classes (Multiple Angles)', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(PLOT_DIR, '13_3d_umap_angles.png'), dpi=200)
    plt.close()

def plot_14_3d_pca(df, pca_3d):
    """3D PCA visualization — multiple angles."""
    print("Plot 14: 3D PCA (multiple angles)...")
    fig = plt.figure(figsize=(24, 18))

    angles = [(30, 45), (30, 135), (60, 45), (10, 90)]
    for i, (elev, azim) in enumerate(angles):
        ax = fig.add_subplot(2, 2, i+1, projection='3d')

        for cls in ALL_CLASSES:
            mask = df['primary_label'] == cls
            if mask.sum() > 0:
                alpha = 0.08 if cls == 'Class_0' else 0.4
                size = 2 if cls == 'Class_0' else 5
                ax.scatter(pca_3d[mask, 0], pca_3d[mask, 1], pca_3d[mask, 2],
                           c=CLASS_COLORS[cls], s=size, alpha=alpha, label=cls)

        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f'Angle: elev={elev}, azim={azim}', fontsize=10)
        if i == 0:
            ax.legend(fontsize=6, markerscale=3, loc='upper left')

    fig.suptitle('3D PCA — All 12 Classes (Multiple Angles)', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(PLOT_DIR, '14_3d_pca_angles.png'), dpi=200)
    plt.close()

def plot_15_pairwise_theme_similarity(centroids):
    """Pairwise cosine similarity between all 12 class centroids."""
    print("Plot 15: Pairwise centroid similarity...")
    classes = [c for c in ALL_CLASSES if c in centroids]
    n = len(classes)
    sim_matrix = np.zeros((n, n))
    for i, c1 in enumerate(classes):
        for j, c2 in enumerate(classes):
            v1 = centroids[c1]
            v2 = centroids[c2]
            sim_matrix[i, j] = (v1 @ v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    fig, ax = plt.subplots(figsize=(14, 12))
    mask_tri = np.triu(np.ones_like(sim_matrix, dtype=bool), k=1)
    sns.heatmap(sim_matrix, xticklabels=classes, yticklabels=classes,
                annot=True, fmt='.3f', cmap='RdYlBu_r', ax=ax,
                vmin=0.5, vmax=1.0, square=True)
    ax.set_title('Pairwise Cosine Similarity Between Class Centroids\n(Including Class_0)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '15_pairwise_centroid_similarity.png'), dpi=200)
    plt.close()
    return sim_matrix, classes

def plot_16_correlation_heatmap(df):
    """Correlation between all 12 binary label columns."""
    print("Plot 16: Label correlation heatmap...")
    # Include Class_0
    corr = df[ALL_CLASSES].corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', ax=ax,
                vmin=-1, vmax=1, square=True, linewidths=0.5)
    ax.set_title('Label Correlation Heatmap (12 Classes including Class_0)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '16_label_correlation.png'), dpi=200)
    plt.close()

def plot_17_class_imbalance(df):
    """Class distribution bar chart — all 12."""
    print("Plot 17: Class imbalance...")
    counts = {}
    for cls in ALL_CLASSES:
        counts[cls] = df[cls].sum() if cls != 'Class_0' else df['Class_0'].sum()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    sorted_classes = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    names = [c[0] for c in sorted_classes]
    vals = [c[1] for c in sorted_classes]
    colors = [CLASS_COLORS[n] for n in names]

    ax1.barh(names[::-1], vals[::-1], color=colors[::-1])
    for i, (name, val) in enumerate(zip(names[::-1], vals[::-1])):
        ax1.text(val + 50, i, f'{val:,} ({val/len(df)*100:.1f}%)', va='center', fontsize=9)
    ax1.set_title('Class Distribution (Absolute)', fontweight='bold')
    ax1.set_xlabel('Sentence Count')

    # Log scale
    ax2.barh(names[::-1], vals[::-1], color=colors[::-1])
    ax2.set_xscale('log')
    ax2.set_title('Class Distribution (Log Scale)', fontweight='bold')
    ax2.set_xlabel('Sentence Count (log)')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '17_class_imbalance.png'), dpi=200)
    plt.close()

def plot_18_multi_label_analysis(df):
    """Multi-label distribution: how many themes per sentence."""
    print("Plot 18: Multi-label analysis...")
    theme_counts = df[ALL_THEMES].sum(axis=1)

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Top-left: Distribution of theme count per sentence
    ax = axes[0, 0]
    vc = theme_counts.value_counts().sort_index()
    ax.bar(vc.index, vc.values, color='steelblue')
    for x, y in zip(vc.index, vc.values):
        ax.text(x, y + 50, f'{y:,}\n({y/len(df)*100:.1f}%)', ha='center', fontsize=9)
    ax.set_title('Themes Per Sentence Distribution', fontweight='bold')
    ax.set_xlabel('Number of Themes')
    ax.set_ylabel('Count')

    # Top-right: Theme co-occurrence matrix (Jaccard)
    ax = axes[0, 1]
    jaccard = np.zeros((len(ALL_THEMES), len(ALL_THEMES)))
    for i, t1 in enumerate(ALL_THEMES):
        for j, t2 in enumerate(ALL_THEMES):
            a = set(df.index[df[t1] == 1])
            b = set(df.index[df[t2] == 1])
            if len(a | b) > 0:
                jaccard[i, j] = len(a & b) / len(a | b)
    sns.heatmap(jaccard, xticklabels=ALL_THEMES, yticklabels=ALL_THEMES,
                annot=True, fmt='.2f', cmap='YlOrRd', ax=ax, vmin=0, vmax=0.5)
    ax.set_title('Theme Co-occurrence (Jaccard Similarity)', fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)

    # Bottom-left: Top 15 theme combinations
    ax = axes[1, 0]
    combos = df[ALL_THEMES].apply(lambda r: '+'.join([t for t in ALL_THEMES if r[t] == 1]) or 'Class_0', axis=1)
    top_combos = combos.value_counts().head(15)
    ax.barh(range(len(top_combos)), top_combos.values, color='teal')
    ax.set_yticks(range(len(top_combos)))
    ax.set_yticklabels(top_combos.index, fontsize=8)
    for i, v in enumerate(top_combos.values):
        ax.text(v + 20, i, f'{v:,}', va='center', fontsize=8)
    ax.set_title('Top 15 Theme Combinations', fontweight='bold')
    ax.set_xlabel('Count')
    ax.invert_yaxis()

    # Bottom-right: Single vs multi-label
    ax = axes[1, 1]
    single = (theme_counts == 1).sum()
    multi = (theme_counts > 1).sum()
    zero = (theme_counts == 0).sum()
    ax.pie([zero, single, multi],
           labels=[f'Class_0\n({zero:,})', f'Single-label\n({single:,})', f'Multi-label\n({multi:,})'],
           colors=['#999999', '#4CAF50', '#2196F3'],
           autopct='%1.1f%%', startangle=90, textprops={'fontsize': 11})
    ax.set_title('Label Type Distribution', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '18_multi_label_analysis.png'), dpi=200)
    plt.close()

# ─── Outlier Detection & Flagging ────────────────────────────
def flag_outliers(df, embeddings, centroids, umap_2d, knn, le):
    """Flag suspicious data points for potential removal."""
    print("\nFlagging outliers and suspicious data points...")
    flags = []

    # 1. Class_0 in confusion zones (KNN says should be themed)
    class0_mask = df['Class_0'] == 1
    class0_idx = np.where(class0_mask)[0]

    knn_preds = knn.predict(umap_2d[class0_idx])
    knn_probs = knn.predict_proba(umap_2d[class0_idx])
    for i, idx in enumerate(class0_idx):
        pred_label = le.inverse_transform([knn_preds[i]])[0]
        conf = knn_probs[i].max()
        if pred_label != 'Class_0' and conf > 0.6:
            flags.append({
                'datapoint_index': idx,
                'essay_id': df.iloc[idx]['essay_id'],
                'sentence_id': df.iloc[idx].get('sentence_id', ''),
                'sentence': df.iloc[idx]['sentence'][:100],
                'flag_type': 'Class0_in_themed_zone',
                'target_theme': pred_label,
                'confidence': round(conf, 4),
                'recommendation': 'review_for_removal'
            })

    # 2. Theme outliers (low similarity to own centroid)
    for theme in ALL_THEMES:
        if theme not in centroids:
            continue
        theme_mask = df[theme] == 1
        theme_idx = np.where(theme_mask)[0]
        cent = centroids[theme]

        sims = []
        for idx in theme_idx:
            sim = (embeddings[idx] @ cent) / (np.linalg.norm(embeddings[idx]) * np.linalg.norm(cent) + 1e-10)
            sims.append(sim)
        sims = np.array(sims)

        if len(sims) > 10:
            threshold = np.percentile(sims, 5)  # Bottom 5%
            for i, idx in enumerate(theme_idx):
                if sims[i] < threshold:
                    # Check if closer to Class_0 centroid
                    if 'Class_0' in centroids:
                        sim_c0 = (embeddings[idx] @ centroids['Class_0']) / (
                            np.linalg.norm(embeddings[idx]) * np.linalg.norm(centroids['Class_0']) + 1e-10)
                    else:
                        sim_c0 = 0

                    flags.append({
                        'datapoint_index': idx,
                        'essay_id': df.iloc[idx]['essay_id'],
                        'sentence_id': df.iloc[idx].get('sentence_id', ''),
                        'sentence': df.iloc[idx]['sentence'][:100],
                        'flag_type': 'theme_outlier',
                        'target_theme': theme,
                        'confidence': round(sims[i], 4),
                        'sim_to_class0': round(sim_c0, 4),
                        'threshold': round(threshold, 4),
                        'recommendation': 'review_closer_to_class0' if sim_c0 > sims[i] else 'review_outlier'
                    })

    # 3. High-similarity Class_0 to theme centroids
    for theme in ALL_THEMES:
        if theme not in centroids:
            continue
        cent = centroids[theme]
        for idx in class0_idx:
            sim = (embeddings[idx] @ cent) / (np.linalg.norm(embeddings[idx]) * np.linalg.norm(cent) + 1e-10)
            if sim > np.percentile(
                [(embeddings[j] @ cent) / (np.linalg.norm(embeddings[j]) * np.linalg.norm(cent) + 1e-10)
                 for j in np.where(df[theme] == 1)[0]], 50):  # Above median of actual members
                flags.append({
                    'datapoint_index': idx,
                    'essay_id': df.iloc[idx]['essay_id'],
                    'sentence_id': df.iloc[idx].get('sentence_id', ''),
                    'sentence': df.iloc[idx]['sentence'][:100],
                    'flag_type': 'Class0_high_sim_to_theme',
                    'target_theme': theme,
                    'confidence': round(sim, 4),
                    'recommendation': 'review_possible_misannotation'
                })

    flags_df = pd.DataFrame(flags)
    flags_df.to_csv(os.path.join(BASE_DIR, 'v2_flagged_datapoints.csv'), index=False)
    print(f"  Total flags: {len(flags_df)}")
    if len(flags_df) > 0:
        print(f"  By type: {flags_df['flag_type'].value_counts().to_dict()}")
    return flags_df

# ─── Generate summary report ────────────────────────────────
def generate_summary(df, embeddings, centroids, cluster_labels, hc_labels, sim_matrix,
                     sim_classes, flags_df, best_k_sil, n80, n90, n95):
    """Generate comprehensive analysis summary."""
    print("\nGenerating summary report...")
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("ALMA Round 2 — Comprehensive Data Analysis Summary")
    report_lines.append("=" * 80)

    report_lines.append(f"\nDataset: {len(df)} sentences, 12 classes (11 themes + Class_0)")
    report_lines.append(f"Embedding dimension: {embeddings.shape[1]}")
    report_lines.append(f"PCA: 80% variance at {n80} PCs, 90% at {n90}, 95% at {n95}")
    report_lines.append(f"Best K (silhouette): {best_k_sil}")

    report_lines.append("\n--- Class Distribution ---")
    for cls in ALL_CLASSES:
        n = df[cls].sum() if cls != 'Class_0' else df['Class_0'].sum()
        report_lines.append(f"  {cls:30s}: {n:6d} ({n/len(df)*100:5.1f}%)")

    report_lines.append("\n--- Centroid Similarities (Top Confusable Pairs) ---")
    pairs = []
    for i, c1 in enumerate(sim_classes):
        for j, c2 in enumerate(sim_classes):
            if i < j:
                pairs.append((c1, c2, sim_matrix[i, j]))
    pairs.sort(key=lambda x: x[2], reverse=True)
    for c1, c2, sim in pairs[:15]:
        report_lines.append(f"  {c1:25s} ↔ {c2:25s}: {sim:.4f}")

    report_lines.append(f"\n--- Flagged Data Points ---")
    report_lines.append(f"  Total flags: {len(flags_df)}")
    if len(flags_df) > 0:
        for ftype, count in flags_df['flag_type'].value_counts().items():
            unique = flags_df[flags_df['flag_type'] == ftype]['datapoint_index'].nunique()
            report_lines.append(f"  {ftype}: {count} flags ({unique} unique sentences)")

    report_lines.append("\n--- K-Means Cluster Composition (top themes per cluster) ---")
    for c in range(best_k_sil):
        mask = cluster_labels == c
        n_cluster = mask.sum()
        theme_counts = {}
        for cls in ALL_CLASSES:
            if cls == 'Class_0':
                theme_counts[cls] = df.loc[mask, 'Class_0'].sum()
            else:
                theme_counts[cls] = df.loc[mask, cls].sum()
        top3 = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        top3_str = ', '.join([f'{t}:{n}({n/n_cluster*100:.0f}%)' for t, n in top3])
        report_lines.append(f"  Cluster {c}: {n_cluster:5d} sentences — {top3_str}")

    report_lines.append("\n--- Multi-Label Stats ---")
    theme_counts = df[ALL_THEMES].sum(axis=1)
    report_lines.append(f"  Class_0 (0 themes): {(theme_counts == 0).sum():,}")
    report_lines.append(f"  Single-label (1 theme): {(theme_counts == 1).sum():,}")
    report_lines.append(f"  Multi-label (2+ themes): {(theme_counts > 1).sum():,}")
    report_lines.append(f"  Max themes per sentence: {theme_counts.max()}")

    report = '\n'.join(report_lines)
    with open(os.path.join(BASE_DIR, 'v2_analysis_summary.txt'), 'w') as f:
        f.write(report)
    print(report)
    return report

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    # Load & prepare
    df = load_data()
    df = assign_primary_label(df)
    embeddings = compute_embeddings(df)
    umap_2d, umap_3d = compute_umap(embeddings)
    pca_2d, pca_3d, pca_50, explained_var = compute_pca(embeddings)
    centroids = compute_centroids(df, embeddings)

    # Generate all plots
    plot_01_umap_scatter(df, umap_2d)
    plot_02_pca_scatter(df, pca_2d)
    n80, n90, n95 = plot_03_pca_variance(explained_var)
    plot_04_centroids(df, umap_2d, embeddings, centroids)
    plot_05_per_theme_density(df, umap_2d)
    plot_06_similarity_distributions(df, embeddings, centroids)
    knn, le = plot_07_decision_boundaries(df, umap_2d)
    plot_08_overlap_with_class0(df, umap_2d)
    best_k_sil, best_k_db, silhouettes, davies = plot_09_kmeans_optimal_k(pca_50)
    print(f"\n  Best K (silhouette): {best_k_sil}, Best K (Davies-Bouldin): {best_k_db}")
    cluster_labels = plot_10_kmeans_clusters(df, umap_2d, pca_50, best_k_sil)
    cluster_counts, cluster_pcts = plot_11_cluster_composition(df, cluster_labels, best_k_sil, 'KMeans')
    hc_labels = plot_12_hierarchical_clustering(df, umap_2d, pca_50)
    plot_11_cluster_composition(df, hc_labels, 12, 'Hierarchical')
    plot_13_3d_umap(df, umap_3d)
    plot_14_3d_pca(df, pca_3d)
    sim_matrix, sim_classes = plot_15_pairwise_theme_similarity(centroids)
    plot_16_correlation_heatmap(df)
    plot_17_class_imbalance(df)
    plot_18_multi_label_analysis(df)

    # Outlier flagging
    flags_df = flag_outliers(df, embeddings, centroids, umap_2d, knn, le)

    # Summary report
    generate_summary(df, embeddings, centroids, cluster_labels, hc_labels,
                     sim_matrix, sim_classes, flags_df, best_k_sil, n80, n90, n95)

    print(f"\n{'='*60}")
    print(f"All plots saved to: {PLOT_DIR}")
    print(f"Flagged datapoints: v2_flagged_datapoints.csv")
    print(f"Summary: v2_analysis_summary.txt")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
