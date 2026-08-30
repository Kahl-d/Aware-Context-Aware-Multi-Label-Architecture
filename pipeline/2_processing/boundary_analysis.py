"""
ALMA Round 2 — Boundary Overlap & Class_0-on-Theme Analysis (v2)
Identifies Class_0 sentences sitting on top of themed data and vice versa.
Uses leave-one-out KNN, embedding-space nearest-neighbor distances, and
local theme density — NOT raw margin which over-flags.

Plots 31-36:
  31: Class_0 overlap with themes (where Class_0 sits ON themed points)
  32: Per-theme overlap heatmap (which themes share space with which)
  33: Leave-one-out KNN mismatch analysis (proper cross-validated)
  34: Local density analysis (what % of neighbors are same class?)
  35: Focused removal candidates (conservative, targeted)
  36: Hierarchical clusters vs actual labels (detailed comparison)
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, fcluster
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, 'ALMA_processed_master_dataset.csv')
PLOT_DIR = os.path.join(BASE_DIR, 'plots_boundary')
os.makedirs(PLOT_DIR, exist_ok=True)

ALL_THEMES = ['Aspirational', 'Familial', 'Social', 'Navigational', 'Resistance',
              'Attainment', 'First_Gen', 'Perseverance', 'Filial_Piety',
              'Community_Consciousness', 'Spiritual']
ALL_CLASSES = ['Class_0'] + ALL_THEMES

CLASS_COLORS = {
    'Class_0': '#999999', 'Aspirational': '#e6194b', 'Familial': '#3cb44b',
    'Social': '#4363d8', 'Navigational': '#f58231', 'Resistance': '#911eb4',
    'Attainment': '#42d4f4', 'First_Gen': '#f032e6', 'Perseverance': '#bfef45',
    'Filial_Piety': '#fabed4', 'Community_Consciousness': '#dcbeff', 'Spiritual': '#ffe119'
}


def load_all():
    print("Loading data + cached embeddings...")
    df = pd.read_csv(DATA_FILE)
    if 'Class_0' not in df.columns:
        df['Class_0'] = (df[ALL_THEMES].sum(axis=1) == 0).astype(int)

    # Primary label (rarest theme for multi-label)
    theme_counts = df[ALL_THEMES].sum()
    labels = []
    for _, row in df.iterrows():
        active = [t for t in ALL_THEMES if row[t] == 1]
        if not active:
            labels.append('Class_0')
        elif len(active) == 1:
            labels.append(active[0])
        else:
            labels.append(min(active, key=lambda t: theme_counts[t]))
    df['primary_label'] = labels

    embeddings = np.load(os.path.join(BASE_DIR, 'v2_embeddings.npy'))
    umap_2d = np.load(os.path.join(BASE_DIR, 'v2_umap_2d.npy'))
    print(f"  {len(df)} sentences loaded")
    return df, embeddings, umap_2d


def compute_overlap_metrics(df, embeddings):
    """For each point, find nearest neighbor of DIFFERENT class in embedding space."""
    print("\nComputing cross-class nearest neighbor distances...")

    labels = df['primary_label'].values
    is_class0 = labels == 'Class_0'

    # Build separate NN indices for Class_0 and each theme
    class0_idx = np.where(is_class0)[0]
    themed_idx = np.where(~is_class0)[0]

    # For each Class_0 point: nearest themed neighbor
    if len(class0_idx) > 0 and len(themed_idx) > 0:
        nn_themed = NearestNeighbors(n_neighbors=10, metric='cosine')
        nn_themed.fit(embeddings[themed_idx])
        c0_dists, c0_nn_idx = nn_themed.kneighbors(embeddings[class0_idx])
        # Map back to original indices
        c0_nearest_themed_dist = c0_dists[:, 0]  # cosine distance to nearest themed
        c0_nearest_themed_sim = 1 - c0_nearest_themed_dist
        c0_nearest_themed_label = labels[themed_idx[c0_nn_idx[:, 0]]]
        # Count how many of 10 nearest themed neighbors share the same theme
        c0_nn_themes = np.array([labels[themed_idx[c0_nn_idx[i]]] for i in range(len(class0_idx))])
    else:
        c0_nearest_themed_sim = np.array([])
        c0_nearest_themed_label = np.array([])

    # For each themed point: nearest Class_0 neighbor
    if len(themed_idx) > 0 and len(class0_idx) > 0:
        nn_c0 = NearestNeighbors(n_neighbors=10, metric='cosine')
        nn_c0.fit(embeddings[class0_idx])
        t_dists, t_nn_idx = nn_c0.kneighbors(embeddings[themed_idx])
        t_nearest_c0_sim = 1 - t_dists[:, 0]
    else:
        t_nearest_c0_sim = np.array([])

    return (class0_idx, themed_idx, c0_nearest_themed_sim, c0_nearest_themed_label,
            t_nearest_c0_sim, c0_nn_themes)


def compute_loo_knn(df, embeddings):
    """Leave-one-out KNN: for each point, predict using k=15 neighbors EXCLUDING self."""
    print("Computing leave-one-out KNN predictions...")

    nn = NearestNeighbors(n_neighbors=16, metric='cosine')  # 16 to exclude self
    nn.fit(embeddings)
    dists, indices = nn.kneighbors(embeddings)

    labels = df['primary_label'].values
    loo_preds = []
    loo_confidence = []

    for i in range(len(df)):
        # Exclude self (first neighbor is always self with dist=0)
        neighbors = indices[i, 1:]  # skip index 0 (self)
        neighbor_labels = labels[neighbors]

        # Weighted vote (distance-weighted)
        neighbor_dists = dists[i, 1:]
        weights = 1.0 / (neighbor_dists + 1e-10)

        class_votes = {}
        for lbl, w in zip(neighbor_labels, weights):
            class_votes[lbl] = class_votes.get(lbl, 0) + w

        total_weight = sum(class_votes.values())
        pred = max(class_votes, key=class_votes.get)
        conf = class_votes[pred] / total_weight

        loo_preds.append(pred)
        loo_confidence.append(conf)

    df['loo_pred'] = loo_preds
    df['loo_confidence'] = loo_confidence
    df['loo_mismatch'] = df['loo_pred'] != df['primary_label']

    n_mismatch = df['loo_mismatch'].sum()
    print(f"  LOO-KNN mismatches: {n_mismatch} / {len(df)} ({n_mismatch/len(df)*100:.1f}%)")

    return df


def compute_local_density(df, embeddings):
    """For each point, what % of k=20 nearest neighbors share the same primary label?"""
    print("Computing local same-class density...")

    nn = NearestNeighbors(n_neighbors=21, metric='cosine')
    nn.fit(embeddings)
    _, indices = nn.kneighbors(embeddings)

    labels = df['primary_label'].values
    same_class_pcts = []
    for i in range(len(df)):
        neighbors = indices[i, 1:]  # exclude self
        same = np.sum(labels[neighbors] == labels[i])
        same_class_pcts.append(same / 20 * 100)

    df['local_same_class_pct'] = same_class_pcts
    return df


# ================================================================
# PLOTS
# ================================================================

def plot_31_class0_on_themes(df, umap_2d, c0_idx, c0_nearest_themed_sim, c0_nearest_themed_label):
    """Where Class_0 points sit ON TOP of themed points."""
    print("\nPlot 31: Class_0 overlapping themed regions...")
    fig, axes = plt.subplots(2, 3, figsize=(30, 20))

    # (0,0): All Class_0 colored by similarity to nearest themed neighbor
    themed_mask = df['primary_label'] != 'Class_0'
    axes[0, 0].scatter(umap_2d[themed_mask, 0], umap_2d[themed_mask, 1],
                       c='lightgray', s=2, alpha=0.1, label='Themed (background)')
    sc = axes[0, 0].scatter(umap_2d[c0_idx, 0], umap_2d[c0_idx, 1],
                            c=c0_nearest_themed_sim, cmap='RdYlGn_r',
                            s=6, alpha=0.5, vmin=0.5, vmax=1.0)
    plt.colorbar(sc, ax=axes[0, 0], label='Cosine sim to nearest themed point')
    axes[0, 0].set_title('Class_0 Points — Color = Similarity to Nearest Themed\n'
                         'RED = sitting on top of a themed point', fontweight='bold')

    # (0,1): Histogram of Class_0 nearest-themed-neighbor similarity
    axes[0, 1].hist(c0_nearest_themed_sim, bins=100, color='#999999', edgecolor='black', linewidth=0.3)
    p90 = np.percentile(c0_nearest_themed_sim, 90)
    p95 = np.percentile(c0_nearest_themed_sim, 95)
    p99 = np.percentile(c0_nearest_themed_sim, 99)
    axes[0, 1].axvline(p90, color='orange', linestyle='--', linewidth=2, label=f'P90 = {p90:.3f}')
    axes[0, 1].axvline(p95, color='red', linestyle='--', linewidth=2, label=f'P95 = {p95:.3f}')
    axes[0, 1].axvline(p99, color='darkred', linestyle='--', linewidth=2, label=f'P99 = {p99:.3f}')
    axes[0, 1].set_title(f'Class_0: Distance to Nearest Themed Neighbor\n'
                         f'Mean={np.mean(c0_nearest_themed_sim):.3f}, '
                         f'Median={np.median(c0_nearest_themed_sim):.3f}', fontweight='bold')
    axes[0, 1].set_xlabel('Cosine Similarity to Nearest Themed Point')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].legend()

    # (0,2): Which themes do Class_0 points overlap with? (nearest themed neighbor)
    theme_overlap_counts = pd.Series(c0_nearest_themed_label).value_counts()
    colors = [CLASS_COLORS.get(t, '#333') for t in theme_overlap_counts.index]
    axes[0, 2].barh(range(len(theme_overlap_counts)), theme_overlap_counts.values, color=colors)
    axes[0, 2].set_yticks(range(len(theme_overlap_counts)))
    axes[0, 2].set_yticklabels(theme_overlap_counts.index, fontsize=9)
    for i, v in enumerate(theme_overlap_counts.values):
        pct = v / len(c0_idx) * 100
        axes[0, 2].text(v + 20, i, f'{v} ({pct:.1f}%)', va='center', fontsize=8)
    axes[0, 2].set_title('Class_0 Overlaps With Which Theme?\n(Nearest themed neighbor)',
                         fontweight='bold')
    axes[0, 2].invert_yaxis()

    # (1,0): HIGH-SIM Class_0 points only (>P95) — these are the problem ones
    high_sim_mask = c0_nearest_themed_sim >= p95
    high_sim_idx = c0_idx[high_sim_mask]
    other_c0_idx = c0_idx[~high_sim_mask]

    # Background: themed points colored by class
    for cls in ALL_THEMES:
        mask = df['primary_label'] == cls
        if mask.sum() > 0:
            axes[1, 0].scatter(umap_2d[mask, 0], umap_2d[mask, 1],
                             c=CLASS_COLORS[cls], s=3, alpha=0.15, label=cls)
    # Normal Class_0
    axes[1, 0].scatter(umap_2d[other_c0_idx, 0], umap_2d[other_c0_idx, 1],
                       c='lightgray', s=2, alpha=0.05)
    # High-sim Class_0 in RED
    axes[1, 0].scatter(umap_2d[high_sim_idx, 0], umap_2d[high_sim_idx, 1],
                       c='red', s=15, alpha=0.7, edgecolors='black', linewidth=0.3,
                       label=f'Class_0 ON themes ({len(high_sim_idx)})', zorder=5)
    axes[1, 0].set_title(f'Class_0 Points ON TOP of Themes (P95 sim > {p95:.3f})\n'
                         f'{len(high_sim_idx)} Class_0 points (RED) sitting inside themed regions',
                         fontweight='bold')
    axes[1, 0].legend(fontsize=7, markerscale=2, loc='upper left')

    # (1,1): Which themes these high-sim Class_0 overlap with
    if len(high_sim_idx) > 0:
        high_sim_themes = c0_nearest_themed_label[high_sim_mask]
        hi_counts = pd.Series(high_sim_themes).value_counts()
        colors_hi = [CLASS_COLORS.get(t, '#333') for t in hi_counts.index]
        axes[1, 1].barh(range(len(hi_counts)), hi_counts.values, color=colors_hi)
        axes[1, 1].set_yticks(range(len(hi_counts)))
        axes[1, 1].set_yticklabels(hi_counts.index, fontsize=9)
        for i, v in enumerate(hi_counts.values):
            axes[1, 1].text(v + 1, i, f'{v}', va='center', fontsize=9, fontweight='bold')
        axes[1, 1].set_title(f'Top-{len(high_sim_idx)} Worst Class_0 Overlaps:\n'
                             f'Which Themes Are They On?', fontweight='bold')
        axes[1, 1].invert_yaxis()

    # (1,2): Per-theme: what % of Class_0-nearest-neighbor are high-sim?
    theme_total = pd.Series(c0_nearest_themed_label).value_counts()
    theme_high = pd.Series(c0_nearest_themed_label[high_sim_mask]).value_counts() if high_sim_mask.sum() > 0 else pd.Series(dtype=int)
    theme_pct = {}
    for t in ALL_THEMES:
        total = theme_total.get(t, 0)
        high = theme_high.get(t, 0)
        if total > 0:
            theme_pct[t] = (high / total * 100, high, total)
    sorted_themes = sorted(theme_pct.items(), key=lambda x: x[1][0], reverse=True)
    names = [s[0] for s in sorted_themes]
    pcts = [s[1][0] for s in sorted_themes]
    colors_pct = [CLASS_COLORS[n] for n in names]
    axes[1, 2].barh(range(len(names)), pcts, color=colors_pct)
    axes[1, 2].set_yticks(range(len(names)))
    axes[1, 2].set_yticklabels(names, fontsize=9)
    for i, (p, item) in enumerate(zip(pcts, sorted_themes)):
        _, high, total = item[1]
        axes[1, 2].text(p + 0.3, i, f'{p:.1f}% ({high}/{total})', va='center', fontsize=8)
    axes[1, 2].set_title('Per-Theme: % of Nearby Class_0 That Are High-Overlap\n'
                         '(How contaminated is each theme zone by Class_0?)', fontweight='bold')
    axes[1, 2].invert_yaxis()

    fig.suptitle('Plot 31 — Class_0 Sitting On Top of Themed Data Points', fontsize=18, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '31_class0_on_themes.png'), dpi=200, bbox_inches='tight')
    plt.close()

    return p90, p95, p99


def plot_32_pairwise_overlap(df, embeddings, umap_2d):
    """Per-theme-pair: how much do they overlap in embedding space?"""
    print("Plot 32: Pairwise theme overlap heatmap...")
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))

    # For each class pair: average cosine sim of nearest cross-class neighbor
    overlap_matrix = np.zeros((len(ALL_CLASSES), len(ALL_CLASSES)))
    count_matrix = np.zeros((len(ALL_CLASSES), len(ALL_CLASSES)))

    # Compute centroids
    centroids = {}
    for cls in ALL_CLASSES:
        if cls == 'Class_0':
            mask = df['Class_0'] == 1
        else:
            mask = df[cls] == 1
        if mask.sum() > 0:
            centroids[cls] = embeddings[mask].mean(axis=0)

    # Compute pairwise nearest-neighbor distances between classes
    for i, cls_a in enumerate(ALL_CLASSES):
        mask_a = df['primary_label'] == cls_a
        idx_a = np.where(mask_a)[0]
        if len(idx_a) == 0:
            continue
        for j, cls_b in enumerate(ALL_CLASSES):
            if i == j:
                overlap_matrix[i, j] = 1.0
                continue
            mask_b = df['primary_label'] == cls_b
            idx_b = np.where(mask_b)[0]
            if len(idx_b) == 0:
                continue

            # Sample for speed (max 500 per class)
            sample_a = np.random.choice(idx_a, min(500, len(idx_a)), replace=False)
            sample_b = np.random.choice(idx_b, min(500, len(idx_b)), replace=False)

            # Average cosine similarity between nearest pairs
            sims = cosine_similarity(embeddings[sample_a], embeddings[sample_b])
            max_sims = sims.max(axis=1)  # for each a, max sim to any b
            overlap_matrix[i, j] = max_sims.mean()
            count_matrix[i, j] = len(sample_a)

    # Heatmap
    im = axes[0].imshow(overlap_matrix, cmap='RdYlGn_r', vmin=0.6, vmax=1.0)
    axes[0].set_xticks(range(len(ALL_CLASSES)))
    axes[0].set_xticklabels(ALL_CLASSES, rotation=45, ha='right', fontsize=8)
    axes[0].set_yticks(range(len(ALL_CLASSES)))
    axes[0].set_yticklabels(ALL_CLASSES, fontsize=8)
    for i in range(len(ALL_CLASSES)):
        for j in range(len(ALL_CLASSES)):
            color = 'white' if overlap_matrix[i, j] > 0.85 else 'black'
            axes[0].text(j, i, f'{overlap_matrix[i, j]:.2f}', ha='center', va='center',
                        fontsize=6, color=color)
    plt.colorbar(im, ax=axes[0], label='Avg max cosine sim (nearest cross-class neighbor)')
    axes[0].set_title('Pairwise Class Overlap\n(Higher = more overlapping regions)', fontweight='bold')

    # Right: UMAP showing the most overlapping class pairs
    # Show Class_0 that overlaps with each major theme
    for cls in ALL_THEMES:
        mask = df['primary_label'] == cls
        if mask.sum() > 0:
            axes[1].scatter(umap_2d[mask, 0], umap_2d[mask, 1],
                          c=CLASS_COLORS[cls], s=4, alpha=0.2, label=cls)
    c0_mask = df['primary_label'] == 'Class_0'
    axes[1].scatter(umap_2d[c0_mask, 0], umap_2d[c0_mask, 1],
                   c='#999999', s=3, alpha=0.08, label='Class_0')
    axes[1].set_title('All Classes on UMAP\n(Reference for overlap interpretation)', fontweight='bold')
    axes[1].legend(fontsize=7, markerscale=3, loc='upper left')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '32_pairwise_overlap.png'), dpi=200)
    plt.close()


def plot_33_loo_knn(df, umap_2d):
    """Leave-one-out KNN mismatch analysis."""
    print("Plot 33: LOO-KNN mismatch analysis...")
    fig, axes = plt.subplots(2, 3, figsize=(30, 20))

    mis = df['loo_mismatch']

    # (0,0): All mismatches on UMAP
    axes[0, 0].scatter(umap_2d[~mis, 0], umap_2d[~mis, 1], c='lightgray', s=2, alpha=0.05)
    axes[0, 0].scatter(umap_2d[mis, 0], umap_2d[mis, 1], c='red', s=5, alpha=0.3)
    axes[0, 0].set_title(f'LOO-KNN Mismatches — {mis.sum():,} points ({mis.sum()/len(df)*100:.1f}%)\n'
                         f'RED = neighbors disagree with label', fontweight='bold')

    # (0,1): Per-class mismatch %
    mismatch_by_class = {}
    for cls in ALL_CLASSES:
        mask = df['primary_label'] == cls
        total = mask.sum()
        mis_n = (mask & mis).sum()
        if total > 0:
            mismatch_by_class[cls] = (mis_n / total * 100, mis_n, total)

    sorted_cls = sorted(mismatch_by_class.items(), key=lambda x: x[1][0], reverse=True)
    names = [s[0] for s in sorted_cls]
    pcts = [s[1][0] for s in sorted_cls]
    colors = [CLASS_COLORS[n] for n in names]
    axes[0, 1].barh(range(len(names)), pcts, color=colors)
    axes[0, 1].set_yticks(range(len(names)))
    axes[0, 1].set_yticklabels(names, fontsize=9)
    for i, (p, item) in enumerate(zip(pcts, sorted_cls)):
        _, n, total = item[1]
        axes[0, 1].text(p + 0.3, i, f'{p:.1f}% ({n}/{total})', va='center', fontsize=8)
    axes[0, 1].set_title('LOO-KNN Mismatch % Per Class\n(Higher = less separable)', fontweight='bold')
    axes[0, 1].invert_yaxis()

    # (0,2): Confusion: what do mismatched points get predicted as?
    mis_df = df[mis]
    confusion = pd.crosstab(mis_df['primary_label'], mis_df['loo_pred'])
    # Reorder
    present_actual = [c for c in ALL_CLASSES if c in confusion.index]
    present_pred = [c for c in ALL_CLASSES if c in confusion.columns]
    if len(present_actual) > 0 and len(present_pred) > 0:
        confusion = confusion.reindex(index=present_actual, columns=present_pred, fill_value=0)
        sns.heatmap(confusion, annot=True, fmt='d', cmap='YlOrRd', ax=axes[0, 2],
                   cbar_kws={'label': 'Count'})
        axes[0, 2].set_title('Mismatch Confusion Matrix\n(Actual Label → KNN Predicted)', fontweight='bold')
        axes[0, 2].set_xlabel('LOO-KNN Predicted')
        axes[0, 2].set_ylabel('Actual Label')

    # (1,0): High-confidence mismatches only (conf > 0.5)
    high_conf_mis = df[mis & (df['loo_confidence'] > 0.5)]
    axes[1, 0].scatter(umap_2d[~mis, 0], umap_2d[~mis, 1], c='lightgray', s=2, alpha=0.05)
    for cls in ALL_CLASSES:
        hcm = high_conf_mis[high_conf_mis['primary_label'] == cls]
        if len(hcm) > 0:
            axes[1, 0].scatter(umap_2d[hcm.index, 0], umap_2d[hcm.index, 1],
                             c=CLASS_COLORS[cls], s=10, alpha=0.6,
                             label=f'{cls} ({len(hcm)})')
    axes[1, 0].set_title(f'High-Confidence Mismatches (conf > 0.5)\n'
                         f'{len(high_conf_mis)} points — neighbors STRONGLY disagree',
                         fontweight='bold')
    axes[1, 0].legend(fontsize=7, markerscale=2)

    # (1,1): Class_0 mismatches specifically: what are they predicted as?
    c0_mis = df[(df['primary_label'] == 'Class_0') & mis]
    if len(c0_mis) > 0:
        c0_pred_counts = c0_mis['loo_pred'].value_counts()
        colors_c0 = [CLASS_COLORS.get(t, '#333') for t in c0_pred_counts.index]
        axes[1, 1].barh(range(len(c0_pred_counts)), c0_pred_counts.values, color=colors_c0)
        axes[1, 1].set_yticks(range(len(c0_pred_counts)))
        axes[1, 1].set_yticklabels(c0_pred_counts.index, fontsize=9)
        for i, v in enumerate(c0_pred_counts.values):
            axes[1, 1].text(v + 1, i, f'{v}', va='center', fontsize=9, fontweight='bold')
        axes[1, 1].set_title(f'Class_0 Mismatches: What Are They Predicted As?\n'
                             f'{len(c0_mis)} Class_0 points where neighbors say "themed"',
                             fontweight='bold')
        axes[1, 1].invert_yaxis()

    # (1,2): Themed mismatches: what are they predicted as?
    theme_mis = df[(df['primary_label'] != 'Class_0') & mis]
    pred_c0_count = (theme_mis['loo_pred'] == 'Class_0').sum()
    pred_other_count = (theme_mis['loo_pred'] != 'Class_0').sum()
    if len(theme_mis) > 0:
        axes[1, 2].pie([pred_c0_count, pred_other_count],
                      labels=[f'Predicted Class_0\n({pred_c0_count})',
                              f'Predicted Other Theme\n({pred_other_count})'],
                      colors=['#999999', '#4CAF50'], autopct='%1.1f%%',
                      startangle=90, textprops={'fontsize': 12})
        axes[1, 2].set_title(f'Themed Point Mismatches: Predicted as...\n'
                             f'{len(theme_mis)} themed points where neighbors disagree',
                             fontweight='bold')

    fig.suptitle('Plot 33 — Leave-One-Out KNN Mismatch Analysis (Proper Cross-Validated)',
                 fontsize=18, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '33_loo_knn_analysis.png'), dpi=200, bbox_inches='tight')
    plt.close()


def plot_34_local_density(df, umap_2d):
    """Local density: what % of each point's neighbors are the same class?"""
    print("Plot 34: Local same-class density analysis...")
    fig, axes = plt.subplots(2, 3, figsize=(30, 20))

    # (0,0): UMAP colored by local same-class density
    sc = axes[0, 0].scatter(umap_2d[:, 0], umap_2d[:, 1],
                            c=df['local_same_class_pct'], cmap='RdYlGn',
                            s=3, alpha=0.4, vmin=0, vmax=100)
    plt.colorbar(sc, ax=axes[0, 0], label='% neighbors same class')
    axes[0, 0].set_title('Local Same-Class Density\n'
                         'GREEN = surrounded by same class, RED = isolated', fontweight='bold')

    # (0,1): Histogram of local density by Class_0 vs Themed
    c0_mask = df['primary_label'] == 'Class_0'
    axes[0, 1].hist(df.loc[c0_mask, 'local_same_class_pct'], bins=50, alpha=0.6,
                   color='#999999', density=True, label=f'Class_0 (n={c0_mask.sum()})')
    axes[0, 1].hist(df.loc[~c0_mask, 'local_same_class_pct'], bins=50, alpha=0.6,
                   color='#4CAF50', density=True, label=f'Themed (n=(~c0_mask).sum())')
    axes[0, 1].axvline(20, color='red', linestyle='--', linewidth=2, label='20% threshold')
    axes[0, 1].set_title('Local Same-Class Density Distribution\n'
                         'Left of red line = very isolated points', fontweight='bold')
    axes[0, 1].set_xlabel('% of 20 nearest neighbors with same label')
    axes[0, 1].legend()

    # (0,2): Per-class average local density
    avg_density = {}
    for cls in ALL_CLASSES:
        mask = df['primary_label'] == cls
        if mask.sum() > 0:
            avg_density[cls] = df.loc[mask, 'local_same_class_pct'].mean()
    sorted_cls = sorted(avg_density.items(), key=lambda x: x[1])
    names = [s[0] for s in sorted_cls]
    avgs = [s[1] for s in sorted_cls]
    colors = [CLASS_COLORS[n] for n in names]
    axes[0, 2].barh(range(len(names)), avgs, color=colors)
    axes[0, 2].set_yticks(range(len(names)))
    axes[0, 2].set_yticklabels(names, fontsize=9)
    for i, a in enumerate(avgs):
        axes[0, 2].text(a + 0.3, i, f'{a:.1f}%', va='center', fontsize=9, fontweight='bold')
    axes[0, 2].set_title('Average Local Density Per Class\n'
                         '(Lower = more scattered/mixed with other classes)', fontweight='bold')
    axes[0, 2].set_xlabel('Avg % same-class neighbors')

    # (1,0): Class_0 points with very LOW same-class density (<20%)
    # These are Class_0 points deep in themed territory
    c0_low_density = df[(df['primary_label'] == 'Class_0') & (df['local_same_class_pct'] < 20)]
    axes[1, 0].scatter(umap_2d[:, 0], umap_2d[:, 1], c='lightgray', s=2, alpha=0.03)
    for cls in ALL_THEMES:
        mask = df['primary_label'] == cls
        if mask.sum() > 0:
            axes[1, 0].scatter(umap_2d[mask, 0], umap_2d[mask, 1],
                             c=CLASS_COLORS[cls], s=3, alpha=0.1)
    if len(c0_low_density) > 0:
        axes[1, 0].scatter(umap_2d[c0_low_density.index, 0], umap_2d[c0_low_density.index, 1],
                          c='red', s=15, alpha=0.7, edgecolors='black', linewidth=0.3,
                          label=f'Class_0 isolated ({len(c0_low_density)})', zorder=5)
    axes[1, 0].set_title(f'Class_0 with <20% Same-Class Neighbors\n'
                         f'{len(c0_low_density)} points deep in themed territory (REMOVE)',
                         fontweight='bold')
    axes[1, 0].legend(fontsize=8, markerscale=2)

    # (1,1): Themed points with very LOW same-class density (<10%)
    theme_low_density = df[(df['primary_label'] != 'Class_0') & (df['local_same_class_pct'] < 10)]
    axes[1, 1].scatter(umap_2d[:, 0], umap_2d[:, 1], c='lightgray', s=2, alpha=0.03)
    for cls in ALL_THEMES:
        low = theme_low_density[theme_low_density['primary_label'] == cls]
        if len(low) > 0:
            axes[1, 1].scatter(umap_2d[low.index, 0], umap_2d[low.index, 1],
                             c=CLASS_COLORS[cls], s=15, alpha=0.7, edgecolors='black',
                             linewidth=0.3, label=f'{cls} ({len(low)})')
    axes[1, 1].set_title(f'Themed Points with <10% Same-Class Neighbors\n'
                         f'{len(theme_low_density)} isolated themed points',
                         fontweight='bold')
    axes[1, 1].legend(fontsize=7, markerscale=2)

    # (1,2): Combined: low-density by class
    low_density_all = df[((df['primary_label'] == 'Class_0') & (df['local_same_class_pct'] < 20)) |
                         ((df['primary_label'] != 'Class_0') & (df['local_same_class_pct'] < 10))]
    ld_counts = low_density_all['primary_label'].value_counts()
    total_counts = df['primary_label'].value_counts()
    ld_pcts = {}
    for cls in ALL_CLASSES:
        if cls in ld_counts.index and cls in total_counts.index:
            ld_pcts[cls] = (ld_counts[cls] / total_counts[cls] * 100, ld_counts[cls], total_counts[cls])
        else:
            ld_pcts[cls] = (0, 0, total_counts.get(cls, 0))

    sorted_cls = sorted(ld_pcts.items(), key=lambda x: x[1][0], reverse=True)
    names = [s[0] for s in sorted_cls]
    pcts = [s[1][0] for s in sorted_cls]
    colors = [CLASS_COLORS[n] for n in names]
    axes[1, 2].barh(range(len(names)), pcts, color=colors)
    axes[1, 2].set_yticks(range(len(names)))
    axes[1, 2].set_yticklabels(names, fontsize=9)
    for i, (p, item) in enumerate(zip(pcts, sorted_cls)):
        _, n, total = item[1]
        axes[1, 2].text(p + 0.3, i, f'{p:.1f}% ({n}/{total})', va='center', fontsize=8)
    axes[1, 2].set_title('% Isolated Points Per Class\n'
                         '(Class_0 <20% density, Themed <10% density)', fontweight='bold')
    axes[1, 2].invert_yaxis()

    fig.suptitle('Plot 34 — Local Neighborhood Density Analysis', fontsize=18, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '34_local_density.png'), dpi=200, bbox_inches='tight')
    plt.close()


def plot_35_removal_candidates(df, umap_2d, c0_idx, c0_nearest_themed_sim, p95):
    """Conservative removal candidates combining multiple signals."""
    print("Plot 35: Focused removal candidates...")

    # --- Identify candidates ---
    removal_reasons = {}

    # R1: Class_0 with high similarity to themed neighbors (P95+) AND low same-class density (<30%)
    c0_mask = df['primary_label'] == 'Class_0'
    c0_high_sim = set()
    for i, orig_idx in enumerate(c0_idx):
        if c0_nearest_themed_sim[i] >= p95 and df.loc[orig_idx, 'local_same_class_pct'] < 30:
            c0_high_sim.add(orig_idx)
            removal_reasons[orig_idx] = 'class0_on_theme_P95'

    # R2: Class_0 with LOO-KNN mismatch AND low same-class density (<20%)
    c0_loo_mis = set()
    c0_mis_mask = c0_mask & df['loo_mismatch'] & (df['local_same_class_pct'] < 20)
    for idx in df[c0_mis_mask].index:
        c0_loo_mis.add(idx)
        if idx not in removal_reasons:
            removal_reasons[idx] = 'class0_loo_mismatch_isolated'

    # R3: Class_0 with very low same-class density (<10%) — deep in themed territory
    c0_very_isolated = set()
    c0_iso_mask = c0_mask & (df['local_same_class_pct'] < 10)
    for idx in df[c0_iso_mask].index:
        c0_very_isolated.add(idx)
        if idx not in removal_reasons:
            removal_reasons[idx] = 'class0_deep_in_theme_zone'

    # R4: Themed points with LOO-KNN predicting Class_0 with high confidence AND low density
    theme_to_c0 = set()
    theme_mis_mask = (~c0_mask) & df['loo_mismatch'] & (df['loo_pred'] == 'Class_0') & \
                     (df['loo_confidence'] > 0.4) & (df['local_same_class_pct'] < 15)
    for idx in df[theme_mis_mask].index:
        theme_to_c0.add(idx)
        if idx not in removal_reasons:
            removal_reasons[idx] = 'theme_predicted_class0'

    # R5: Themed points with LOO-KNN predicting OTHER theme with high conf AND very low density
    theme_confused = set()
    theme_conf_mask = (~c0_mask) & df['loo_mismatch'] & (df['loo_pred'] != 'Class_0') & \
                      (df['loo_confidence'] > 0.5) & (df['local_same_class_pct'] < 10)
    for idx in df[theme_conf_mask].index:
        theme_confused.add(idx)
        if idx not in removal_reasons:
            removal_reasons[idx] = 'theme_strong_confusion'

    all_removable_idx = sorted(set().union(c0_high_sim, c0_loo_mis, c0_very_isolated,
                                           theme_to_c0, theme_confused))

    print(f"\n  === CONSERVATIVE REMOVAL CANDIDATES ===")
    print(f"  R1 — Class_0 on theme (P95 sim + <30% density):     {len(c0_high_sim)}")
    print(f"  R2 — Class_0 LOO mismatch + isolated (<20%):        {len(c0_loo_mis)}")
    print(f"  R3 — Class_0 deep in theme zone (<10% density):     {len(c0_very_isolated)}")
    print(f"  R4 — Theme predicted as Class_0 (conf>0.4, <15%):   {len(theme_to_c0)}")
    print(f"  R5 — Theme strongly confused with other (>0.5, <10%): {len(theme_confused)}")
    print(f"  TOTAL UNIQUE: {len(all_removable_idx)}")

    # Count by class
    rem_df = df.loc[all_removable_idx].copy()
    rem_df['removal_reason'] = [removal_reasons[i] for i in all_removable_idx]

    print(f"\n  By class:")
    for cls in ALL_CLASSES:
        n = (rem_df['primary_label'] == cls).sum()
        total = (df['primary_label'] == cls).sum()
        if n > 0:
            print(f"    {cls:30s}: {n:5d} / {total:5d} ({n/total*100:5.1f}%)")

    # --- PLOT ---
    fig, axes = plt.subplots(2, 3, figsize=(30, 20))

    keep_mask = ~df.index.isin(all_removable_idx)

    # (0,0): Removal candidates on UMAP by reason
    reason_colors = {
        'class0_on_theme_P95': '#e6194b',
        'class0_loo_mismatch_isolated': '#f58231',
        'class0_deep_in_theme_zone': '#3cb44b',
        'theme_predicted_class0': '#4363d8',
        'theme_strong_confusion': '#911eb4'
    }
    axes[0, 0].scatter(umap_2d[keep_mask, 0], umap_2d[keep_mask, 1], c='lightgray', s=2, alpha=0.05)
    for reason, color in reason_colors.items():
        r_idx = [i for i in all_removable_idx if removal_reasons[i] == reason]
        if r_idx:
            axes[0, 0].scatter(umap_2d[r_idx, 0], umap_2d[r_idx, 1], c=color, s=10, alpha=0.6,
                             label=f'{reason} ({len(r_idx)})')
    axes[0, 0].set_title(f'Removal Candidates — {len(all_removable_idx)} total\nColored by Reason',
                         fontweight='bold')
    axes[0, 0].legend(fontsize=7, markerscale=2)

    # (0,1): By original class
    axes[0, 1].scatter(umap_2d[keep_mask, 0], umap_2d[keep_mask, 1], c='lightgray', s=2, alpha=0.05)
    for cls in ALL_CLASSES:
        cls_rem = [i for i in all_removable_idx if df.loc[i, 'primary_label'] == cls]
        if cls_rem:
            axes[0, 1].scatter(umap_2d[cls_rem, 0], umap_2d[cls_rem, 1], c=CLASS_COLORS[cls],
                             s=10, alpha=0.6, label=f'{cls} ({len(cls_rem)})')
    axes[0, 1].set_title('Removal Candidates — By Class', fontweight='bold')
    axes[0, 1].legend(fontsize=7, markerscale=2)

    # (0,2): What remains after removal
    for cls in ALL_CLASSES:
        cls_keep = keep_mask & (df['primary_label'] == cls)
        if cls_keep.sum() > 0:
            alpha = 0.1 if cls == 'Class_0' else 0.3
            axes[0, 2].scatter(umap_2d[cls_keep, 0], umap_2d[cls_keep, 1], c=CLASS_COLORS[cls],
                             s=3, alpha=alpha, label=f'{cls} ({cls_keep.sum():,})')
    axes[0, 2].set_title(f'After Removal — {keep_mask.sum():,} remain', fontweight='bold')
    axes[0, 2].legend(fontsize=7, markerscale=3, loc='upper left')

    # (1,0): Before vs After counts
    before = {}
    after = {}
    for cls in ALL_CLASSES:
        before[cls] = (df['primary_label'] == cls).sum()
        removed = (rem_df['primary_label'] == cls).sum()
        after[cls] = before[cls] - removed

    x = np.arange(len(ALL_CLASSES))
    width = 0.35
    axes[1, 0].barh(x - width/2, [before[c] for c in ALL_CLASSES], width,
                    label='Before', color='steelblue', alpha=0.7)
    axes[1, 0].barh(x + width/2, [after[c] for c in ALL_CLASSES], width,
                    label='After', color='coral', alpha=0.7)
    axes[1, 0].set_yticks(x)
    axes[1, 0].set_yticklabels(ALL_CLASSES, fontsize=9)
    axes[1, 0].set_title('Before vs After Removal', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].invert_yaxis()

    # (1,1): % removed per class
    pct_removed = {}
    for cls in ALL_CLASSES:
        if before[cls] > 0:
            pct_removed[cls] = ((before[cls] - after[cls]) / before[cls] * 100,
                                before[cls] - after[cls], before[cls])
    sorted_cls = sorted(pct_removed.items(), key=lambda x: x[1][0], reverse=True)
    names = [s[0] for s in sorted_cls]
    pcts = [s[1][0] for s in sorted_cls]
    colors = [CLASS_COLORS[n] for n in names]
    axes[1, 1].barh(range(len(names)), pcts, color=colors)
    axes[1, 1].set_yticks(range(len(names)))
    axes[1, 1].set_yticklabels(names, fontsize=9)
    for i, (p, item) in enumerate(zip(pcts, sorted_cls)):
        _, n, total = item[1]
        axes[1, 1].text(p + 0.3, i, f'{p:.1f}% ({n}/{total})', va='center', fontsize=8)
    axes[1, 1].set_title('% Removed Per Class', fontweight='bold')
    axes[1, 1].invert_yaxis()

    # (1,2): New balance
    themed_after = sum(v for k, v in after.items() if k != 'Class_0')
    c0_after = after['Class_0']
    total_after = c0_after + themed_after
    ratio = c0_after / themed_after if themed_after > 0 else 0

    labels_pie = [f'Class_0\n{c0_after:,}', f'Themed\n{themed_after:,}']
    sizes = [c0_after, themed_after]
    axes[1, 2].pie(sizes, labels=labels_pie, colors=['#999999', '#4CAF50'],
                  autopct='%1.1f%%', startangle=90, textprops={'fontsize': 12})
    axes[1, 2].set_title(f'After Removal Balance\nClass_0:Themed = {ratio:.2f}:1\n'
                         f'Total: {total_after:,} (was {len(df):,})',
                         fontweight='bold')

    fig.suptitle('Plot 35 — Conservative Removal Candidates (Multi-Signal)',
                 fontsize=18, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '35_removal_candidates.png'), dpi=200, bbox_inches='tight')
    plt.close()

    return rem_df, all_removable_idx, removal_reasons


def plot_36_hierarchical_vs_labels(df, embeddings, umap_2d):
    """Detailed hierarchical cluster vs actual label comparison."""
    print("Plot 36: Hierarchical clusters vs actual labels...")

    # Run hierarchical clustering
    from scipy.cluster.hierarchy import linkage, fcluster
    # Sample for linkage (full data too large for Ward)
    n = len(df)
    if n > 5000:
        sample_idx = np.random.RandomState(42).choice(n, 5000, replace=False)
        sample_emb = embeddings[sample_idx]
    else:
        sample_idx = np.arange(n)
        sample_emb = embeddings

    Z = linkage(sample_emb, method='ward', metric='euclidean')

    # Assign all points to 12 clusters using nearest-centroid from sample
    sample_labels_12 = fcluster(Z, t=12, criterion='maxclust')

    # Compute cluster centroids from sample
    cluster_centroids = {}
    for k in range(1, 13):
        mask = sample_labels_12 == k
        if mask.sum() > 0:
            cluster_centroids[k] = sample_emb[mask].mean(axis=0)

    # Assign ALL points to nearest cluster centroid
    centroid_matrix = np.array([cluster_centroids[k] for k in range(1, 13)])
    sims = cosine_similarity(embeddings, centroid_matrix)
    all_cluster_labels = sims.argmax(axis=1)  # 0-indexed

    df['hier_cluster'] = all_cluster_labels

    fig, axes = plt.subplots(2, 3, figsize=(30, 20))

    # (0,0): Hierarchical clusters on UMAP
    cmap = plt.cm.get_cmap('tab20', 12)
    for k in range(12):
        mask = all_cluster_labels == k
        if mask.sum() > 0:
            axes[0, 0].scatter(umap_2d[mask, 0], umap_2d[mask, 1], c=[cmap(k)],
                             s=3, alpha=0.2, label=f'Cluster {k} ({mask.sum():,})')
    axes[0, 0].set_title('Hierarchical Clusters (K=12) on UMAP', fontweight='bold')
    axes[0, 0].legend(fontsize=6, markerscale=3, loc='upper left', ncol=2)

    # (0,1): Actual labels on UMAP (for comparison)
    for cls in ALL_CLASSES:
        mask = df['primary_label'] == cls
        if mask.sum() > 0:
            alpha = 0.08 if cls == 'Class_0' else 0.2
            axes[0, 1].scatter(umap_2d[mask, 0], umap_2d[mask, 1], c=CLASS_COLORS[cls],
                             s=3, alpha=alpha, label=f'{cls} ({mask.sum():,})')
    axes[0, 1].set_title('Actual Labels on UMAP (for comparison)', fontweight='bold')
    axes[0, 1].legend(fontsize=6, markerscale=3, loc='upper left', ncol=2)

    # (0,2): Composition heatmap — rows = clusters, cols = actual labels
    comp_matrix = np.zeros((12, len(ALL_CLASSES)))
    for k in range(12):
        k_mask = all_cluster_labels == k
        k_total = k_mask.sum()
        if k_total > 0:
            for j, cls in enumerate(ALL_CLASSES):
                if cls == 'Class_0':
                    comp_matrix[k, j] = (df.loc[k_mask, 'Class_0'] == 1).sum() / k_total * 100
                else:
                    comp_matrix[k, j] = (df.loc[k_mask, cls] == 1).sum() / k_total * 100

    sns.heatmap(comp_matrix, annot=True, fmt='.0f', cmap='YlOrRd',
               xticklabels=ALL_CLASSES, yticklabels=[f'Cluster {k}' for k in range(12)],
               ax=axes[1, 0], cbar_kws={'label': '% of cluster'})
    axes[1, 0].set_title('Cluster Composition: % of Each Label in Each Cluster\n'
                         '(Rows = clusters, Cols = labels, sums > 100% due to multi-label)',
                         fontweight='bold')
    axes[1, 0].set_xticklabels(ALL_CLASSES, rotation=45, ha='right', fontsize=7)

    # (1,1): Reverse: for each actual label, which clusters contain most of it?
    reverse_matrix = np.zeros((len(ALL_CLASSES), 12))
    for j, cls in enumerate(ALL_CLASSES):
        if cls == 'Class_0':
            cls_total = (df['Class_0'] == 1).sum()
        else:
            cls_total = (df[cls] == 1).sum()
        if cls_total > 0:
            for k in range(12):
                k_mask = all_cluster_labels == k
                if cls == 'Class_0':
                    reverse_matrix[j, k] = (df.loc[k_mask, 'Class_0'] == 1).sum() / cls_total * 100
                else:
                    reverse_matrix[j, k] = (df.loc[k_mask, cls] == 1).sum() / cls_total * 100

    sns.heatmap(reverse_matrix, annot=True, fmt='.0f', cmap='YlOrRd',
               xticklabels=[f'C{k}' for k in range(12)],
               yticklabels=ALL_CLASSES,
               ax=axes[1, 1], cbar_kws={'label': '% of label'})
    axes[1, 1].set_title('Label Distribution: % of Each Label Going to Each Cluster\n'
                         '(Rows = labels, Cols = clusters. Concentrated = separable)',
                         fontweight='bold')

    # (1,2): Dominant theme per cluster (bar chart)
    dominant_themes = []
    cluster_sizes = []
    class0_pcts = []
    dominant_pcts = []
    for k in range(12):
        k_mask = all_cluster_labels == k
        k_total = k_mask.sum()
        cluster_sizes.append(k_total)

        c0_pct = (df.loc[k_mask, 'Class_0'] == 1).sum() / k_total * 100 if k_total > 0 else 0
        class0_pcts.append(c0_pct)

        best_theme = None
        best_pct = 0
        for cls in ALL_THEMES:
            pct = (df.loc[k_mask, cls] == 1).sum() / k_total * 100 if k_total > 0 else 0
            if pct > best_pct:
                best_pct = pct
                best_theme = cls
        dominant_themes.append(best_theme or 'None')
        dominant_pcts.append(best_pct)

    x = np.arange(12)
    bars1 = axes[1, 2].bar(x - 0.2, class0_pcts, 0.4, label='Class_0 %', color='#999999')
    bars2 = axes[1, 2].bar(x + 0.2, dominant_pcts, 0.4,
                           color=[CLASS_COLORS.get(t, '#333') for t in dominant_themes])
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels([f'C{k}\n({cluster_sizes[k]:,})' for k in range(12)], fontsize=7)
    # Label dominant themes
    for i, (t, p) in enumerate(zip(dominant_themes, dominant_pcts)):
        short = t[:4] if t else '?'
        axes[1, 2].text(i + 0.2, p + 1, f'{short}\n{p:.0f}%', ha='center', fontsize=6, fontweight='bold')
        axes[1, 2].text(i - 0.2, class0_pcts[i] + 1, f'{class0_pcts[i]:.0f}%',
                        ha='center', fontsize=6, color='gray')
    axes[1, 2].set_title('Per Cluster: Class_0 % vs Dominant Theme %\n'
                         '(When theme > Class_0, cluster is theme-dominated)',
                         fontweight='bold')
    axes[1, 2].set_ylabel('Percentage')
    axes[1, 2].legend(['Class_0', 'Dominant Theme'])

    fig.suptitle('Plot 36 — Hierarchical Clustering (K=12) vs Actual Labels',
                 fontsize=18, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '36_hierarchical_vs_labels.png'), dpi=200, bbox_inches='tight')
    plt.close()


def save_removable_csv(df, rem_df, removal_reasons):
    """Save detailed removal candidates with IDs."""
    print("\nSaving removal candidates...")
    cols = ['essay_id']
    if 'sentence_id' in df.columns:
        cols.append('sentence_id')
    cols += ['sentence', 'primary_label', 'loo_pred', 'loo_confidence',
             'local_same_class_pct', 'removal_reason']

    out = rem_df[[c for c in cols if c in rem_df.columns]].copy()
    if 'sentence' in out.columns:
        out['sentence'] = out['sentence'].str[:150]

    out.to_csv(os.path.join(BASE_DIR, 'v2_boundary_removable.csv'), index=True)
    print(f"  Saved {len(out)} candidates to v2_boundary_removable.csv")


def main():
    df, embeddings, umap_2d = load_all()

    # Compute metrics
    (c0_idx, themed_idx, c0_nearest_themed_sim, c0_nearest_themed_label,
     t_nearest_c0_sim, c0_nn_themes) = compute_overlap_metrics(df, embeddings)

    df = compute_loo_knn(df, embeddings)
    df = compute_local_density(df, embeddings)

    # Plots
    p90, p95, p99 = plot_31_class0_on_themes(df, umap_2d, c0_idx,
                                              c0_nearest_themed_sim, c0_nearest_themed_label)
    plot_32_pairwise_overlap(df, embeddings, umap_2d)
    plot_33_loo_knn(df, umap_2d)
    plot_34_local_density(df, umap_2d)
    rem_df, all_removable_idx, removal_reasons = plot_35_removal_candidates(
        df, umap_2d, c0_idx, c0_nearest_themed_sim, p95)
    plot_36_hierarchical_vs_labels(df, embeddings, umap_2d)

    save_removable_csv(df, rem_df, removal_reasons)

    print(f"\n{'='*60}")
    print(f"Boundary plots saved to: {PLOT_DIR}")
    print(f"Removable candidates: v2_boundary_removable.csv")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
