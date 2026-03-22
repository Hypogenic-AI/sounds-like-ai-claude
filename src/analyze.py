"""
Analyze saved activations from Pythia-2.8B.
Loads pre-computed activations and runs all analysis steps efficiently.
"""

import json
import random
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

RESULTS_DIR = Path("/workspaces/sounds-like-ai-claude/results")
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Load saved activations
print("=== Loading saved activations ===")
human_acts = torch.load(RESULTS_DIR / "human_activations.pt", weights_only=True)
ai_acts = torch.load(RESULTS_DIR / "ai_activations.pt", weights_only=True)
print(f"Human: {human_acts.shape}, AI: {ai_acts.shape}")

n_samples = human_acts.shape[0]
n_layers = human_acts.shape[1]
d_model = human_acts.shape[2]
print(f"Samples: {n_samples}, Layers: {n_layers}, d_model: {d_model}")

# Load source info
hc3_path = "/workspaces/sounds-like-ai-claude/datasets/hc3/all.jsonl"
sources = []
data_items = []
with open(hc3_path) as f:
    for line in f:
        item = json.loads(line)
        if item.get("human_answers") and item.get("chatgpt_answers"):
            h = item["human_answers"][0][:512]
            a = item["chatgpt_answers"][0][:512]
            if len(h.split()) > 10 and len(a.split()) > 10:
                sources.append(item.get("source", "unknown"))
                data_items.append(item)
                if len(sources) >= n_samples:
                    break

# Recompute DiffMean directions
print("\n=== Computing DiffMean directions ===")
diff_means = []
for layer in range(n_layers):
    h_mean = human_acts[:, layer, :].mean(dim=0)
    a_mean = ai_acts[:, layer, :].mean(dim=0)
    diff = a_mean - h_mean
    diff = diff / diff.norm()
    diff_means.append(diff)
diff_means = torch.stack(diff_means)

# ── Linear probes (using SGDClassifier for speed) ──
print("\n=== Linear probes per layer ===")
probe_results = []

for layer in range(n_layers):
    X = torch.cat([human_acts[:, layer, :], ai_acts[:, layer, :]], dim=0).numpy()
    y = np.array([0] * n_samples + [1] * n_samples)

    # Use SGDClassifier (much faster than LR for high-dim)
    clf = SGDClassifier(loss='log_loss', max_iter=1000, random_state=SEED, alpha=1e-4)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # Scale features for SGD
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='accuracy')

    # DiffMean 1D projection
    proj = X @ diff_means[layer].numpy()
    threshold = np.median(proj)
    diffmean_acc = ((proj > threshold) == y).mean()

    probe_results.append({
        'layer': layer,
        'lr_accuracy_mean': scores.mean(),
        'lr_accuracy_std': scores.std(),
        'diffmean_1d_accuracy': diffmean_acc,
    })

    print(f"  Layer {layer:2d}: LR={scores.mean():.3f}±{scores.std():.3f}, DiffMean={diffmean_acc:.3f}")

probe_df = pd.DataFrame(probe_results)
probe_df.to_csv(RESULTS_DIR / "probe_results.csv", index=False)

best_layer = int(probe_df.loc[probe_df['lr_accuracy_mean'].idxmax(), 'layer'])
print(f"\nBest layer: {best_layer} (LR acc = {probe_df.loc[best_layer, 'lr_accuracy_mean']:.3f})")

# ── PCA visualization at best layer ──
print("\n=== PCA visualization ===")
X_best = torch.cat([human_acts[:, best_layer, :], ai_acts[:, best_layer, :]], dim=0).numpy()
y_best = np.array([0] * n_samples + [1] * n_samples)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_best)

fig, ax = plt.subplots(figsize=(8, 6))
for label, color, name in [(0, 'blue', 'Human'), (1, 'red', 'AI (ChatGPT)')]:
    mask = y_best == label
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, alpha=0.4, s=20, label=name)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
ax.set_title(f"PCA of Pythia-2.8B Residual Stream at Layer {best_layer}\nHuman vs AI-Generated Text")
ax.legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / "pca_best_layer.png", dpi=150)
plt.close()
print("PCA plot saved.")

# ── Also do PCA along DiffMean direction ──
# Project onto DiffMean direction and a perpendicular direction
dm_dir = diff_means[best_layer].numpy()
proj_dm = X_best @ dm_dir  # projection onto DiffMean
# Get perpendicular component via PCA on residual
X_residual = X_best - np.outer(proj_dm, dm_dir)
pca_resid = PCA(n_components=1)
proj_perp = pca_resid.fit_transform(X_residual).flatten()

fig, ax = plt.subplots(figsize=(8, 6))
for label, color, name in [(0, 'blue', 'Human'), (1, 'red', 'AI (ChatGPT)')]:
    mask = y_best == label
    ax.scatter(proj_dm[mask], proj_perp[mask], c=color, alpha=0.4, s=20, label=name)
ax.set_xlabel("Projection onto DiffMean Direction")
ax.set_ylabel("Projection onto Top Perpendicular PC")
ax.set_title(f"Projection onto 'AI Style' Direction (Layer {best_layer})")
ax.legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / "diffmean_projection.png", dpi=150)
plt.close()

# ── 1D projection histogram ──
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(proj_dm[y_best == 0], bins=40, alpha=0.6, label='Human', color='blue', density=True)
ax.hist(proj_dm[y_best == 1], bins=40, alpha=0.6, label='AI (ChatGPT)', color='red', density=True)
ax.set_xlabel("Projection onto DiffMean Direction")
ax.set_ylabel("Density")
ax.set_title(f"Distribution of Projections onto 'AI Style' Direction\n(Pythia-2.8B, Layer {best_layer})")
ax.legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / "projection_histogram.png", dpi=150)
plt.close()

# ── Layer-wise accuracy plot ──
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(probe_df['layer'], probe_df['lr_accuracy_mean'], 'b-o', markersize=4, label='Linear Classifier (5-fold CV)')
ax.fill_between(probe_df['layer'],
                probe_df['lr_accuracy_mean'] - probe_df['lr_accuracy_std'],
                probe_df['lr_accuracy_mean'] + probe_df['lr_accuracy_std'],
                alpha=0.2)
ax.plot(probe_df['layer'], probe_df['diffmean_1d_accuracy'], 'r-s', markersize=4, label='DiffMean 1D Projection')
ax.axhline(y=0.5, color='gray', linestyle='--', label='Chance')
ax.set_xlabel("Layer")
ax.set_ylabel("Accuracy")
ax.set_title("Human vs AI Text Classification Accuracy by Layer\n(Pythia-2.8B)")
ax.legend()
ax.set_ylim(0.4, 1.05)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "layer_accuracy.png", dpi=150)
plt.close()
print("Layer accuracy plot saved.")

# ── Direction cosine similarity ──
print("\n=== Direction cosine similarity ===")
cos_sim_matrix = torch.zeros(n_layers, n_layers)
for i in range(n_layers):
    for j in range(n_layers):
        cos_sim_matrix[i, j] = torch.dot(diff_means[i], diff_means[j]).item()

fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(cos_sim_matrix.numpy(), cmap='RdBu_r', center=0, vmin=-1, vmax=1, ax=ax)
ax.set_xlabel("Layer")
ax.set_ylabel("Layer")
ax.set_title("Cosine Similarity of 'AI Style' Directions Across Layers\n(Pythia-2.8B)")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "direction_cosine_similarity.png", dpi=150)
plt.close()

# ── Cross-domain generalization ──
print("\n=== Cross-domain generalization ===")
source_arr = np.array(sources[:n_samples])
unique_sources = list(set(sources[:n_samples]))
print(f"Domains: {unique_sources}")

cross_domain_results = []
for train_src in unique_sources:
    for test_src in unique_sources:
        if train_src == test_src:
            continue
        train_mask = source_arr == train_src
        test_mask = source_arr == test_src
        if train_mask.sum() < 10 or test_mask.sum() < 10:
            continue

        # Use DiffMean direction for simplicity and speed
        X_train = torch.cat([
            human_acts[train_mask, best_layer, :],
            ai_acts[train_mask, best_layer, :]
        ], dim=0).numpy()
        y_train = np.array([0] * train_mask.sum() + [1] * train_mask.sum())

        X_test = torch.cat([
            human_acts[test_mask, best_layer, :],
            ai_acts[test_mask, best_layer, :]
        ], dim=0).numpy()
        y_test = np.array([0] * test_mask.sum() + [1] * test_mask.sum())

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = SGDClassifier(loss='log_loss', max_iter=1000, random_state=SEED, alpha=1e-4)
        clf.fit(X_train_s, y_train)
        acc = clf.score(X_test_s, y_test)

        cross_domain_results.append({
            'train_domain': train_src,
            'test_domain': test_src,
            'accuracy': acc,
            'n_train': len(y_train),
            'n_test': len(y_test),
        })

cross_df = pd.DataFrame(cross_domain_results)
cross_df.to_csv(RESULTS_DIR / "cross_domain_results.csv", index=False)
print(cross_df.to_string())

if len(cross_df) > 0:
    pivot = cross_df.pivot(index='train_domain', columns='test_domain', values='accuracy')
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='YlOrRd', vmin=0.5, vmax=1.0, ax=ax)
    ax.set_title(f"Cross-Domain Generalization (Pythia-2.8B, Layer {best_layer})")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "cross_domain_heatmap.png", dpi=150)
    plt.close()

# ── Random direction baseline ──
print("\n=== Random direction baseline ===")
n_random = 500
random_accs = []
for _ in range(n_random):
    rand_dir = torch.randn(d_model)
    rand_dir = rand_dir / rand_dir.norm()
    X_best_t = torch.cat([human_acts[:, best_layer, :], ai_acts[:, best_layer, :]], dim=0)
    proj = (X_best_t @ rand_dir).numpy()
    threshold = np.median(proj)
    acc = ((proj > threshold) == y_best).mean()
    random_accs.append(acc)

random_accs = np.array(random_accs)
actual_diffmean_acc = probe_df.loc[best_layer, 'diffmean_1d_accuracy']
z_score = (actual_diffmean_acc - random_accs.mean()) / random_accs.std()
p_value = (random_accs >= actual_diffmean_acc).mean()

print(f"Random direction acc: {random_accs.mean():.3f} ± {random_accs.std():.3f}")
print(f"DiffMean direction acc: {actual_diffmean_acc:.3f}")
print(f"Z-score: {z_score:.1f}")
print(f"Empirical p-value: {p_value}")

# Histogram of random vs actual
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(random_accs, bins=30, alpha=0.7, label='Random directions')
ax.axvline(actual_diffmean_acc, color='red', linewidth=2, linestyle='--',
           label=f'DiffMean (acc={actual_diffmean_acc:.3f})')
ax.set_xlabel("Accuracy")
ax.set_ylabel("Count")
ax.set_title(f"DiffMean Direction vs Random Directions (n={n_random})\nZ-score = {z_score:.1f}")
ax.legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / "random_baseline.png", dpi=150)
plt.close()

# ── Multi-layer analysis: which layers matter? ──
print("\n=== Multi-layer direction analysis ===")
# Norm of the raw (unnormalized) difference-in-means at each layer
diff_norms = []
for layer in range(n_layers):
    h_mean = human_acts[:, layer, :].mean(dim=0)
    a_mean = ai_acts[:, layer, :].mean(dim=0)
    diff = a_mean - h_mean
    diff_norms.append(diff.norm().item())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(range(n_layers), diff_norms, 'g-o', markersize=4)
axes[0].set_xlabel("Layer")
axes[0].set_ylabel("L2 Norm")
axes[0].set_title("L2 Norm of DiffMean Vector by Layer")

# Variance explained by DiffMean direction at each layer
var_explained = []
for layer in range(n_layers):
    X_l = torch.cat([human_acts[:, layer, :], ai_acts[:, layer, :]], dim=0)
    total_var = X_l.var(dim=0).sum().item()
    proj_l = (X_l @ diff_means[layer]).numpy()
    proj_var = proj_l.var()
    var_explained.append(proj_var / total_var)

axes[1].plot(range(n_layers), var_explained, 'm-o', markersize=4)
axes[1].set_xlabel("Layer")
axes[1].set_ylabel("Fraction of Variance Explained")
axes[1].set_title("Variance Explained by DiffMean Direction")

plt.tight_layout()
plt.savefig(PLOTS_DIR / "layer_direction_analysis.png", dpi=150)
plt.close()

# ── Save summary ──
summary = {
    'model': 'pythia-2.8b',
    'n_layers': n_layers,
    'd_model': d_model,
    'n_samples': n_samples,
    'best_layer': best_layer,
    'best_lr_accuracy': float(probe_df.loc[best_layer, 'lr_accuracy_mean']),
    'best_lr_accuracy_std': float(probe_df.loc[best_layer, 'lr_accuracy_std']),
    'best_diffmean_accuracy': float(actual_diffmean_acc),
    'random_baseline_accuracy': float(random_accs.mean()),
    'random_baseline_std': float(random_accs.std()),
    'z_score_vs_random': float(z_score),
    'p_value_vs_random': float(p_value),
    'seed': SEED,
}

with open(RESULTS_DIR / "summary.json", 'w') as f:
    json.dump(summary, f, indent=2)

print("\n" + "="*60)
print("ANALYSIS COMPLETE (Pythia-2.8B)")
print("="*60)
for k, v in summary.items():
    print(f"  {k}: {v}")
print(f"\nPlots saved to: {PLOTS_DIR}")
