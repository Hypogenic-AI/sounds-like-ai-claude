"""
Fast analysis of Qwen2.5-3B-Instruct saved activations.
Uses SGDClassifier for speed.
"""

import json
import random
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import SGDClassifier
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

RESULTS_DIR = Path("/workspaces/sounds-like-ai-claude/results/qwen")
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

print("=== Loading Qwen activations ===")
human_acts = torch.load(RESULTS_DIR / "human_activations.pt", weights_only=True)
ai_acts = torch.load(RESULTS_DIR / "ai_activations.pt", weights_only=True)
print(f"Human: {human_acts.shape}, AI: {ai_acts.shape}")

n_samples = human_acts.shape[0]
n_layers = human_acts.shape[1]
d_model = human_acts.shape[2]

# Load sources
hc3_path = "/workspaces/sounds-like-ai-claude/datasets/hc3/all.jsonl"
data = []
with open(hc3_path) as f:
    for line in f:
        item = json.loads(line)
        if item.get("human_answers") and item.get("chatgpt_answers"):
            data.append(item)
random.shuffle(data)

sources = []
for item in data[:400]:
    h = item["human_answers"][0][:512]
    a = item["chatgpt_answers"][0][:512]
    if len(h.split()) > 10 and len(a.split()) > 10:
        sources.append(item.get("source", "unknown"))
        if len(sources) >= n_samples:
            break

# DiffMean directions
print("\n=== Computing DiffMean directions ===")
diff_means = []
for layer in range(n_layers):
    h_mean = human_acts[:, layer, :].mean(dim=0)
    a_mean = ai_acts[:, layer, :].mean(dim=0)
    diff = a_mean - h_mean
    diff = diff / diff.norm()
    diff_means.append(diff)
diff_means = torch.stack(diff_means)

# Linear probes
print("\n=== Linear probes per layer ===")
probe_results = []
for layer in range(n_layers):
    X = torch.cat([human_acts[:, layer, :], ai_acts[:, layer, :]], dim=0).numpy()
    y = np.array([0] * n_samples + [1] * n_samples)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = SGDClassifier(loss='log_loss', max_iter=1000, random_state=SEED, alpha=1e-4)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='accuracy')

    # DiffMean 1D
    proj = X @ diff_means[layer].numpy()
    threshold = np.median(proj)
    diffmean_acc = ((proj > threshold) == y).mean()

    # LR vs DiffMean cosine similarity
    clf.fit(X_scaled, y)
    lr_dir = clf.coef_[0] / np.linalg.norm(clf.coef_[0])
    # Need to account for scaling
    dm_scaled = scaler.transform(diff_means[layer].numpy().reshape(1, -1)).flatten()
    dm_scaled = dm_scaled / np.linalg.norm(dm_scaled)
    cos_sim = np.dot(lr_dir, dm_scaled)

    probe_results.append({
        'layer': layer,
        'lr_accuracy_mean': scores.mean(),
        'lr_accuracy_std': scores.std(),
        'diffmean_1d_accuracy': diffmean_acc,
        'lr_diffmean_cosine': cos_sim,
    })
    print(f"  Layer {layer:2d}: LR={scores.mean():.3f}±{scores.std():.3f}, "
          f"DiffMean={diffmean_acc:.3f}, cos={cos_sim:.3f}")

probe_df = pd.DataFrame(probe_results)
probe_df.to_csv(RESULTS_DIR / "probe_results.csv", index=False)

best_layer = int(probe_df.loc[probe_df['lr_accuracy_mean'].idxmax(), 'layer'])
print(f"\nBest layer: {best_layer} (LR acc = {probe_df.loc[best_layer, 'lr_accuracy_mean']:.3f})")

# PCA
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
ax.set_title(f"PCA of Qwen2.5-3B-Instruct Residual Stream (Layer {best_layer})")
ax.legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / "pca_best_layer.png", dpi=150)
plt.close()

# Projection histogram
dm_dir = diff_means[best_layer].numpy()
proj_dm = X_best @ dm_dir

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(proj_dm[y_best == 0], bins=40, alpha=0.6, label='Human', color='blue', density=True)
ax.hist(proj_dm[y_best == 1], bins=40, alpha=0.6, label='AI (ChatGPT)', color='red', density=True)
ax.set_xlabel("Projection onto DiffMean Direction")
ax.set_ylabel("Density")
ax.set_title(f"Projection onto 'AI Style' Direction\n(Qwen2.5-3B-Instruct, Layer {best_layer})")
ax.legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / "projection_histogram.png", dpi=150)
plt.close()

# Layer accuracy plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(probe_df['layer'], probe_df['lr_accuracy_mean'], 'b-o', markersize=4, label='Linear Classifier')
axes[0].fill_between(probe_df['layer'],
                     probe_df['lr_accuracy_mean'] - probe_df['lr_accuracy_std'],
                     probe_df['lr_accuracy_mean'] + probe_df['lr_accuracy_std'], alpha=0.2)
axes[0].plot(probe_df['layer'], probe_df['diffmean_1d_accuracy'], 'r-s', markersize=4, label='DiffMean 1D')
axes[0].axhline(y=0.5, color='gray', linestyle='--', label='Chance')
axes[0].set_xlabel("Layer")
axes[0].set_ylabel("Accuracy")
axes[0].set_title("Classification Accuracy by Layer (Qwen2.5-3B-Instruct)")
axes[0].legend()
axes[0].set_ylim(0.4, 1.05)

axes[1].plot(probe_df['layer'], probe_df['lr_diffmean_cosine'], 'g-^', markersize=4)
axes[1].set_xlabel("Layer")
axes[1].set_ylabel("Cosine Similarity")
axes[1].set_title("LR vs DiffMean Direction Alignment")
axes[1].axhline(y=0, color='gray', linestyle='--')
plt.tight_layout()
plt.savefig(PLOTS_DIR / "layer_analysis.png", dpi=150)
plt.close()

# Direction cosine similarity matrix
cos_sim_matrix = torch.zeros(n_layers, n_layers)
for i in range(n_layers):
    for j in range(n_layers):
        cos_sim_matrix[i, j] = torch.dot(diff_means[i], diff_means[j]).item()

fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(cos_sim_matrix.numpy(), cmap='RdBu_r', center=0, vmin=-1, vmax=1, ax=ax)
ax.set_xlabel("Layer")
ax.set_ylabel("Layer")
ax.set_title("Direction Cosine Similarity (Qwen2.5-3B-Instruct)")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "direction_cosine_similarity.png", dpi=150)
plt.close()

# Cross-domain
print("\n=== Cross-domain generalization ===")
source_arr = np.array(sources[:n_samples])
unique_sources = sorted(set(sources[:n_samples]))
cross_results = []
for train_src in unique_sources:
    for test_src in unique_sources:
        if train_src == test_src:
            continue
        train_mask = source_arr == train_src
        test_mask = source_arr == test_src
        if train_mask.sum() < 5 or test_mask.sum() < 5:
            continue

        X_train = torch.cat([human_acts[train_mask, best_layer, :],
                            ai_acts[train_mask, best_layer, :]], dim=0).numpy()
        y_train = np.array([0] * train_mask.sum() + [1] * train_mask.sum())
        X_test = torch.cat([human_acts[test_mask, best_layer, :],
                           ai_acts[test_mask, best_layer, :]], dim=0).numpy()
        y_test = np.array([0] * test_mask.sum() + [1] * test_mask.sum())

        scaler = StandardScaler()
        clf = SGDClassifier(loss='log_loss', max_iter=1000, random_state=SEED, alpha=1e-4)
        clf.fit(scaler.fit_transform(X_train), y_train)
        acc = clf.score(scaler.transform(X_test), y_test)
        cross_results.append({
            'train_domain': train_src, 'test_domain': test_src,
            'accuracy': acc, 'n_train': len(y_train), 'n_test': len(y_test),
        })

cross_df = pd.DataFrame(cross_results)
cross_df.to_csv(RESULTS_DIR / "cross_domain_results.csv", index=False)
print(cross_df.to_string())
if len(cross_df) > 0:
    print(f"\nMean cross-domain accuracy: {cross_df['accuracy'].mean():.3f}")

    pivot = cross_df.pivot(index='train_domain', columns='test_domain', values='accuracy')
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='YlOrRd', vmin=0.5, vmax=1.0, ax=ax)
    ax.set_title(f"Cross-Domain Generalization (Qwen2.5-3B-Instruct, Layer {best_layer})")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "cross_domain_heatmap.png", dpi=150)
    plt.close()

# Random direction baseline
print("\n=== Random direction baseline ===")
random_accs = []
for _ in range(500):
    rand_dir = torch.randn(d_model)
    rand_dir /= rand_dir.norm()
    X_t = torch.cat([human_acts[:, best_layer, :], ai_acts[:, best_layer, :]], dim=0)
    proj = (X_t @ rand_dir).numpy()
    acc = ((proj > np.median(proj)) == y_best).mean()
    random_accs.append(acc)

random_accs = np.array(random_accs)
actual_acc = probe_df.loc[best_layer, 'diffmean_1d_accuracy']
z = (actual_acc - random_accs.mean()) / random_accs.std()
p = (random_accs >= actual_acc).mean()

print(f"Random: {random_accs.mean():.3f}±{random_accs.std():.3f}")
print(f"DiffMean: {actual_acc:.3f}")
print(f"Z-score: {z:.1f}, p-value: {p}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(random_accs, bins=30, alpha=0.7, label='Random directions')
ax.axvline(actual_acc, color='red', linewidth=2, linestyle='--',
           label=f'DiffMean (acc={actual_acc:.3f})')
ax.set_xlabel("Accuracy")
ax.set_title(f"DiffMean vs Random (n=500), Z={z:.1f}")
ax.legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / "random_baseline.png", dpi=150)
plt.close()

# Summary
summary = {
    'model': 'Qwen/Qwen2.5-3B-Instruct',
    'n_layers': n_layers,
    'd_model': d_model,
    'n_samples': n_samples,
    'best_layer': best_layer,
    'best_lr_accuracy': float(probe_df.loc[best_layer, 'lr_accuracy_mean']),
    'best_lr_accuracy_std': float(probe_df.loc[best_layer, 'lr_accuracy_std']),
    'best_diffmean_accuracy': float(actual_acc),
    'random_baseline_accuracy': float(random_accs.mean()),
    'random_baseline_std': float(random_accs.std()),
    'z_score_vs_random': float(z),
    'p_value_vs_random': float(p),
    'mean_cross_domain_accuracy': float(cross_df['accuracy'].mean()) if len(cross_df) > 0 else None,
    'seed': SEED,
}
with open(RESULTS_DIR / "summary.json", 'w') as f:
    json.dump(summary, f, indent=2)

print("\n" + "="*60)
print("ANALYSIS COMPLETE (Qwen2.5-3B-Instruct)")
print("="*60)
for k, v in summary.items():
    print(f"  {k}: {v}")
