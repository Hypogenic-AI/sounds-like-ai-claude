"""Create combined comparison figures for the final report."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

RESULTS = Path("/workspaces/sounds-like-ai-claude/results")
PLOTS = RESULTS / "plots"

# Load probe results for both models
pythia_probes = pd.read_csv(RESULTS / "probe_results.csv")
qwen_probes = pd.read_csv(RESULTS / "qwen" / "probe_results.csv")

with open(RESULTS / "summary.json") as f:
    pythia_summary = json.load(f)
with open(RESULTS / "qwen" / "summary.json") as f:
    qwen_summary = json.load(f)

# ── Combined layer accuracy comparison ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(pythia_probes['layer'], pythia_probes['lr_accuracy_mean'], 'b-o',
             markersize=4, label='Linear Classifier')
axes[0].fill_between(pythia_probes['layer'],
                     pythia_probes['lr_accuracy_mean'] - pythia_probes['lr_accuracy_std'],
                     pythia_probes['lr_accuracy_mean'] + pythia_probes['lr_accuracy_std'], alpha=0.2)
axes[0].plot(pythia_probes['layer'], pythia_probes['diffmean_1d_accuracy'], 'r-s',
             markersize=4, label='DiffMean 1D')
axes[0].axhline(y=0.5, color='gray', linestyle='--', label='Chance')
axes[0].set_xlabel("Layer")
axes[0].set_ylabel("Accuracy")
axes[0].set_title("Pythia-2.8B (Base Model)")
axes[0].legend()
axes[0].set_ylim(0.4, 1.05)

axes[1].plot(qwen_probes['layer'], qwen_probes['lr_accuracy_mean'], 'b-o',
             markersize=4, label='Linear Classifier')
axes[1].fill_between(qwen_probes['layer'],
                     qwen_probes['lr_accuracy_mean'] - qwen_probes['lr_accuracy_std'],
                     qwen_probes['lr_accuracy_mean'] + qwen_probes['lr_accuracy_std'], alpha=0.2)
axes[1].plot(qwen_probes['layer'], qwen_probes['diffmean_1d_accuracy'], 'r-s',
             markersize=4, label='DiffMean 1D')
axes[1].axhline(y=0.5, color='gray', linestyle='--', label='Chance')
axes[1].set_xlabel("Layer")
axes[1].set_ylabel("Accuracy")
axes[1].set_title("Qwen2.5-3B-Instruct")
axes[1].legend()
axes[1].set_ylim(0.4, 1.05)

fig.suptitle("Human vs AI Text Classification Accuracy by Layer", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(PLOTS / "combined_layer_accuracy.png", dpi=150, bbox_inches='tight')
plt.close()

# ── Summary comparison bar chart ──
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Bar chart: best accuracies
models = ['Pythia-2.8B\n(Base)', 'Qwen2.5-3B\n(Instruct)']
lr_accs = [pythia_summary['best_lr_accuracy'], qwen_summary['best_lr_accuracy']]
lr_stds = [pythia_summary['best_lr_accuracy_std'], qwen_summary['best_lr_accuracy_std']]
dm_accs = [pythia_summary['best_diffmean_accuracy'], qwen_summary['best_diffmean_accuracy']]
random_accs = [pythia_summary['random_baseline_accuracy'], qwen_summary['random_baseline_accuracy']]

x = np.arange(len(models))
width = 0.25
axes[0].bar(x - width, lr_accs, width, yerr=lr_stds, label='Linear Probe', capsize=5)
axes[0].bar(x, dm_accs, width, label='DiffMean 1D', capsize=5)
axes[0].bar(x + width, random_accs, width, label='Random Baseline', capsize=5)
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Best Classification Accuracy')
axes[0].set_xticks(x)
axes[0].set_xticklabels(models)
axes[0].legend()
axes[0].set_ylim(0, 1.1)
axes[0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

# Best layers
best_layers = [pythia_summary['best_layer'], qwen_summary['best_layer']]
n_layers_list = [pythia_summary['n_layers'], qwen_summary['n_layers']]
normalized_layers = [bl/nl for bl, nl in zip(best_layers, n_layers_list)]

axes[1].bar(models, normalized_layers, color=['steelblue', 'coral'])
axes[1].set_ylabel('Best Layer (fraction of total)')
axes[1].set_title('Where is AI Style Encoded?')
for i, (bl, nl) in enumerate(zip(best_layers, n_layers_list)):
    axes[1].text(i, normalized_layers[i] + 0.02, f'Layer {bl}/{nl}', ha='center', fontsize=10)
axes[1].set_ylim(0, 1.1)

# Z-scores
z_scores = [pythia_summary['z_score_vs_random'], qwen_summary['z_score_vs_random']]
axes[2].bar(models, z_scores, color=['steelblue', 'coral'])
axes[2].set_ylabel('Z-score')
axes[2].set_title('DiffMean vs Random Directions')
axes[2].axhline(y=1.96, color='red', linestyle='--', alpha=0.5, label='p=0.05 (1-tail)')
axes[2].legend()

plt.tight_layout()
plt.savefig(PLOTS / "combined_summary.png", dpi=150, bbox_inches='tight')
plt.close()

# ── Cross-domain comparison ──
pythia_cross = pd.read_csv(RESULTS / "cross_domain_results.csv")
qwen_cross = pd.read_csv(RESULTS / "qwen" / "cross_domain_results.csv")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for i, (df, title) in enumerate([(pythia_cross, 'Pythia-2.8B'), (qwen_cross, 'Qwen2.5-3B-Instruct')]):
    if len(df) > 0:
        pivot = df.pivot(index='train_domain', columns='test_domain', values='accuracy')
        sns.heatmap(pivot, annot=True, fmt='.2f', cmap='YlOrRd', vmin=0.5, vmax=1.0,
                   ax=axes[i])
        axes[i].set_title(f'{title}\n(Mean: {df["accuracy"].mean():.3f})')

fig.suptitle("Cross-Domain Generalization", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(PLOTS / "combined_cross_domain.png", dpi=150, bbox_inches='tight')
plt.close()

print("Combined plots saved.")
print(f"\nPythia-2.8B: best LR={pythia_summary['best_lr_accuracy']:.3f} at layer {pythia_summary['best_layer']}")
print(f"Qwen2.5-3B-Instruct: best LR={qwen_summary['best_lr_accuracy']:.3f} at layer {qwen_summary['best_layer']}")
print(f"Cross-domain: Pythia={pythia_cross['accuracy'].mean():.3f}, Qwen={qwen_cross['accuracy'].mean():.3f}")
