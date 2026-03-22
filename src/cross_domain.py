"""
Cross-domain generalization analysis.
Uses saved activations and properly tracks domain sources.
"""

import json
import random
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import SGDClassifier
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

# Reload data with matching shuffle
hc3_path = "/workspaces/sounds-like-ai-claude/datasets/hc3/all.jsonl"
data = []
with open(hc3_path) as f:
    for line in f:
        item = json.loads(line)
        if item.get("human_answers") and item.get("chatgpt_answers"):
            data.append(item)

random.shuffle(data)  # Same seed=42

human_texts, ai_texts, sources = [], [], []
for item in data[:400]:
    h = item["human_answers"][0][:512]
    a = item["chatgpt_answers"][0][:512]
    if len(h.split()) > 10 and len(a.split()) > 10:
        human_texts.append(h)
        ai_texts.append(a)
        sources.append(item.get("source", "unknown"))

print(f"Total samples: {len(sources)}")
print(f"Sources: {pd.Series(sources).value_counts().to_dict()}")

# Load activations
human_acts = torch.load(RESULTS_DIR / "human_activations.pt", weights_only=True)
ai_acts = torch.load(RESULTS_DIR / "ai_activations.pt", weights_only=True)
n_samples = human_acts.shape[0]

assert len(sources) == n_samples, f"Source mismatch: {len(sources)} vs {n_samples}"

# Read best layer from summary
with open(RESULTS_DIR / "summary.json") as f:
    summary = json.load(f)
best_layer = summary['best_layer']
print(f"Best layer: {best_layer}")

# Cross-domain generalization
source_arr = np.array(sources)
unique_sources = sorted(set(sources))
print(f"Unique domains: {unique_sources}")
for s in unique_sources:
    print(f"  {s}: {(source_arr == s).sum()} samples")

cross_results = []
for train_src in unique_sources:
    for test_src in unique_sources:
        if train_src == test_src:
            continue
        train_mask = source_arr == train_src
        test_mask = source_arr == test_src
        if train_mask.sum() < 5 or test_mask.sum() < 5:
            continue

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

        cross_results.append({
            'train_domain': train_src,
            'test_domain': test_src,
            'accuracy': acc,
            'n_train': len(y_train),
            'n_test': len(y_test),
        })

cross_df = pd.DataFrame(cross_results)
cross_df.to_csv(RESULTS_DIR / "cross_domain_results.csv", index=False)

print("\nCross-domain results:")
print(cross_df.to_string())

if len(cross_df) > 0:
    mean_acc = cross_df['accuracy'].mean()
    print(f"\nMean cross-domain accuracy: {mean_acc:.3f}")

    pivot = cross_df.pivot(index='train_domain', columns='test_domain', values='accuracy')
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='YlOrRd', vmin=0.5, vmax=1.0, ax=ax)
    ax.set_title(f"Cross-Domain Generalization\n(Pythia-2.8B, Layer {best_layer})")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "cross_domain_heatmap.png", dpi=150)
    plt.close()
    print("Cross-domain heatmap saved.")
else:
    print("Not enough cross-domain pairs with sufficient samples.")
