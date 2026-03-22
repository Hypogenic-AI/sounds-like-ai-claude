"""
Experiment with Qwen2.5-3B-Instruct: An instruct-tuned model that actually
generates "AI-style" text, making the direction more likely to be present and
causally relevant.
"""

import json
import random
import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

RESULTS_DIR = Path("/workspaces/sounds-like-ai-claude/results/qwen")
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ── Load HC3 data ──

def load_hc3_data(path, max_samples=400, max_text_len=512):
    data = []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            human_answers = item.get("human_answers", [])
            chatgpt_answers = item.get("chatgpt_answers", [])
            source = item.get("source", "unknown")
            if human_answers and chatgpt_answers:
                data.append({
                    "question": item["question"],
                    "human": human_answers,
                    "ai": chatgpt_answers,
                    "source": source,
                })

    random.shuffle(data)
    human_texts, ai_texts, sources = [], [], []
    for item in data[:max_samples]:
        h = item["human"][0][:max_text_len]
        a = item["ai"][0][:max_text_len]
        if len(h.split()) > 10 and len(a.split()) > 10:
            human_texts.append(h)
            ai_texts.append(a)
            sources.append(item["source"])

    print(f"Loaded {len(human_texts)} paired samples")
    print(f"Sources: {pd.Series(sources).value_counts().to_dict()}")
    return human_texts, ai_texts, sources

print("\n=== Loading HC3 data ===")
human_texts, ai_texts, sources = load_hc3_data(
    "/workspaces/sounds-like-ai-claude/datasets/hc3/all.jsonl"
)

# ── Load Qwen2.5-3B-Instruct ──

print("\n=== Loading Qwen2.5-3B-Instruct ===")
import transformer_lens

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
device = "cuda:0"

model = transformer_lens.HookedTransformer.from_pretrained(
    MODEL_NAME,
    device=device,
    dtype=torch.float16,
)
print(f"Model: {MODEL_NAME}")
print(f"Layers: {model.cfg.n_layers}, d_model: {model.cfg.d_model}")

n_layers = model.cfg.n_layers
d_model = model.cfg.d_model

# ── Collect residual stream activations ──

def get_residual_activations(model, texts, batch_size=8, max_tokens=128):
    all_activations = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Collecting activations"):
        batch_texts = texts[i:i+batch_size]
        tokens = model.to_tokens(batch_texts, prepend_bos=True)
        if tokens.shape[1] > max_tokens:
            tokens = tokens[:, :max_tokens]

        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens,
                names_filter=lambda name: "resid_post" in name,
            )

        batch_acts = []
        for layer in range(n_layers):
            act = cache[f"blocks.{layer}.hook_resid_post"]
            last_pos_act = act[:, -1, :].cpu().float()
            batch_acts.append(last_pos_act)

        batch_acts = torch.stack(batch_acts, dim=1)
        all_activations.append(batch_acts)
        del cache
        torch.cuda.empty_cache()

    return torch.cat(all_activations, dim=0)

print("\n=== Collecting activations ===")
human_acts = get_residual_activations(model, human_texts)
print(f"Human activations: {human_acts.shape}")
ai_acts = get_residual_activations(model, ai_texts)
print(f"AI activations: {ai_acts.shape}")

torch.save(human_acts, RESULTS_DIR / "human_activations.pt")
torch.save(ai_acts, RESULTS_DIR / "ai_activations.pt")

# ── Compute DiffMean directions ──

print("\n=== Computing DiffMean directions ===")
n_samples = human_acts.shape[0]
diff_means = []
for layer in range(n_layers):
    h_mean = human_acts[:, layer, :].mean(dim=0)
    a_mean = ai_acts[:, layer, :].mean(dim=0)
    diff = a_mean - h_mean
    diff = diff / diff.norm()
    diff_means.append(diff)
diff_means = torch.stack(diff_means)

# ── Linear probes ──

print("\n=== Linear probes per layer ===")
probe_results = []
for layer in range(n_layers):
    X = torch.cat([human_acts[:, layer, :], ai_acts[:, layer, :]], dim=0).numpy()
    y = np.array([0] * n_samples + [1] * n_samples)

    clf = LogisticRegression(max_iter=1000, random_state=SEED)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
    clf.fit(X, y)

    proj = X @ diff_means[layer].numpy()
    threshold = np.median(proj)
    diffmean_acc = ((proj > threshold) == y).mean()

    # Cosine similarity between LR direction and DiffMean direction
    lr_dir = clf.coef_[0] / np.linalg.norm(clf.coef_[0])
    cos_sim = np.dot(lr_dir, diff_means[layer].numpy())

    probe_results.append({
        'layer': layer,
        'lr_accuracy_mean': scores.mean(),
        'lr_accuracy_std': scores.std(),
        'diffmean_1d_accuracy': diffmean_acc,
        'lr_diffmean_cosine': cos_sim,
    })

    if layer % 5 == 0 or layer == n_layers - 1:
        print(f"  Layer {layer:2d}: LR={scores.mean():.3f}±{scores.std():.3f}, "
              f"DiffMean={diffmean_acc:.3f}, cos(LR,DM)={cos_sim:.3f}")

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
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y_best, cmap='coolwarm', alpha=0.5, s=20)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
ax.set_title(f"PCA of Qwen2.5-3B-Instruct Residual Stream (Layer {best_layer})\n(Blue=Human, Red=AI)")
plt.colorbar(scatter, label="0=Human, 1=AI")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "pca_best_layer.png", dpi=150)
plt.close()

# ── Layer accuracy plot ──

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(probe_df['layer'], probe_df['lr_accuracy_mean'], 'b-o', markersize=4, label='LR (5-fold CV)')
axes[0].fill_between(probe_df['layer'],
                     probe_df['lr_accuracy_mean'] - probe_df['lr_accuracy_std'],
                     probe_df['lr_accuracy_mean'] + probe_df['lr_accuracy_std'], alpha=0.2)
axes[0].plot(probe_df['layer'], probe_df['diffmean_1d_accuracy'], 'r-s', markersize=4, label='DiffMean 1D')
axes[0].axhline(y=0.5, color='gray', linestyle='--', label='Chance')
axes[0].set_xlabel("Layer")
axes[0].set_ylabel("Accuracy")
axes[0].set_title("Human vs AI Classification Accuracy by Layer\n(Qwen2.5-3B-Instruct)")
axes[0].legend()
axes[0].set_ylim(0.4, 1.05)

axes[1].plot(probe_df['layer'], probe_df['lr_diffmean_cosine'], 'g-^', markersize=4)
axes[1].set_xlabel("Layer")
axes[1].set_ylabel("Cosine Similarity")
axes[1].set_title("LR vs DiffMean Direction Alignment by Layer")
axes[1].axhline(y=0, color='gray', linestyle='--')

plt.tight_layout()
plt.savefig(PLOTS_DIR / "layer_analysis.png", dpi=150)
plt.close()

# ── Direction cosine similarity matrix ──

cos_sim_matrix = torch.zeros(n_layers, n_layers)
for i in range(n_layers):
    for j in range(n_layers):
        cos_sim_matrix[i, j] = torch.dot(diff_means[i], diff_means[j]).item()

fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(cos_sim_matrix.numpy(), cmap='RdBu_r', center=0, vmin=-1, vmax=1, ax=ax)
ax.set_xlabel("Layer")
ax.set_ylabel("Layer")
ax.set_title("Cosine Similarity of AI-Style Directions Across Layers\n(Qwen2.5-3B-Instruct)")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "direction_cosine_similarity.png", dpi=150)
plt.close()

# ── Cross-domain generalization ──

print("\n=== Cross-domain generalization ===")
source_arr = np.array(sources)
unique_sources = list(set(sources))
cross_domain_results = []

for train_src in unique_sources:
    for test_src in unique_sources:
        if train_src == test_src:
            continue
        train_mask = source_arr == train_src
        test_mask = source_arr == test_src
        if train_mask.sum() < 10 or test_mask.sum() < 10:
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

        clf = LogisticRegression(max_iter=1000, random_state=SEED)
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)

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
    ax.set_title(f"Cross-Domain Generalization (Qwen2.5-3B-Instruct, Layer {best_layer})")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "cross_domain_heatmap.png", dpi=150)
    plt.close()

# ── Random direction baseline ──

print("\n=== Random direction baseline ===")
n_random = 100
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

print(f"Random direction acc: {random_accs.mean():.3f} ± {random_accs.std():.3f}")
print(f"DiffMean direction acc: {actual_diffmean_acc:.3f}")
print(f"Z-score: {z_score:.1f}")

# ── Causal intervention: steering ──

print("\n=== Causal intervention: steering test ===")
steering_dir = diff_means[best_layer].to(device).half()

test_prompts = [
    "Explain why the sky is blue.",
    "What are the benefits of exercise?",
    "How does a car engine work?",
]

alphas = [-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0]
steering_results = []

for prompt in tqdm(test_prompts, desc="Steering"):
    for alpha in alphas:
        try:
            tokens = model.to_tokens(prompt, prepend_bos=True)

            if alpha == 0.0:
                with torch.no_grad():
                    output = model.generate(tokens, max_new_tokens=80, temperature=0.7, top_p=0.9)
            else:
                def make_hook(a, sv):
                    def hook_fn(activation, hook):
                        activation[:, :, :] += a * sv
                        return activation
                    return hook_fn

                hook_name = f"blocks.{best_layer}.hook_resid_post"
                with torch.no_grad():
                    with model.hooks(fwd_hooks=[(hook_name, make_hook(alpha, steering_dir))]):
                        output = model.generate(tokens, max_new_tokens=80, temperature=0.7, top_p=0.9)

            text = model.to_string(output[0])
            steering_results.append({'prompt': prompt, 'alpha': alpha, 'output': text})
        except Exception as e:
            print(f"  Error α={alpha}: {e}")
            steering_results.append({'prompt': prompt, 'alpha': alpha, 'output': f"ERROR: {e}"})

steering_df = pd.DataFrame(steering_results)
steering_df.to_csv(RESULTS_DIR / "steering_results.csv", index=False)

# Print examples
print("\n--- Steering examples ---")
for prompt in test_prompts[:2]:
    print(f"\nPrompt: {prompt}")
    for alpha in alphas:
        row = steering_df[(steering_df['prompt'] == prompt) & (steering_df['alpha'] == alpha)]
        if len(row) > 0:
            output = row.iloc[0]['output']
            suffix = output[len(prompt):] if prompt in output else output
            print(f"  α={alpha:+6.1f}: {suffix[:200].strip()}")

# ── Projection shift under steering ──

print("\n=== Projection shift under steering ===")
projection_by_alpha = {}
for alpha in alphas:
    projs = []
    for prompt in test_prompts:
        try:
            tokens = model.to_tokens(prompt, prepend_bos=True)
            if alpha == 0.0:
                with torch.no_grad():
                    _, cache = model.run_with_cache(
                        tokens,
                        names_filter=lambda name: f"blocks.{best_layer}.hook_resid_post" in name,
                    )
            else:
                def make_hook2(a, sv):
                    def hook_fn(activation, hook):
                        activation[:, :, :] += a * sv
                        return activation
                    return hook_fn
                with torch.no_grad():
                    with model.hooks(fwd_hooks=[(f"blocks.{best_layer}.hook_resid_post", make_hook2(alpha, steering_dir))]):
                        _, cache = model.run_with_cache(
                            tokens,
                            names_filter=lambda name: f"blocks.{best_layer}.hook_resid_post" in name,
                        )
            act = cache[f"blocks.{best_layer}.hook_resid_post"][0, -1, :]
            proj = (act.float().cpu() @ diff_means[best_layer]).item()
            projs.append(proj)
            del cache
            torch.cuda.empty_cache()
        except:
            pass
    if projs:
        projection_by_alpha[alpha] = np.mean(projs)

print("Mean projection onto AI direction by steering:")
for alpha, proj in sorted(projection_by_alpha.items()):
    print(f"  α={alpha:+6.1f}: projection = {proj:.2f}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(range(len(projection_by_alpha)), list(projection_by_alpha.values()),
       tick_label=[f"α={a}" for a in projection_by_alpha.keys()])
ax.set_xlabel("Steering strength")
ax.set_ylabel("Mean projection onto AI direction")
ax.set_title("Effect of Steering on AI Direction Projection\n(Qwen2.5-3B-Instruct)")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "steering_projection.png", dpi=150)
plt.close()

# ── Save summary ──

summary = {
    'model': MODEL_NAME,
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
    'seed': SEED,
}

with open(RESULTS_DIR / "summary.json", 'w') as f:
    json.dump(summary, f, indent=2)

print("\n" + "="*60)
print("EXPERIMENT COMPLETE (Qwen2.5-3B-Instruct)")
print("="*60)
print(f"Model: {MODEL_NAME}")
print(f"Samples: {n_samples}")
print(f"Best layer: {best_layer}")
print(f"Best LR accuracy: {summary['best_lr_accuracy']:.3f} ± {summary['best_lr_accuracy_std']:.3f}")
print(f"DiffMean 1D accuracy: {summary['best_diffmean_accuracy']:.3f}")
print(f"Random baseline: {summary['random_baseline_accuracy']:.3f} ± {summary['random_baseline_std']:.3f}")
print(f"Z-score: {summary['z_score_vs_random']:.1f}")
