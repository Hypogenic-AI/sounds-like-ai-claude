"""
Experiment: Is there a "sounds like AI" direction in the residual stream?

This script:
1. Loads paired human/ChatGPT text from HC3
2. Processes through a model via TransformerLens
3. Collects residual stream activations
4. Computes difference-in-means direction
5. Validates with linear probes, PCA, and causal interventions
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
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

RESULTS_DIR = Path("/workspaces/sounds-like-ai-claude/results")
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ── Step 1: Load and prepare HC3 data ──

def load_hc3_data(path, max_samples=500, max_text_len=512):
    """Load HC3 dataset and create paired human/AI text samples."""
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

    # Sample balanced pairs
    random.shuffle(data)

    human_texts = []
    ai_texts = []
    sources = []

    for item in data[:max_samples]:
        # Take first human and first AI answer for each question
        h = item["human"][0][:max_text_len]
        a = item["ai"][0][:max_text_len]
        if len(h.split()) > 10 and len(a.split()) > 10:  # Filter very short
            human_texts.append(h)
            ai_texts.append(a)
            sources.append(item["source"])

    print(f"Loaded {len(human_texts)} paired samples")
    print(f"Sources: {pd.Series(sources).value_counts().to_dict()}")
    return human_texts, ai_texts, sources


print("\n=== Loading HC3 data ===")
hc3_path = "/workspaces/sounds-like-ai-claude/datasets/hc3/all.jsonl"
human_texts, ai_texts, sources = load_hc3_data(hc3_path, max_samples=400)

# ── Step 2: Load model via TransformerLens ──

print("\n=== Loading model ===")
import transformer_lens

# Use Pythia-2.8B — open model, well-supported by TransformerLens
MODEL_NAME = "pythia-2.8b"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = transformer_lens.HookedTransformer.from_pretrained(
    MODEL_NAME,
    device=device,
    dtype=torch.float16,
)
print(f"Model: {MODEL_NAME}")
print(f"Layers: {model.cfg.n_layers}, d_model: {model.cfg.d_model}")
print(f"Device: {device}")

n_layers = model.cfg.n_layers
d_model = model.cfg.d_model

# ── Step 3: Collect residual stream activations ──

def get_residual_activations(model, texts, batch_size=8, max_tokens=128):
    """
    Run texts through model, collect residual stream activations at the last token position.
    Returns: tensor of shape (n_texts, n_layers, d_model)
    """
    all_activations = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Collecting activations"):
        batch_texts = texts[i:i+batch_size]

        # Tokenize with truncation
        tokens = model.to_tokens(batch_texts, prepend_bos=True)
        if tokens.shape[1] > max_tokens:
            tokens = tokens[:, :max_tokens]

        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens,
                names_filter=lambda name: "resid_post" in name,
            )

        # Get last token activations for each layer
        batch_acts = []
        for layer in range(n_layers):
            # resid_post at each layer, last token position
            act = cache[f"blocks.{layer}.hook_resid_post"]  # (batch, seq, d_model)
            # Use last non-padding token
            last_pos_act = act[:, -1, :]  # (batch, d_model)
            batch_acts.append(last_pos_act.cpu().float())

        # Stack: (batch, n_layers, d_model)
        batch_acts = torch.stack(batch_acts, dim=1)
        all_activations.append(batch_acts)

        # Free cache memory
        del cache
        torch.cuda.empty_cache()

    return torch.cat(all_activations, dim=0)  # (n_texts, n_layers, d_model)


print("\n=== Collecting activations for human texts ===")
human_acts = get_residual_activations(model, human_texts)
print(f"Human activations shape: {human_acts.shape}")

print("\n=== Collecting activations for AI texts ===")
ai_acts = get_residual_activations(model, ai_texts)
print(f"AI activations shape: {ai_acts.shape}")

# Save activations
torch.save(human_acts, RESULTS_DIR / "human_activations.pt")
torch.save(ai_acts, RESULTS_DIR / "ai_activations.pt")
print("Activations saved.")

# ── Step 4: Compute difference-in-means direction ──

print("\n=== Computing difference-in-means directions ===")

# DiffMean direction at each layer
diff_means = []  # (n_layers, d_model)
for layer in range(n_layers):
    h_mean = human_acts[:, layer, :].mean(dim=0)
    a_mean = ai_acts[:, layer, :].mean(dim=0)
    diff = a_mean - h_mean  # "AI direction" = AI - Human
    diff = diff / diff.norm()  # Normalize
    diff_means.append(diff)

diff_means = torch.stack(diff_means)  # (n_layers, d_model)
print(f"Direction matrix shape: {diff_means.shape}")

# ── Step 5: Linear probe analysis ──

print("\n=== Training linear probes per layer ===")

probe_results = []
n_samples = human_acts.shape[0]

for layer in range(n_layers):
    # Combine human and AI activations
    X = torch.cat([human_acts[:, layer, :], ai_acts[:, layer, :]], dim=0).numpy()
    y = np.array([0] * n_samples + [1] * n_samples)  # 0=human, 1=AI

    # Cross-validated logistic regression
    clf = LogisticRegression(max_iter=1000, random_state=SEED, C=1.0)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')

    # Also fit on all data for the probe direction
    clf.fit(X, y)

    # DiffMean 1D projection accuracy
    proj = X @ diff_means[layer].numpy()
    threshold = np.median(proj)
    diffmean_acc = ((proj > threshold) == y).mean()

    probe_results.append({
        'layer': layer,
        'lr_accuracy_mean': scores.mean(),
        'lr_accuracy_std': scores.std(),
        'diffmean_1d_accuracy': diffmean_acc,
        'lr_coef_norm': np.linalg.norm(clf.coef_),
    })

    if layer % 5 == 0 or layer == n_layers - 1:
        print(f"  Layer {layer:2d}: LR acc = {scores.mean():.3f} ± {scores.std():.3f}, "
              f"DiffMean 1D acc = {diffmean_acc:.3f}")

probe_df = pd.DataFrame(probe_results)
probe_df.to_csv(RESULTS_DIR / "probe_results.csv", index=False)

# ── Step 6: PCA visualization ──

print("\n=== PCA visualization ===")
from sklearn.decomposition import PCA

# Find best layer (highest LR accuracy)
best_layer = probe_df.loc[probe_df['lr_accuracy_mean'].idxmax(), 'layer']
best_layer = int(best_layer)
print(f"Best layer for classification: {best_layer} (acc = {probe_df.loc[best_layer, 'lr_accuracy_mean']:.3f})")

# PCA on combined activations at best layer
X_best = torch.cat([human_acts[:, best_layer, :], ai_acts[:, best_layer, :]], dim=0).numpy()
y_best = np.array([0] * n_samples + [1] * n_samples)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_best)

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y_best, cmap='coolwarm', alpha=0.5, s=20)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
ax.set_title(f"PCA of Residual Stream at Layer {best_layer}\n(Blue=Human, Red=AI)")
plt.colorbar(scatter, label="Class (0=Human, 1=AI)")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "pca_best_layer.png", dpi=150)
plt.close()
print(f"PCA plot saved.")

# ── Step 7: Layer-wise accuracy plot ──

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.plot(probe_df['layer'], probe_df['lr_accuracy_mean'], 'b-o', label='Logistic Regression (5-fold CV)', markersize=4)
ax.fill_between(probe_df['layer'],
                probe_df['lr_accuracy_mean'] - probe_df['lr_accuracy_std'],
                probe_df['lr_accuracy_mean'] + probe_df['lr_accuracy_std'],
                alpha=0.2)
ax.plot(probe_df['layer'], probe_df['diffmean_1d_accuracy'], 'r-s', label='DiffMean 1D projection', markersize=4)
ax.axhline(y=0.5, color='gray', linestyle='--', label='Chance')
ax.set_xlabel("Layer")
ax.set_ylabel("Accuracy")
ax.set_title("Human vs AI Text Classification Accuracy by Layer")
ax.legend()
ax.set_ylim(0.4, 1.05)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "layer_accuracy.png", dpi=150)
plt.close()
print("Layer accuracy plot saved.")

# ── Step 8: Direction cosine similarity across layers ──

print("\n=== Direction consistency across layers ===")
cos_sim_matrix = torch.zeros(n_layers, n_layers)
for i in range(n_layers):
    for j in range(n_layers):
        cos_sim_matrix[i, j] = torch.dot(diff_means[i], diff_means[j]).item()

fig, ax = plt.subplots(1, 1, figsize=(8, 7))
sns.heatmap(cos_sim_matrix.numpy(), cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            xticklabels=range(n_layers), yticklabels=range(n_layers), ax=ax)
ax.set_xlabel("Layer")
ax.set_ylabel("Layer")
ax.set_title("Cosine Similarity of 'AI Style' Directions Across Layers")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "direction_cosine_similarity.png", dpi=150)
plt.close()
print("Cosine similarity heatmap saved.")

# ── Step 9: Cross-domain generalization ──

print("\n=== Cross-domain generalization ===")

# Split data by source domain
source_arr = np.array(sources)
unique_sources = list(set(sources))
print(f"Available domains: {unique_sources}")

if len(unique_sources) >= 2:
    cross_domain_results = []

    for train_source in unique_sources:
        for test_source in unique_sources:
            if train_source == test_source:
                continue

            train_mask = source_arr == train_source
            test_mask = source_arr == test_source

            if train_mask.sum() < 10 or test_mask.sum() < 10:
                continue

            # Use best layer
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
                'train_domain': train_source,
                'test_domain': test_source,
                'accuracy': acc,
                'n_train': len(y_train),
                'n_test': len(y_test),
            })

    cross_df = pd.DataFrame(cross_domain_results)
    cross_df.to_csv(RESULTS_DIR / "cross_domain_results.csv", index=False)
    print(cross_df.to_string())

    # Heatmap
    if len(cross_df) > 0:
        pivot = cross_df.pivot(index='train_domain', columns='test_domain', values='accuracy')
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(pivot, annot=True, fmt='.2f', cmap='YlOrRd', vmin=0.5, vmax=1.0, ax=ax)
        ax.set_title(f"Cross-Domain Generalization (Layer {best_layer})")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "cross_domain_heatmap.png", dpi=150)
        plt.close()
        print("Cross-domain heatmap saved.")

# ── Step 10: Random direction baseline ──

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

print(f"Random direction acc: {random_accs.mean():.3f} ± {random_accs.std():.3f}")
print(f"DiffMean direction acc: {actual_diffmean_acc:.3f}")
print(f"Z-score: {(actual_diffmean_acc - random_accs.mean()) / random_accs.std():.1f}")

# ── Step 11: Causal intervention (steering) ──

print("\n=== Causal intervention: steering test ===")

# Test: add/remove AI direction during generation
test_prompts = [
    "Explain why the sky is blue.",
    "What are the benefits of exercise?",
    "How does a car engine work?",
    "Why is sleep important?",
    "What causes earthquakes?",
]

steering_dir = diff_means[best_layer].to(device).half()

def generate_with_steering(model, prompt, steering_vector, alpha=0.0, max_tokens=100):
    """Generate text with optional steering along the AI direction."""
    tokens = model.to_tokens(prompt, prepend_bos=True)

    if alpha == 0.0:
        # Normal generation
        with torch.no_grad():
            output = model.generate(
                tokens,
                max_new_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
            )
        return model.to_string(output[0])

    # Hook for steering
    def steering_hook(activation, hook):
        activation[:, :, :] += alpha * steering_vector
        return activation

    hook_name = f"blocks.{best_layer}.hook_resid_post"

    with torch.no_grad():
        with model.hooks(fwd_hooks=[(hook_name, steering_hook)]):
            output = model.generate(
                tokens,
                max_new_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
            )
    return model.to_string(output[0])


steering_results = []
alphas = [-10.0, -5.0, 0.0, 5.0, 10.0]

for prompt in tqdm(test_prompts[:3], desc="Steering test"):
    for alpha in alphas:
        try:
            text = generate_with_steering(model, prompt, steering_dir, alpha=alpha, max_tokens=80)
            steering_results.append({
                'prompt': prompt,
                'alpha': alpha,
                'output': text,
            })
        except Exception as e:
            print(f"  Error with alpha={alpha}: {e}")
            steering_results.append({
                'prompt': prompt,
                'alpha': alpha,
                'output': f"ERROR: {e}",
            })

# Save steering results
steering_df = pd.DataFrame(steering_results)
steering_df.to_csv(RESULTS_DIR / "steering_results.csv", index=False)

# Print sample outputs
print("\n--- Steering examples ---")
for prompt in test_prompts[:2]:
    print(f"\nPrompt: {prompt}")
    for alpha in alphas:
        row = steering_df[(steering_df['prompt'] == prompt) & (steering_df['alpha'] == alpha)]
        if len(row) > 0:
            output = row.iloc[0]['output']
            # Show just first 200 chars of output after prompt
            output_after_prompt = output[len(prompt):] if prompt in output else output
            print(f"  α={alpha:+5.1f}: {output_after_prompt[:200].strip()}")

# ── Step 12: Measure steering effect on activations ──

print("\n=== Measuring steering effect on activation projections ===")

# For each alpha, generate text and measure how much the activations
# project onto the AI direction
projection_by_alpha = {}

for alpha in alphas:
    projs = []
    for prompt in test_prompts[:3]:
        try:
            tokens = model.to_tokens(prompt, prepend_bos=True)

            if alpha == 0.0:
                with torch.no_grad():
                    _, cache = model.run_with_cache(
                        tokens,
                        names_filter=lambda name: f"blocks.{best_layer}.hook_resid_post" in name,
                    )
            else:
                def steering_hook(activation, hook):
                    activation[:, :, :] += alpha * steering_dir
                    return activation

                hook_name = f"blocks.{best_layer}.hook_resid_post"
                with torch.no_grad():
                    with model.hooks(fwd_hooks=[(hook_name, steering_hook)]):
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

print("Mean projection onto AI direction by steering strength:")
for alpha, proj in sorted(projection_by_alpha.items()):
    print(f"  α={alpha:+5.1f}: projection = {proj:.2f}")

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(range(len(projection_by_alpha)), list(projection_by_alpha.values()),
       tick_label=[f"α={a}" for a in projection_by_alpha.keys()])
ax.set_xlabel("Steering strength")
ax.set_ylabel("Mean projection onto AI direction")
ax.set_title("Effect of Steering on AI Direction Projection")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "steering_projection.png", dpi=150)
plt.close()

# ── Step 13: Save summary results ──

print("\n=== Saving summary ===")
summary = {
    'model': MODEL_NAME,
    'n_layers': n_layers,
    'd_model': d_model,
    'n_samples': n_samples,
    'best_layer': int(best_layer),
    'best_lr_accuracy': float(probe_df.loc[best_layer, 'lr_accuracy_mean']),
    'best_lr_accuracy_std': float(probe_df.loc[best_layer, 'lr_accuracy_std']),
    'best_diffmean_accuracy': float(actual_diffmean_acc),
    'random_baseline_accuracy': float(random_accs.mean()),
    'random_baseline_std': float(random_accs.std()),
    'z_score_vs_random': float((actual_diffmean_acc - random_accs.mean()) / random_accs.std()),
    'seed': SEED,
}

with open(RESULTS_DIR / "summary.json", 'w') as f:
    json.dump(summary, f, indent=2)

print("\n" + "="*60)
print("EXPERIMENT COMPLETE")
print("="*60)
print(f"Model: {MODEL_NAME}")
print(f"Samples: {n_samples} paired human/AI texts")
print(f"Best layer: {best_layer}")
print(f"Best LR accuracy: {summary['best_lr_accuracy']:.3f} ± {summary['best_lr_accuracy_std']:.3f}")
print(f"DiffMean 1D accuracy: {summary['best_diffmean_accuracy']:.3f}")
print(f"Random baseline: {summary['random_baseline_accuracy']:.3f} ± {summary['random_baseline_std']:.3f}")
print(f"Z-score vs random: {summary['z_score_vs_random']:.1f}")
print(f"\nResults saved to: {RESULTS_DIR}")
print(f"Plots saved to: {PLOTS_DIR}")
