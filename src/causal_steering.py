"""
Causal steering experiment: Load Pythia-2.8B and test whether adding/removing
the AI-style direction changes the model's output style.
"""

import json
import random
import numpy as np
import torch
import transformer_lens
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

RESULTS_DIR = Path("/workspaces/sounds-like-ai-claude/results")
PLOTS_DIR = RESULTS_DIR / "plots"

# Load model
print("=== Loading Pythia-2.8B ===")
device = "cuda:0"
model = transformer_lens.HookedTransformer.from_pretrained(
    "pythia-2.8b", device=device, dtype=torch.float16
)

# Load saved DiffMean direction
human_acts = torch.load(RESULTS_DIR / "human_activations.pt", weights_only=True)
ai_acts = torch.load(RESULTS_DIR / "ai_activations.pt", weights_only=True)

with open(RESULTS_DIR / "summary.json") as f:
    summary = json.load(f)
best_layer = summary['best_layer']

# Compute direction at best layer
h_mean = human_acts[:, best_layer, :].mean(dim=0)
a_mean = ai_acts[:, best_layer, :].mean(dim=0)
diff = a_mean - h_mean
ai_direction = (diff / diff.norm()).to(device).half()

print(f"Best layer: {best_layer}")

# Test prompts — diverse topics for generalizability
test_prompts = [
    "Explain why the sky is blue.",
    "What are the benefits of regular exercise?",
    "How does a car engine work?",
    "Why is sleep important for health?",
    "What causes earthquakes?",
    "How do computers process information?",
    "Why do leaves change color in autumn?",
    "What is the theory of relativity?",
]

# Steering alphas
alphas = [-20.0, -10.0, -5.0, 0.0, 5.0, 10.0, 20.0]

print("\n=== Generating steered outputs ===")
results = []

for prompt in test_prompts:
    print(f"\nPrompt: {prompt}")
    for alpha in alphas:
        tokens = model.to_tokens(prompt, prepend_bos=True)

        try:
            if alpha == 0.0:
                with torch.no_grad():
                    output = model.generate(tokens, max_new_tokens=100,
                                          temperature=0.7, top_p=0.9)
            else:
                def make_hook(a, d):
                    def hook_fn(activation, hook):
                        activation[:, :, :] += a * d
                        return activation
                    return hook_fn

                hook_name = f"blocks.{best_layer}.hook_resid_post"
                with torch.no_grad():
                    with model.hooks(fwd_hooks=[(hook_name, make_hook(alpha, ai_direction))]):
                        output = model.generate(tokens, max_new_tokens=100,
                                              temperature=0.7, top_p=0.9)

            text = model.to_string(output[0])
            gen_text = text[len(model.to_string(tokens[0])):].strip()

            results.append({
                'prompt': prompt,
                'alpha': alpha,
                'generated': gen_text,
                'full_output': text,
            })
            print(f"  α={alpha:+6.1f}: {gen_text[:150]}")
        except Exception as e:
            print(f"  α={alpha:+6.1f}: ERROR - {e}")
            results.append({
                'prompt': prompt,
                'alpha': alpha,
                'generated': f"ERROR: {e}",
                'full_output': "",
            })

# Save results
import pandas as pd
df = pd.DataFrame(results)
df.to_csv(RESULTS_DIR / "steering_results.csv", index=False)

# ── Measure projection shift ──
print("\n=== Measuring activation projections under steering ===")

# Compute DiffMean direction (unnormalized) for all layers for multi-layer analysis
diff_means_all = []
for layer in range(model.cfg.n_layers):
    h_m = human_acts[:, layer, :].mean(dim=0)
    a_m = ai_acts[:, layer, :].mean(dim=0)
    d = a_m - h_m
    diff_means_all.append(d / d.norm())

projections_by_alpha = {a: [] for a in alphas}

for alpha in alphas:
    for prompt in test_prompts[:4]:
        tokens = model.to_tokens(prompt, prepend_bos=True)
        try:
            if alpha == 0.0:
                with torch.no_grad():
                    _, cache = model.run_with_cache(tokens,
                        names_filter=lambda name: "resid_post" in name)
            else:
                def make_hook2(a, d):
                    def hook_fn(activation, hook):
                        activation[:, :, :] += a * d
                        return activation
                    return hook_fn
                hook_name = f"blocks.{best_layer}.hook_resid_post"
                with torch.no_grad():
                    with model.hooks(fwd_hooks=[(hook_name, make_hook2(alpha, ai_direction))]):
                        _, cache = model.run_with_cache(tokens,
                            names_filter=lambda name: "resid_post" in name)

            # Get projection at best layer, last token
            act = cache[f"blocks.{best_layer}.hook_resid_post"][0, -1, :]
            proj = (act.float().cpu() @ diff_means_all[best_layer]).item()
            projections_by_alpha[alpha].append(proj)
            del cache
            torch.cuda.empty_cache()
        except:
            pass

print("\nMean projection onto AI direction by steering strength:")
alpha_means = {}
for alpha in sorted(alphas):
    if projections_by_alpha[alpha]:
        mean_proj = np.mean(projections_by_alpha[alpha])
        alpha_means[alpha] = mean_proj
        print(f"  α={alpha:+6.1f}: projection = {mean_proj:.2f}")

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
x = list(alpha_means.keys())
y = list(alpha_means.values())
ax.plot(x, y, 'bo-', markersize=8)
ax.set_xlabel("Steering Strength (α)")
ax.set_ylabel("Mean Projection onto AI Direction")
ax.set_title("Causal Effect of Steering on AI Direction\n(Pythia-2.8B)")
ax.axhline(y=alpha_means.get(0.0, 0), color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "causal_steering_effect.png", dpi=150)
plt.close()
print("Causal steering plot saved.")

# ── Analyze text characteristics under steering ──
print("\n=== Text characteristics under steering ===")

def analyze_text_features(text):
    """Compute simple AI-style features of generated text."""
    words = text.split()
    sentences = text.split('.')

    # AI-style markers
    hedging_words = ['however', 'moreover', 'furthermore', 'additionally',
                     'important', 'note', 'overall', 'essentially',
                     'generally', 'typically', 'particularly']
    formal_markers = ['therefore', 'consequently', 'nevertheless',
                      'nonetheless', 'in conclusion', 'in summary']
    numbered_list = any(f'{i}.' in text or f'{i})' in text for i in range(1, 10))

    hedge_count = sum(1 for w in words if w.lower().strip('.,!?;:') in hedging_words)
    formal_count = sum(1 for w in words if w.lower().strip('.,!?;:') in formal_markers)

    return {
        'word_count': len(words),
        'avg_sentence_len': len(words) / max(len(sentences), 1),
        'hedge_ratio': hedge_count / max(len(words), 1),
        'formal_ratio': formal_count / max(len(words), 1),
        'has_numbered_list': int(numbered_list),
    }

feature_by_alpha = {}
for alpha in alphas:
    features = []
    rows = df[df['alpha'] == alpha]
    for _, row in rows.iterrows():
        if not row['generated'].startswith('ERROR'):
            features.append(analyze_text_features(row['generated']))
    if features:
        feature_df = pd.DataFrame(features)
        feature_by_alpha[alpha] = feature_df.mean().to_dict()

print("\nText features by steering strength:")
for alpha in sorted(feature_by_alpha.keys()):
    f = feature_by_alpha[alpha]
    print(f"  α={alpha:+6.1f}: words={f['word_count']:.0f}, "
          f"sent_len={f['avg_sentence_len']:.1f}, "
          f"hedge={f['hedge_ratio']:.3f}, "
          f"formal={f['formal_ratio']:.3f}")

# Save feature analysis
with open(RESULTS_DIR / "steering_features.json", 'w') as f:
    json.dump(feature_by_alpha, f, indent=2, default=float)

print("\n=== Causal steering experiment complete ===")
