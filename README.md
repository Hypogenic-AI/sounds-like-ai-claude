# Is There a "Sounds Like AI" Direction in the Residual Stream?

Investigating whether the stylistic quality that makes LLM outputs "sound like AI" is linearly represented in the residual stream, following the methodology established for refusal (Arditi et al.), truth (Marks & Tegmark), and sentiment (Turner et al.).

## Key Findings

- **Linear probes achieve 93-96% accuracy** classifying human vs AI text from residual stream activations alone
- **The DiffMean direction achieves 69-81% accuracy** as a single 1D projection, significantly above chance (p < 0.05)
- **Both base (Pythia-2.8B) and instruct-tuned (Qwen2.5-3B-Instruct) models** encode the distinction
- **Cross-domain transfer averages 69-72%** — partial but inconsistent generalization
- **Causal steering produces monotonic projection shifts** along the AI-style axis
- **"AI style" appears multi-dimensional** — the gap between 1D DiffMean (69-81%) and full linear probe (93-96%) suggests a subspace rather than a single direction

## Project Structure

```
├── REPORT.md              # Full research report with all results
├── planning.md            # Research plan and motivation
├── literature_review.md   # Literature review synthesis
├── resources.md           # Resource catalog
├── src/
│   ├── experiment.py      # Main experiment (activation collection)
│   ├── analyze.py         # Pythia-2.8B analysis
│   ├── analyze_qwen.py    # Qwen2.5-3B-Instruct analysis
│   ├── causal_steering.py # Causal intervention experiment
│   ├── cross_domain.py    # Cross-domain generalization
│   └── combined_plots.py  # Combined comparison figures
├── results/
│   ├── summary.json       # Pythia results summary
│   ├── probe_results.csv  # Layer-wise probe accuracies
│   ├── cross_domain_results.csv
│   ├── steering_results.csv
│   ├── plots/             # All visualization plots
│   └── qwen/              # Qwen-specific results
├── datasets/              # HC3 and other datasets
├── papers/                # Research papers (PDFs)
└── code/                  # Reference implementations
```

## How to Reproduce

```bash
# Setup
uv venv && source .venv/bin/activate
uv pip install torch transformers transformer-lens numpy pandas scikit-learn matplotlib seaborn tqdm einops jaxtyping
uv pip install 'transformers<4.50'  # TransformerLens compatibility

# Run experiments
CUDA_VISIBLE_DEVICES=0 python src/experiment.py      # Collect Pythia activations
python src/analyze.py                                 # Analyze Pythia results
CUDA_VISIBLE_DEVICES=1 python src/experiment_qwen.py  # Collect Qwen activations
python src/analyze_qwen.py                            # Analyze Qwen results
python src/causal_steering.py                         # Causal intervention
python src/cross_domain.py                            # Cross-domain transfer
python src/combined_plots.py                          # Combined figures
```

**Requirements**: GPU with 16+ GB VRAM, Python 3.10+

## Full Report

See [REPORT.md](REPORT.md) for detailed methodology, results, analysis, and discussion.
