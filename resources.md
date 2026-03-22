# Resources Catalog

## Summary
This document catalogs all resources gathered for the research project: **"Is there a 'sounds like AI' direction in the residual stream?"**

The research investigates whether LLMs have a linear direction in their residual stream activations that corresponds to text that "sounds like AI" — the distinctive style characterized by formal language, hedging, verbose explanations, and other patterns typical of AI-generated text.

## Papers
Total papers downloaded: **24**

| # | Title | Authors | Year | File | Key Info |
|---|-------|---------|------|------|----------|
| 1 | Refusal Mediated by Single Direction | Arditi et al. | 2024 | arditi_2024_refusal_single_direction.pdf | Template methodology; 519 citations |
| 2 | Geometry of Truth | Marks & Tegmark | 2024 | marks_2023_geometry_of_truth.pdf | Mass-mean probing; 427 citations |
| 3 | Activation Engineering (ActAdd) | Turner et al. | 2023 | turner_2023_activation_engineering.pdf | Steering method; 400 citations |
| 4 | Contrastive Activation Addition | Rimsky et al. | 2024 | rimsky_2023_contrastive_activation_addition.pdf | CAA method; 579 citations |
| 5 | Linear Representation Hypothesis | Park et al. | 2023 | park_2023_linear_representation_hypothesis.pdf | Theoretical foundation; 408 citations |
| 6 | Space and Time Representations | Gurnee & Tegmark | 2023 | gurnee_2023_space_and_time.pdf | Linear space/time; 280 citations |
| 7 | Sparse Autoencoders | Cunningham et al. | 2023 | cunningham_2023_sparse_autoencoders.pdf | SAE methodology; 948 citations |
| 8 | Gated SAEs | Rajamanoharan et al. | 2024 | gated_sae_2024.pdf | Improved SAEs |
| 9 | Multi-Layer SAEs | Lawson et al. | 2024 | lawson_2024_multilayer_sae.pdf | Cross-layer analysis |
| 10 | AbsTopK SAEs | Zhu et al. | 2025 | zhu_2025_abstopk_sae.pdf | Bidirectional features |
| 11 | Mean-Centring Steering | Jorgensen et al. | 2023 | jorgensen_2023_mean_centring.pdf | Mean-centring improvement |
| 12 | Preference via Residual Steering | La Cava & Tagarelli | 2025 | lacava_2025_preference_residual_steering.pdf | Preference alignment |
| 13 | Refusal Feature Adversarial | Yu et al. | 2024 | yu_2024_refusal_feature_adversarial.pdf | Refusal as attack surface |
| 14 | Geometry of Refusal | Wollschlager et al. | 2025 | wollschlager_2025_geometry_refusal.pdf | Multiple refusal directions |
| 15 | Hidden Dimensions Alignment | Pan et al. | 2025 | pan_2025_hidden_dimensions_alignment.pdf | Multi-dim safety |
| 16 | Dissecting LLM Refusal | Prakash et al. | 2025 | prakash_2025_beyond_sorry.pdf | SAE refusal analysis |
| 17 | Mechanistic DPO/Toxicity | Lee et al. | 2024 | lee_2024_mechanistic_alignment_dpo.pdf | Alignment mechanics |
| 18 | Belief State Geometry | Shai et al. | 2024 | shai_2024_belief_state_geometry.pdf | Belief state encoding |
| 19 | Gaussian Concept Subspace | Zhao et al. | 2024 | zhao_2024_gaussian_concept_subspace.pdf | Concept subspaces |
| 20 | Single Direction of Truth | O'Neill et al. | 2025 | oneill_2025_single_direction_truth.pdf | Hallucination direction |
| 21 | Restricted Embeddings Detection | Various | 2024 | restricted_embeddings_ai_detection.pdf | AI detection embeddings |
| 22 | StyloAI | Various | 2024 | styloai_2024.pdf | Stylometric AI detection |
| 23 | Reliable AI Detection? | Sadasivan et al. | 2023 | sadasivan_2023_reliable_detection.pdf | Detection limits |
| 24 | DetectGPT | Mitchell et al. | 2023 | mitchell_2023_detectgpt.pdf | Zero-shot detection |

See `papers/README.md` for detailed descriptions.

## Datasets
Total datasets downloaded: **3** (with samples/subsets)

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| HC3 | Hello-SimpleAI/HC3 | 24K pairs, 72MB | Human vs ChatGPT paired QA | datasets/hc3/ | Primary dataset |
| AI Text Detection Pile | artem9k/ai-text-detection-pile | 5K sample | Human vs multi-model AI text | datasets/ai_text_detection/ | Multi-generator |
| Anthropic HH-RLHF | Anthropic/hh-rlhf | 3K sample | Preference pairs | datasets/anthropic_hh/ | Preference signal |

See `datasets/README.md` for detailed descriptions and download instructions.

## Code Repositories
Total repositories cloned: **3**

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| geometry-of-truth | github.com/saprmarks/geometry-of-truth | Truth direction probing | code/geometry-of-truth/ | Key template |
| refusal_direction | github.com/andyrdt/refusal_direction | Refusal direction finding | code/refusal_direction/ | Most direct template |
| CAA | github.com/nrimsky/CAA | Contrastive steering | code/CAA/ | Steering infrastructure |

See `code/README.md` for detailed descriptions.

## Resource Gathering Notes

### Search Strategy
- Used paper-finder service (diligent mode) with queries focused on: residual stream directions, linear representations, activation steering, AI text style
- Supplemented with targeted searches for AI text detection datasets and mechanistic interpretability tools
- Prioritized papers with high citation counts and direct methodological relevance

### Selection Criteria
1. **Papers**: Prioritized work demonstrating linear directions for specific behavioral properties (truth, refusal, sentiment) as methodological templates, plus AI text detection work
2. **Datasets**: Focused on paired human/AI text datasets where content is controlled (same question/topic)
3. **Code**: Selected repos with reusable infrastructure for activation extraction, probing, and steering

### Challenges Encountered
- HC3 HuggingFace loading script deprecated; used direct JSONL download instead
- DeepfakeTextDetect dataset also uses deprecated loading scripts
- LMSYS Chatbot Arena dataset is gated (requires authentication)
- The Arditi refusal direction paper PDF chunking had issues; relied on abstract and paper search results for detailed notes

### Gaps and Workarounds
- **No existing "sounds like AI" direction work**: This is genuinely novel — no prior work directly studies this
- **Limited paired datasets controlling for content**: HC3 is best but only covers ChatGPT; may need to generate paired data using multiple models
- **Evaluation challenge**: No established metric for "how AI-like does this sound" — may need human evaluation or proxy metrics

## Recommendations for Experiment Design

### 1. Primary Dataset
**HC3** — Use paired human/ChatGPT responses to compute contrastive directions. The same-question pairing controls for topic, isolating stylistic differences.

### 2. Baseline Methods
- **Difference-in-Means (DiffMean)**: Primary direction-finding method (proven most causally relevant by Marks & Tegmark)
- **PCA**: For visualization and confirming linear structure
- **Logistic Regression probe**: For comparison baseline
- **Random direction**: Null baseline

### 3. Evaluation Metrics
- **Linear probe accuracy** (human vs AI classification from activations)
- **Normalized Indirect Effect (NIE)** of direction ablation/addition
- **Cross-domain generalization** (train on QA, test on academic text)
- **Steering effectiveness** (does adding direction make output more AI-like?)

### 4. Code to Adapt/Reuse
- **refusal_direction pipeline**: Adapt for "AI style" instead of refusal. Replace harmful/harmless inputs with AI-written/human-written inputs.
- **geometry-of-truth probing**: Use mass-mean probing code for finding and validating directions.
- **TransformerLens**: For activation extraction (install separately).

### 5. Suggested Experiment Flow
```
1. Load model via TransformerLens (start with Gemma-2-2B or Llama-3.1-8B)
2. Process HC3 data: run human answers and ChatGPT answers through model
3. Collect residual stream activations at each layer (last token position)
4. Compute DiffMean direction at each layer
5. Visualize with PCA — confirm linear separation
6. Train linear probes — measure classification accuracy
7. Test cross-domain generalization
8. Causal interventions — ablate/add direction, measure effect on output style
9. (Optional) Use SAEs to decompose the direction into interpretable features
```
