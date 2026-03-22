# Literature Review: Is There a "Sounds Like AI" Direction in the Residual Stream?

## Research Area Overview

This research sits at the intersection of three active areas: (1) the linear representation hypothesis in LLMs, (2) activation steering and representation engineering, and (3) AI-generated text detection. The core question is whether the stylistic quality that makes text "sound like AI" — characterized by overly formal language, hedging, verbose explanations, numbered lists, and other telltale patterns — is linearly represented in a model's residual stream, analogous to how truth, refusal, and other high-level concepts have been shown to be linearly encoded.

## Key Papers

### 1. Refusal in Language Models Is Mediated by a Single Direction (Arditi et al., 2024)
- **Authors**: Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka, Nina Rimsky, Wes Gurnee, Neel Nanda
- **Source**: COLM 2024 (arXiv: 2406.11717)
- **Key Contribution**: Demonstrates that refusal behavior across 13 chat models (up to 72B parameters) is mediated by a single direction in the residual stream. Erasing this direction disables refusal; adding it elicits refusal on harmless inputs.
- **Methodology**:
  1. Collect residual stream activations on harmful vs harmless instructions
  2. Compute difference-in-means across all layers and positions
  3. Select the direction with highest separation
  4. Validate via ablation (removing the direction) and addition (inserting it)
- **Results**: Single direction ablation achieves jailbreak success across all tested models with minimal capability degradation. Adversarial suffixes work by suppressing this direction.
- **Relevance to Our Research**: This is the most direct template. If refusal is a single direction, "sounds like AI" might also be a single (or few) direction(s). The methodology — contrastive activation collection, difference-in-means, ablation/addition validation — transfers directly.
- **Citations**: 519

### 2. The Geometry of Truth (Marks & Tegmark, 2024)
- **Authors**: Samuel Marks, Max Tegmark
- **Source**: COLM 2024 (arXiv: 2310.06824)
- **Key Contribution**: Shows that LLMs linearly represent the truth/falsehood of factual statements. Introduces mass-mean probing (difference-in-means with covariance correction), which identifies more causally-implicated directions than logistic regression.
- **Methodology**:
  1. Curate true/false statement datasets with controlled diversity
  2. Localize truth representations via activation patching
  3. Visualize with PCA (clear linear separation emerges)
  4. Train probes (LR, mass-mean, CCS) and test cross-dataset generalization
  5. Causal interventions: shift activations along probe direction
- **Key Finding**: Mass-mean probes generalize as well as LR/CCS for classification but identify directions that are MORE causally implicated (NIE of 0.85-0.97 vs 0.13-0.52 for LR on some conditions).
- **Datasets Used**: cities, sp_en_trans, larger_than, companies_true_false, counterfact_true_false, likely
- **Code**: https://github.com/saprmarks/geometry-of-truth
- **Relevance**: The mass-mean probing technique is ideal for our research. We should use difference-in-means (not logistic regression) to find the "sounds like AI" direction, as it better identifies causally-relevant directions.
- **Citations**: 427

### 3. Steering Language Models With Activation Engineering (Turner et al., 2023)
- **Authors**: Alexander Matt Turner et al.
- **Source**: arXiv: 2308.10248
- **Key Contribution**: Introduces Activation Addition (ActAdd) — computing steering vectors from contrastive prompt pairs (e.g., "Love" vs "Hate") and adding them during inference.
- **Results**: Achieves SOTA on negative-to-positive sentiment shift and detoxification. Works with a single pair of data points.
- **Relevance**: Demonstrates that high-level output properties like topic and sentiment can be controlled via simple activation addition. If "AI style" is linearly represented, ActAdd-style steering should work.
- **Citations**: 400

### 4. Steering Llama 2 via Contrastive Activation Addition (Rimsky et al., 2024)
- **Authors**: Nina Rimsky, Nick Gabrieli, Julia Schulz, Meg Tong, Evan Hubinger, Alexander Matt Turner
- **Source**: ACL 2024 (arXiv: 2312.06681)
- **Key Contribution**: Scales up ActAdd with Contrastive Activation Addition (CAA). Averages residual stream differences over many contrastive pairs for more robust steering vectors.
- **Methodology**: Compute steering vector = mean(activations on positive examples) - mean(activations on negative examples). Add at all token positions after user prompt.
- **Results**: Significantly alters model behavior on sycophancy, corrigibility, hallucination, and other behavioral axes.
- **Code**: https://github.com/nrimsky/CAA
- **Relevance**: CAA methodology directly applicable: use human-written text as one class and AI-written text as another to compute "AI style" steering vectors.
- **Citations**: 579

### 5. The Linear Representation Hypothesis and the Geometry of Large Language Models (Park et al., 2023)
- **Authors**: Kiho Park, Yo Joong Choe, Victor Veitch
- **Source**: arXiv: 2311.09421
- **Key Contribution**: Provides formal theoretical grounding for the linear representation hypothesis. Defines linear representation using counterfactuals and proves connections to linear probing and model steering.
- **Key Result**: Identifies a non-Euclidean (causal) inner product that respects language structure, unifying all notions of linear representation.
- **Relevance**: Theoretical framework for understanding what "linear direction for AI style" would mean formally.
- **Citations**: 408

### 6. Sparse Autoencoders Find Highly Interpretable Features in Language Models (Cunningham et al., 2023)
- **Authors**: Hoagy Cunningham, Aidan Ewart, Logan Riggs, Robert Huben, Lee Sharkey
- **Source**: arXiv: 2309.08902
- **Key Contribution**: Demonstrates SAEs can decompose activations into sparse, interpretable features, resolving polysemanticity/superposition.
- **Relevance**: SAEs could identify individual features contributing to "AI style" — e.g., features for hedging, formal language, numbered lists. These might compose into the overall "sounds like AI" direction.
- **Citations**: 948

### 7. Language Models Represent Space and Time (Gurnee & Tegmark, 2023)
- **Authors**: Wes Gurnee, Max Tegmark
- **Source**: arXiv: 2309.08600
- **Key Contribution**: Shows LLMs learn linear representations of spatial and temporal concepts. Identifies individual "space neurons" and "time neurons."
- **Relevance**: Demonstrates that abstract concepts (space, time) are linearly encoded. "AI style" is another abstract property that might follow the same pattern.
- **Citations**: 280

### 8. A Mechanistic Understanding of Alignment Algorithms: DPO and Toxicity (Lee et al., 2024)
- **Authors**: Andrew Lee et al.
- **Source**: arXiv: 2401.01967
- **Key Contribution**: Shows that DPO doesn't remove toxic capabilities but bypasses them. Pre-training representations persist but are routed around.
- **Relevance**: Suggests that "AI style" may be an artifact of alignment/RLHF training that overlays pre-training representations. The direction might emerge during fine-tuning.
- **Citations**: 166

### 9. Improving Activation Steering with Mean-Centring (Jorgensen et al., 2023)
- **Authors**: Ole Jorgensen et al.
- **Source**: arXiv: 2310.15154
- **Key Contribution**: Shows that mean-centring (subtracting the overall mean from steering vectors) significantly improves effectiveness.
- **Methodology**: steering_vector = mean(target_activations) - mean(all_activations)
- **Relevance**: Mean-centring should be applied when computing the "sounds like AI" direction to improve signal quality.
- **Citations**: 59

### 10. A Single Direction of Truth (O'Neill et al., 2025)
- **Authors**: Charles O'Neill et al.
- **Source**: arXiv: 2503.05858
- **Key Contribution**: Shows a generator-agnostic observer model can detect hallucinations via a linear probe on its residual stream. Manipulating this direction causally steers hallucination rates.
- **Relevance**: Demonstrates the observer-model approach — using one model to probe another's outputs. Relevant if we want to detect "AI style" from an observer model's perspective.
- **Citations**: 4

## Common Methodologies

### Direction-Finding Approaches
1. **Difference-in-Means (DiffMean)**: Compute mean activations for each class, take the difference. Simple, optimization-free, and highly causal. Used in Marks & Tegmark, Arditi et al.
2. **Contrastive Activation Addition (CAA)**: Average activation differences over many contrastive pairs. More robust than single-pair methods. Used in Rimsky et al.
3. **Linear Probing (Logistic Regression)**: Train a linear classifier. Good for classification but identifies less causally-relevant directions than DiffMean.
4. **PCA**: Visualize top principal components of centered activations. Useful for confirming linear structure exists.
5. **Sparse Autoencoders**: Decompose activations into sparse features. Can identify individual contributing features.

### Validation Approaches
1. **Ablation**: Remove the direction from residual stream; check if the behavior disappears.
2. **Addition**: Add the direction; check if the behavior is induced.
3. **Cross-dataset transfer**: Train probe on one dataset, test on another.
4. **Normalized Indirect Effect (NIE)**: Quantify how much an intervention changes model outputs.

## Standard Baselines
- **Random direction**: Ablating/adding a random direction should have no systematic effect
- **PCA-based direction**: Top PC of combined data
- **Logistic regression probe**: Standard classification approach
- **Few-shot prompting**: Behavioral baseline without activation intervention

## Evaluation Metrics
- **Classification accuracy**: Can a linear probe trained on the direction classify human vs. AI text?
- **Cross-dataset generalization**: Does the direction transfer across domains/genres?
- **Normalized Indirect Effect (NIE)**: Does intervening along the direction causally change model behavior?
- **Cosine similarity**: How aligned are directions found from different datasets/layers?
- **Steering effectiveness**: Does adding/removing the direction make outputs more/less AI-like?

## Datasets in the Literature
- **True/false datasets**: cities, sp_en_trans, larger_than (Marks & Tegmark)
- **Harmful/harmless instructions**: AdvBench, JailbreakBench (Arditi et al.)
- **Behavioral multiple-choice**: sycophancy, corrigibility, etc. (Rimsky et al.)
- **HC3**: Human vs. ChatGPT paired responses (for AI detection research)
- **TuringBench**: Multi-generator human vs. AI text

## Gaps and Opportunities

1. **No prior work directly studies "sounds like AI" as a linear direction.** While truth, refusal, sentiment, toxicity, and hallucination have all been studied as linear features, the specific stylistic quality of "sounding AI-generated" has not been investigated from a mechanistic interpretability perspective.

2. **AI detection research focuses on output-level features, not internal representations.** Most AI text detection work uses surface-level stylometric features or output probabilities, not residual stream directions.

3. **The relationship between "AI style" and alignment is unexplored.** "Sounding like AI" likely emerges during RLHF/alignment training. Understanding this mechanistically could inform better alignment techniques.

4. **Multi-model generalizability is untested.** Does the "sounds like AI" direction transfer across model families, or is it model-specific?

## Recommendations for Our Experiment

### Recommended Approach
1. **Use HC3 dataset** as primary data source (paired human/ChatGPT responses)
2. **Process texts through a target LLM** (e.g., Gemma-2-2B, Llama-3.1-8B) using TransformerLens
3. **Collect residual stream activations** at all layers for the last token position
4. **Compute difference-in-means direction** (AI activations - human activations) at each layer
5. **Validate with PCA visualization** — look for linear separation
6. **Test causal role** — add/remove direction and measure effect on output style
7. **Test generalization** — train on one domain, test on others

### Recommended Models
- **Gemma-2-2B**: Small enough for rapid iteration, well-supported by TransformerLens
- **Llama-3.1-8B-Instruct**: Mid-size model with strong "AI style"
- **Pythia family**: For scale analysis (70M to 12B)

### Recommended Metrics
- **Probe accuracy**: Linear probe classification of human vs AI text
- **NIE**: Causal effect of direction intervention
- **Perplexity-based AI detection**: Does steering change detectability?
- **Human evaluation**: Does steering make text sound more/less AI-like?

### Key Methodological Considerations
- **Use difference-in-means, not logistic regression** — DiffMean finds more causally relevant directions
- **Apply mean-centring** to improve steering vector quality
- **Test at multiple layers** — the direction likely emerges at specific layers
- **Control for content** — ensure the direction captures style, not topic
- **Include diverse text types** — test generalization across domains
