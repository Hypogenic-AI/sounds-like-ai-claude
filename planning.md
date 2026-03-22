# Research Plan: Is There a "Sounds Like AI" Direction in the Residual Stream?

## Motivation & Novelty Assessment

### Why This Research Matters
LLMs produce text with a distinctive "AI style" — formal, hedging, verbose, with numbered lists and diplomatic phrasing. If this style is linearly encoded as a direction in the residual stream (like truth, refusal, and sentiment have been shown to be), it would mean: (1) AI-sounding output is a controllable feature, not an emergent artifact; (2) models could be steered to sound more or less "AI-like" without retraining; (3) AI text detection could leverage internal representations rather than surface features.

### Gap in Existing Work
The linear representation hypothesis has been validated for truth (Marks & Tegmark), refusal (Arditi et al.), sentiment (Turner et al.), and hallucination (O'Neill et al.) — but nobody has investigated whether "sounding like AI" is one of these linear features. AI text detection research focuses on surface-level stylometric features or output probabilities, not internal model representations. This is a clear gap at the intersection of mechanistic interpretability and AI-generated text detection.

### Our Novel Contribution
We are the first to directly test whether "sounding like AI" is linearly represented in the residual stream, using established contrastive activation analysis methods adapted for stylistic rather than factual/behavioral properties.

### Experiment Justification
- **Experiment 1 (Direction Finding)**: Use difference-in-means on HC3 paired data to find the candidate "AI style" direction at each layer. Necessary to establish whether the direction exists at all.
- **Experiment 2 (Probe Validation)**: Train linear probes to classify human vs AI text from activations. Tests whether the separation is real and measurable.
- **Experiment 3 (Layer Analysis)**: Analyze which layers encode the direction most strongly. Identifies where in the model the concept is represented.
- **Experiment 4 (Causal Validation)**: Ablate/add the direction and measure if model outputs change style. Tests whether the direction is causally relevant, not just correlated.
- **Experiment 5 (Generalization)**: Test cross-domain transfer (train on reddit_eli5, test on other HC3 domains). Tests whether we found a general "AI style" direction vs domain-specific artifact.

## Research Question
Does there exist a linear direction in the residual stream of an LLM that corresponds to text "sounding like AI," and is this direction causally implicated in producing AI-style outputs?

## Hypothesis Decomposition
1. **H1 (Existence)**: Human-written and AI-written text produce linearly separable activations in the residual stream.
2. **H2 (Specificity)**: The separating direction is concentrated in specific layers (likely middle-to-late layers, by analogy with truth/refusal).
3. **H3 (Causality)**: Intervening on this direction changes model output style (adding it makes output more AI-like; removing it makes output more human-like).
4. **H4 (Generality)**: The direction generalizes across text domains/topics.

## Proposed Methodology

### Approach
Follow the established contrastive activation analysis pipeline from Arditi et al. and Marks & Tegmark, adapted for "AI style":
1. Process paired human/AI texts through a model
2. Collect residual stream activations
3. Compute difference-in-means direction
4. Validate via probes, PCA, and causal interventions

### Model
**Gemma-2-2B** via TransformerLens — small enough for rapid iteration, well-supported, and exhibits clear "AI style" in its outputs.

### Dataset
**HC3** (Hello-SimpleAI) — ~24K paired human/ChatGPT QA responses. Same question with multiple human and ChatGPT answers. The content-controlled pairing isolates stylistic differences.

### Baselines
- Random direction (null baseline)
- PCA top component
- Logistic regression probe direction (compared to DiffMean)

### Evaluation Metrics
- Linear probe accuracy (human vs AI classification)
- Cosine similarity of directions across layers
- PCA visualization (qualitative linear separation)
- Generation quality change under intervention (measured by perplexity and AI-detection classifier)

### Statistical Analysis Plan
- Bootstrap confidence intervals for probe accuracy
- Permutation test for direction significance (compare to random directions)
- Cohen's d for effect sizes

## Expected Outcomes
- **Supporting H1**: Probe accuracy > 85% at best layer
- **Supporting H2**: Clear peak in probe accuracy at specific layers
- **Supporting H3**: Measurable style change under intervention
- **Supporting H4**: Cross-domain probe accuracy > 75%
- **Refuting**: Probe accuracy near chance (50%), no causal effect

## Timeline
1. Environment setup & data prep: 15 min
2. Activation collection: 30 min
3. Direction finding & probes: 20 min
4. Causal validation: 30 min
5. Cross-domain analysis: 15 min
6. Documentation: 20 min

## Potential Challenges
- Model may need significant GPU memory for activation storage → use batching
- HC3 only has ChatGPT text → limited generator diversity
- "AI style" may be multi-dimensional rather than a single direction → test with multiple directions
