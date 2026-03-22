# Cloned Repositories

## Repo 1: Geometry of Truth
- **URL**: https://github.com/saprmarks/geometry-of-truth
- **Purpose**: Implementation of "The Geometry of Truth" (Marks & Tegmark, 2024). Demonstrates linear representations of truth/falsehood in LLM residual streams using difference-in-mean probes.
- **Location**: `code/geometry-of-truth/`
- **Key files**:
  - `dataexplorer/`: Interactive visualization tools
  - `datasets/`: Curated true/false statement datasets
  - Scripts for probing, PCA visualization, and causal intervention
- **Relevance**: Direct methodological template. The same approach (difference-in-mean probes, PCA visualization, causal interventions) can be applied to find a "sounds like AI" direction instead of a "truth" direction.

## Repo 2: Refusal Direction
- **URL**: https://github.com/andyrdt/refusal_direction
- **Purpose**: Implementation of "Refusal in Language Models Is Mediated by a Single Direction" (Arditi et al., 2024). Shows refusal behavior is encoded as a single linear direction in the residual stream.
- **Location**: `code/refusal_direction/`
- **Key files**:
  - `pipeline/`: End-to-end pipeline for finding and ablating the refusal direction
  - `pipeline/model_utils/`: Model loading and activation extraction utilities
  - `pipeline/submodules/`: Evaluation, generation, and direction-finding code
- **Relevance**: Most direct template for our research. The methodology of finding a single direction that mediates a behavioral property (refusal → "sounds like AI") is exactly what we want to replicate.

## Repo 3: Contrastive Activation Addition (CAA)
- **URL**: https://github.com/nrimsky/CAA
- **Purpose**: Implementation of "Steering Llama 2 via Contrastive Activation Addition" (Rimsky et al., 2024). Computes steering vectors from contrastive pairs and adds them during inference.
- **Location**: `code/CAA/`
- **Key files**:
  - `llama_wrapper.py`: Wrapper for hooking into LLaMA residual stream
  - `prompting_with_steering.py`: Applying steering vectors at inference time
  - Various evaluation and analysis scripts
- **Relevance**: Provides practical infrastructure for computing contrastive steering vectors and applying them. Could be adapted to steer models toward/away from "AI-sounding" output.

## Additional Recommended Repositories (Not Cloned)

### TransformerLens
- **URL**: https://github.com/TransformerLensOrg/TransformerLens
- **Purpose**: Standard library for mechanistic interpretability. Provides hook-based access to all intermediate activations.
- **Install**: `pip install transformer-lens`

### SAELens
- **URL**: https://github.com/decoderesearch/SAELens
- **Purpose**: Library for training and analyzing Sparse Autoencoders on LLM activations.
- **Install**: `pip install sae-lens`

### steering-vectors
- **URL**: https://github.com/steering-vectors/steering-vectors
- **Purpose**: General-purpose library for training and applying steering vectors to HuggingFace models.
- **Install**: `pip install steering-vectors`
