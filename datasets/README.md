# Downloaded Datasets

This directory contains datasets for the research project: "Is there a 'sounds like AI' direction in the residual stream?"

Data files are NOT committed to git due to size. Follow the download instructions below.

## Dataset 1: HC3 (Human ChatGPT Comparison Corpus)

### Overview
- **Source**: `Hello-SimpleAI/HC3` on HuggingFace
- **Size**: 24,322 question-answer pairs (~72MB JSONL)
- **Format**: JSONL with fields: question, human_answers, chatgpt_answers, index, source
- **Task**: Paired human vs. ChatGPT responses to the same questions
- **Domains**: Open-domain QA, finance, medicine, law, psychology (ELI5, WikiQA, etc.)
- **License**: CC-BY-SA-4.0

### Download Instructions

```bash
# Direct download (JSONL format)
wget https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/main/all.jsonl -O datasets/hc3/all.jsonl
```

### Loading
```python
import json
data = [json.loads(line) for line in open("datasets/hc3/all.jsonl")]
# Each row has: question, human_answers (list), chatgpt_answers (list)
```

### Why This Dataset
Provides directly paired human vs. ChatGPT responses to identical questions. Ideal for:
- Computing contrastive activation vectors (human response activations vs AI response activations)
- Training difference-in-mean probes for "sounds like AI" direction
- Controlling for content/topic (same question, different style)

---

## Dataset 2: AI Text Detection Pile

### Overview
- **Source**: `artem9k/ai-text-detection-pile` on HuggingFace
- **Size**: ~5000 samples downloaded (full dataset larger)
- **Format**: CSV with fields: source, id, text
- **Task**: Binary classification (human vs AI-generated text)
- **Genres**: Essays, long-form text from GPT-2, GPT-3, ChatGPT, GPTJ, human

### Download Instructions
```python
from datasets import load_dataset
ds = load_dataset("artem9k/ai-text-detection-pile", split="train")
```

### Why This Dataset
Multi-generator dataset useful for testing whether the "sounds like AI" direction generalizes across different AI models (GPT-2, GPT-3, ChatGPT, GPTJ).

---

## Dataset 3: Anthropic HH-RLHF

### Overview
- **Source**: `Anthropic/hh-rlhf` on HuggingFace
- **Size**: 169,352 preference pairs (~79MB); 3000 sampled locally
- **Format**: JSON with chosen/rejected conversation pairs
- **Task**: Human preference between assistant responses
- **License**: MIT

### Download Instructions
```python
from datasets import load_dataset
ds = load_dataset("Anthropic/hh-rlhf", split="train")
```

### Why This Dataset
Contains preference annotations between different assistant responses. The "chosen" vs "rejected" signal may correlate with stylistic qualities. Useful for understanding what makes text sound more or less AI-like from a human preference perspective.

---

## Additional Recommended Datasets (Not Downloaded)

### LMSYS Chatbot Arena (Gated - requires authentication)
- **Source**: `lmsys/chatbot_arena_conversations`
- 33K+ conversations from 20+ LLMs with human preference votes
- Requires HuggingFace authentication

### DeepfakeTextDetect
- **Source**: `yaful/DeepfakeTextDetect`
- 447K samples from 27 LLMs + human text
- Currently requires older datasets library version

### M4 (Multi-generator, Multi-domain)
- **Source**: `mbzuai-nlp/M4` (GitHub)
- 200K+ samples from 6 LLMs across multiple domains
- EACL 2024 Best Resource Paper

### TuringBench
- **Source**: turingbench.ist.psu.edu
- 200K samples from 19 generators + human text

## Notes
- HC3 is the primary dataset for initial experiments (paired human/AI responses)
- The AI text detection pile provides multi-generator diversity
- For the actual experiment, we need to process these texts through a target LLM and collect residual stream activations
