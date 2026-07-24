---
language:
- tt
license: cc-by-nc-sa-4.0
library_name: transformers
pipeline_tag: text-generation
tags:
- tatar
- turkic
- qypchaq
- from-scratch
- literary
- base-model
---

# TATlit — a native literary Tatar base language model

TATlit is a 478M-parameter, from-scratch base language model for literary Tatar (ISO 639-3 `tat`), a Qypchaq Turkic language written in Cyrillic. It is trained without adapting any English- or Russian-dominant base, so that it produces native Tatar morphosyntax by default rather than acting as a translation layer over a dominant language. This is a base model, not instruction-tuned.

## Usage

```python
# pip install sentencepiece
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("ilchats/TATlit")
tok   = AutoTokenizer.from_pretrained("ilchats/TATlit", use_fast=False)
```

The tokenizer is SentencePiece-based; pass `use_fast=False` so tokenization matches training exactly (the auto-converted fast tokenizer differs on the leading token). The weights are a standard Llama-architecture checkpoint in safetensors.

## Model
- Llama-style decoder, 24 layers, hidden size 1280, 16 attention heads with 4-way grouped-query attention, SwiGLU, RoPE, RMSNorm, tied embeddings, context 4096.
- 48K SentencePiece Unigram tokenizer, Tatar fertility 4.2-4.5 chars/token, 1.5-3x more efficient than general tokenizers.
- 478M parameters, bf16.

## Training
Two stages. Stage 1 pretrains on a Cyrillic-Qypchaq pool (Tatar 43 / Kazakh 30 / Kyrgyz 16 / Bashkir 10 / smaller siblings ~1 percent) under target-capped sampling. Stage 2 specializes on pure, register-weighted literary Tatar with an anti-memorization regime that keeps verbatim reproduction low.

## Results

TATlit is compared against three Tatar-specialized baselines (Goldfish-tat 125M, Tweety-7B-tatar, mGPT-1.3B-tatar) and four frontier models from 31B to 120B (Gemma-4-31B, Llama-4-Scout, Qwen3-32B, gpt-oss-120b), each measured with the same harness per benchmark.

**Byte-per-byte on external public sets** (FLORES, BOUQuET, UD), lower is better.

| Model | Params | FLORES | BOUQuET | UD |
|---|---|---|---|---|
| **TATlit** | **478M** | **.740** | **.777** | **.502** |
| Tweety-7B | 7B | .757 | .800 | .544 |
| Gemma-4-31B | 31B | .772 | .939 | .691 |
| Goldfish-tat | 125M | .778 | .801 | **.502** |
| mGPT-1.3B | 1.3B | .950 | 1.186 | .880 |
| gpt-oss-120b | 120B | 1.092 | 1.164 | 1.099 |
| Qwen3-32B | 32B | 1.144 | 1.423 | 1.159 |
| Llama-4-Scout | 109B | 1.222 | 1.498 | 1.080 |

**Byte-per-byte on in-house held-out sets** (literary, periodical), lower is better.

| Model | Params | Held-out lit | Held-out per |
|---|---|---|---|
| **TATlit** | **478M** | **.732** | **.701** |
| Gemma-4-31B | 31B | .790 | .705 |
| Tweety-7B | 7B | .810 | .778 |
| Goldfish-tat | 125M | .929 | .874 |
| Llama-4-Scout | 109B | 1.084 | .995 |
| Qwen3-32B | 32B | 1.105 | .979 |
| mGPT-1.3B | 1.3B | 1.199 | 1.094 |
| gpt-oss-120b | 120B | 2.181 | 1.857 |

**TatBLiMP, morphological acceptability**, higher is better.

| Model | Params | acc | acc_norm |
|---|---|---|---|
| **TATlit** | **478M** | **.975** | **.958** |
| Goldfish-tat | 125M | .974 | **.958** |
| Tweety-7B | 7B | .956 | .915 |
| Gemma-4-31B | 31B | .924 | .839 |
| Llama-4-Scout | 109B | .889 | .806 |
| Qwen3-32B | 32B | .811 | .708 |
| gpt-oss-120b | 120B | .803 | .677 |
| mGPT-1.3B | 1.3B | .736 | .639 |

On the school-knowledge benchmark TUMLU-mini the model sits near chance (0.297 against a 0.25 random baseline), which is expected for a base model with no encyclopedic training.

## Limitations
Base model, no instruction following, no encyclopedic knowledge. Softest axis is the person/possessive system inside izafet.

## Authors
Ilshat Saetov, Dmitry Gaynullin.

## Data
The training corpus is withheld on stewardship grounds for a minoritized language's literary heritage; a full datasheet with per-document provenance is available on reasonable request. The tokenizer and the model weights are released.

## License
The model weights and the tokenizer are released under **CC BY-NC-SA 4.0**. The accompanying [TatBLiMP](https://github.com/ilchats/TatBLiMP) benchmark is released under **CC BY-NC 4.0**. The training corpus is not released.

## Citation

```bibtex
@misc{saetov2026tatlit,
  title  = {Thanks to the Siblings: Bootstrapping a Native Tatar Language Model from a Cyrillic Qypchaq Pool},
  author = {Saetov Ilshat, Gaynullin Dmitry},
  year   = {2026}
}
```
