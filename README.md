# Evaluating Isotropy in Modern Masked Language Models

This repository contains the code and experiments for measuring isotropy in the static vocabulary embedding and output matrices of modern MLM encoders, as described in the paper.

## ToC

1. [Cite](#cite)
1. [Project Structure](#project-structure)
1. [Models](#models)
1. [Installation](#installation)
1. [Running the Analysis](#running-the-analysis)
1. [Reference Implementation](#reference-implementation)
1. [Results Format](#results-format)

## Cite

- the paper is currently **in press** ([ERAMIA-RS 2026](https://sol.sbc.org.br/index.php/eramia-rs)).

```bibtex
@inproceedings{madeira2026isotropy,
  title={Evaluating Isotropy in Modern Masked Language Models},
  author={Madeira, Caio and Griebler, Dalvan Jair},
  booktitle={Anais da Escola Regional de Aprendizado de M{\'a}quina e Intelig{\^e}ncia Artificial do Rio Grande do Sul},
  pages={},
  year={in press},
  organization={SBC}
}
```

## Project Structure

- `main.py`: Entry point. Loads each embedding matrix, computes every metric reported in the paper, and writes a timestamped log.
- `plots.py`: SVD projections of the embedding cloud (scatter and log-density hexbin), saved to `figs/`.
- `machina/src/`: Reference implementation of [Machina and Mercer 2024](https://aclanthology.org/2024.naacl-long.274/), used to cross-check our estimator. Optional — see [Reference Implementation](#reference-implementation).
- `notebooks/`: Per-model exploratory notebooks (one per model), kept for the record.
- `figs/`, `logs/`: Created on the first run; see [Results Format](#results-format).

## Models

All matrices are read straight from each checkpoint's `model.safetensors`. No model is loaded or run — only its vocabulary parameters are read.

| Model | Hugging Face ID | V | d | Weight tying |
| --- | --- | --- | --- | --- |
| ModernBERT-base | [`answerdotai/ModernBERT-base`](https://huggingface.co/answerdotai/ModernBERT-base) | 50,368 | 768 | tied |
| mmBERT-base | [`jhu-clsp/mmBERT-base`](https://huggingface.co/jhu-clsp/mmBERT-base) | 256,000 | 768 | tied |
| NeoBERT | [`chandar-lab/NeoBERT`](https://huggingface.co/chandar-lab/NeoBERT) | 30,522 | 768 | **untied** |
| RoBERTa-base | [`FacebookAI/roberta-base`](https://huggingface.co/FacebookAI/roberta-base) | 50,265 | 768 | tied |
| BERT-base-uncased | [`google-bert/bert-base-uncased`](https://huggingface.co/google-bert/bert-base-uncased) | 30,522 | 768 | tied |
| BERT-base-cased | [`google-bert/bert-base-cased`](https://huggingface.co/google-bert/bert-base-cased) | 28,996 | 768 | tied |

NeoBERT is the only untied model, so it is the only one whose input and output matrices can be measured separately. `main.py` detects it by name and additionally reads `decoder.weight` through `AutoModelForMaskedLM`.

The input matrix lives under a different key in each family; `get_tensor_name()` resolves it automatically:

| Family | Tensor key |
| --- | --- |
| ModernBERT, mmBERT | `embeddings.tok_embeddings.weight` (or `model.`-prefixed) |
| NeoBERT | `encoder.weight` |
| BERT, RoBERTa | `embeddings.word_embeddings.weight` |

## Installation

Install the required dependencies using pip:

```shell
pip install -r requirements.txt
```

Download the checkpoints you want to measure. Only `model.safetensors` is needed (plus the config for NeoBERT's output matrix):

```shell
huggingface-cli download answerdotai/ModernBERT-base --local-dir models/modernbert
huggingface-cli download jhu-clsp/mmBERT-base        --local-dir models/mmbert
huggingface-cli download chandar-lab/NeoBERT         --local-dir models/neobert
huggingface-cli download FacebookAI/roberta-base     --local-dir models/roberta
```

## Running the Analysis

### Configuration

Point `MODERN_MODELS` and `OLD_MODELS` in `main.py` at your local checkpoints. Each entry needs the path to the `.safetensors` file and the Hugging Face ID (the ID is only used to pull NeoBERT's output matrix):

```python
MODERN_MODELS = {
    "modernbert_base": {
        "path": "models/modernbert/model.safetensors",
        "model_id": "answerdotai/ModernBERT-base",
    },
    # ...
}
```

### Execution

Run the script from the root directory:

```shell
python main.py
```

For every matrix, the script reports:

- shape, vocabulary size `V`, dimension `d`, parameter count;
- mean row norm `||w||` and its standard deviation;
- centroid norm `||w̄||` and the relative displacement `||w̄||/||w||`, which is invariant to the scale of `W`;
- the mean pairwise cosine, computed in `O(V·d)` instead of `O(V²·d)` by summing the normalized rows first;
- `I(W)` on the raw matrix, on the mean-centered matrix `Ŵ = W - w̄`, and on a random cloud with the same row norms and uniform directions (the baseline reported as `I(W_ref)`);
- `I(W)` under both sign conventions: over the `d` eigenvectors of `WᵀW` as returned by `eigh` (the convention reported in the paper, matching Machina and Mercer), and over all `2d` directions `±v` (deterministic, since eigenvector signs are arbitrary). The second is always the lower of the two.

Matrices are cast to `float64` before any exponential. All partition functions use the log-sum-exp form `log Σ exp(sᵢ) = m + log Σ exp(sᵢ - m)`, so no overflow occurs at any vocabulary size.

## Reference Implementation

To reproduce the cross-check reported in the paper, place the code released by Machina and Mercer (2024) under `machina/src/` so that `partition_function.py` and `cosine_improvement.py` are importable:

```shell
machina/
└── src/
    ├── partition_function.py
    └── cosine_improvement.py
```

With it present, `main.py` additionally runs their `compute_IW` in both `float32` and `float64`, their `fast_cosine` over the full vocabulary, and their `fast_cosine`/`slow_cosine` on a 10,000-token sample, printing the agreement with our estimator. Without it, the script prints the missing import and skips only that section.

## Results Format

Both directories are created on the first run.

### Logs

Every run writes the complete stdout and stderr to `logs/run_<YYYYmmdd>_<HHMMSS>.log`, so each execution is a self-contained record of the numbers reported in the paper.

### Figures

Each matrix produces two SVD projections onto the top two eigenvectors of `WᵀW`, saved to `figs/`:

- `figs/<model>_input.png`: scatter projection, with the axes symmetric around the origin so that a displaced cloud is visible as displacement rather than being re-centered by the plot.
- `figs/<model>_input_hexbin.png`: log-density hexbin of the same projection, with the origin marked. The fraction of variance covered by the two axes is printed to the log.

Both are produced for the raw matrix and for the centered matrix `Ŵ`, since centering removes exactly the common translation that `I(W)` is sensitive to.

### Notes:
Evidence of RoBERTa x mmBERT scalar:
```
# I(alpha * W) mesma matriz porem multiplicada a um escalar
_, eigenvectors = np.linalg.eigh(W.T @ W)
projections = W @ eigenvectors 

for alpha in [1, 2.49, 5]:
    scaled_projs = alpha * projections 
    maxi = scaled_projs.max(axis=0)
    log_Z = maxi + np.log(np.exp(scaled_projs - maxi).sum(axis=0))  # log-sum-exp
    I = np.exp(log_Z.min() - log_Z.max())
    print(f"alpha = {alpha:5.2f}   norma media = {alpha * norms.mean():.3f}   I(alpha*W) = {I:.4f}")
```
