# Spectral Clipping

Code accompanying the paper [*Gradient Clipping Beyond Vector Norms: A Spectral Approach for Matrix-Valued Parameters*](https://arxiv.org/abs/2605.11838).

Instead of rescaling a gradient by its $\ell_2$ norm, we clamp every singular value of the gradient matrix above a threshold $\tau$ down to $\tau$. The repo has three threshold strategies and five experiments that reproduce the figures in the paper.

## Quick start

```python
from clipping import (
    SpectralClipperConst,     # fixed threshold
    SpectralClipperEMA,       # EMA of sigma_max
    SpectralClipperQuantile,  # sliding-window quantile of sigma_max
)

clipper = SpectralClipperEMA(model, ema_coef=0.9)

for batch in loader:
    loss = model(batch).loss
    loss.backward()
    clipper.clip(model)
    optimizer.step()
    optimizer.zero_grad()
```

Embedding-style parameters (`wte`, `wpe`, `embed_tokens`, `lm_head`, ...) are skipped by default. 

## Experiments

Each subfolder is self-contained (its own configs, sweep YAMLs, and entry-point script):

- [`experiments/trace_regression`](experiments/trace_regression/) — synthetic trace regression. Reproduces the gradient-bias plot.
- [`experiments/mlp`](experiments/mlp/) — small MLP under heavy-tailed Pareto noise. Compares SGDM with norm/spectral clipping against Adam baselines.
- [`experiments/cifar10_airbench`](experiments/cifar10_airbench/) — CIFAR-10 on the airbench94 ResNet. Compares SGDM with no clipping, norm clipping, and spectral clipping.
- [`experiments/gpt2_fineweb`](experiments/gpt2_fineweb/) — GPT-2 pretraining on FineWeb with the HF Trainer. Compares Muon, SGDM, and Adam under norm vs spectral clipping.
- [`experiments/nanogpt_shakespeare`](experiments/nanogpt_shakespeare/) — char-level nanoGPT on tiny-Shakespeare, with SGDM and Muon under norm vs spectral clipping.

## Citation

If you find this work useful, please cite it as follows:

```bibtex
@misc{yukhimchuk2026gradientclippingvectornorms,
      title={Gradient Clipping Beyond Vector Norms: A Spectral Approach for Matrix-Valued Parameters}, 
      author={Alexander Yukhimchuk and Mladen Kolar and Martin Takáč and Sayantan Choudhury},
      year={2026},
      eprint={2605.11838},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.11838}, 
}
