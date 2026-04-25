# Spectral Clipping

> **Status: WIP.**

Code accompanying the paper **Gradient Clipping Beyond Vector Norms: A Spectral Approach for Matrix-Valued Parameters**.

Drop-in gradient clipper that clamps singular values of each matrix-valued gradient above a threshold $\tau$, instead of rescaling by the $\ell_2$ norm.

## Usage

```python
import torch
from clipping import SpectralClipper

model = ...  # any nn.Module
clipper = SpectralClipper(model, threshold=1.0)

for batch in loader:
    loss = model(batch).loss
    loss.backward()
    clipper.clip(model)        # in-place spectral clip on .grad
    optimizer.step()
    optimizer.zero_grad()
```
