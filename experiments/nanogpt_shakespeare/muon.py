# Derived from https://github.com/infinity-stars/ASGO.
# Muon (https://kellerjordan.github.io/posts/muon/).

import math

import torch


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int) -> torch.Tensor:
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


class Muon(torch.optim.Optimizer):
    """Muon for matrix-valued (ndim >= 2) parameters. Embeddings/1-D parameters
    must be optimised separately (e.g. with Adam)."""

    def __init__(self, params, learning_rate=1e-2, momentum=0.9, weight_decay=0.1, ns_steps=5):
        defaults = dict(lr=learning_rate, momentum=momentum,
                        weight_decay=weight_decay, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr = group["lr"]
            mom = group["momentum"]
            wd = group["weight_decay"]
            ns = group["ns_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                assert p.ndim > 1, "Muon only supports ndim>=2 params; use Adam for the rest."
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(grad)

                buf = state["momentum_buffer"]
                buf.lerp_(grad, 1 - mom)

                update_input = buf.view(len(buf), -1) if buf.ndim == 4 else buf
                update = zeropower_via_newtonschulz5(update_input, steps=ns)
                if p.ndim == 4:
                    update = update.view(p.shape)

                update = 0.2 * math.sqrt(max(p.size(-2), p.size(-1))) * update
                if wd > 0:
                    p.mul_(1 - lr * wd)
                p.add_(update, alpha=-lr)
