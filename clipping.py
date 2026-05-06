from typing import Optional

import torch


@torch.no_grad()
def clip_singular_values_svd(B: torch.Tensor, tau: float) -> tuple[torch.Tensor, float]:
    U, S, Vh = torch.linalg.svd(B, full_matrices=False)
    max_sv = S[0].item() if S.numel() > 0 else 0.0
    return (U * torch.clamp(S, max=tau)) @ Vh, max_sv


@torch.no_grad()
def clip_singular_values_randomized_svd(
    B: torch.Tensor,
    tau: float,
    rank: int,
    p: int = 5,
    n_iter: int = 1,
) -> tuple[torch.Tensor, float]:
    input_dtype = B.dtype
    B_work = B.float() if input_dtype in (torch.float16, torch.bfloat16) else B
    m, n = B_work.shape
    k_eff = min(rank, m, n)
    q = min(k_eff + p, m, n)
    U, S, V = torch.svd_lowrank(B_work, q=q, niter=n_iter)
    U, S, V = U[:, :k_eff], S[:k_eff], V[:, :k_eff]

    max_sv = S[0].item() if S.numel() > 0 else 0.0
    delta = torch.clamp(S, max=tau) - S
    clipped = B_work + (U * delta) @ V.T
    if clipped.dtype != input_dtype:
        clipped = clipped.to(input_dtype)
    return clipped, max_sv


def _is_embedding(name: str) -> bool:
    return (
        ('word_embeddings' in name or 'position_embeddings' in name
         or 'token_type_embeddings' in name or 'embed_tokens' in name
         or 'lm_head' in name or 'wpe' in name or 'wte' in name)
        and 'LayerNorm' not in name
    )


class _BaseSpectralClipper:
    def __init__(
        self,
        model,
        method: str = "svd",
        rank: int = 5,
        randomized_p: int = 5,
        randomized_n_iter: int = 1,
    ):
        if method not in {"svd", "randomized_svd"}:
            raise ValueError(f"method must be 'svd' or 'randomized_svd', got {method!r}")
        self.method = method
        self.rank = rank
        self.randomized_p = randomized_p
        self.randomized_n_iter = randomized_n_iter
        self.param_names = {n for n, _ in model.named_parameters() if not _is_embedding(n)}

    def _threshold(self, param_name: str, max_sv: float) -> Optional[float]:
        raise NotImplementedError

    @torch.no_grad()
    def clip(self, model) -> dict[str, float]:
        thresholds: dict[str, float] = {}
        for name, param in model.named_parameters():
            if param.grad is None or name not in self.param_names:
                continue
            grad = param.grad.data
            shape, dtype = grad.shape, grad.dtype
            G = grad.reshape(grad.shape[0], -1)

            if self.method == "svd":
                U, S, Vh = torch.linalg.svd(G, full_matrices=False)
            else:
                G_work = G.float() if G.dtype in (torch.float16, torch.bfloat16) else G
                m, n = G_work.shape
                k = min(self.rank, m, n)
                q = min(k + self.randomized_p, m, n)
                U, S, V = torch.svd_lowrank(G_work, q=q, niter=self.randomized_n_iter)
                U, S, V = U[:, :k], S[:k], V[:, :k]

            max_sv = S[0].item() if S.numel() > 0 else 0.0
            tau = self._threshold(name, max_sv)
            if tau is None:
                continue
            thresholds[name] = tau

            if self.method == "svd":
                clipped = (U * torch.clamp(S, max=tau)) @ Vh
            else:
                delta = torch.clamp(S, max=tau) - S
                clipped = G_work + (U * delta) @ V.T
                if clipped.dtype != dtype:
                    clipped = clipped.to(dtype)
            param.grad.data = clipped.reshape(shape)

        return thresholds


class SpectralClipperConst(_BaseSpectralClipper):
    def __init__(self, model, threshold: float, **kwargs):
        super().__init__(model, **kwargs)
        self.threshold = float(threshold)

    def _threshold(self, name, max_sv):
        return self.threshold


class SpectralClipperEMA(_BaseSpectralClipper):
    def __init__(self, model, ema_coef: float = 0.9, **kwargs):
        super().__init__(model, **kwargs)
        if not (0.0 < ema_coef < 1.0):
            raise ValueError(f"ema_coef must be in (0, 1), got {ema_coef}")
        self.ema_coef = float(ema_coef)
        self._ema = {n: 0.0 for n in self.param_names}
        self._step = {n: 0 for n in self.param_names}

    def _threshold(self, name, max_sv):
        step = self._step[name]
        if step == 0:
            tau = max_sv
        else:
            tau = max(self._ema[name] / (1.0 - self.ema_coef ** step), 0.0)
        self._ema[name] = self.ema_coef * self._ema[name] + (1.0 - self.ema_coef) * max_sv
        self._step[name] = step + 1
        return tau


class SpectralClipperQuantile(_BaseSpectralClipper):
    def __init__(self, model, q: float = 0.85, window_size: int = 100, **kwargs):
        super().__init__(model, **kwargs)
        if not (0.0 <= q <= 1.0):
            raise ValueError(f"q must be in [0, 1], got {q}")
        if window_size <= 0:
            raise ValueError(f"window_size must be > 0, got {window_size}")
        self.q = float(q)
        self.window_size = int(window_size)
        self._history = {n: [] for n in self.param_names}

    def _threshold(self, name, max_sv):
        history = self._history[name]
        tau = sorted(history)[int(self.q * (len(history) - 1))] if history else None
        history.append(max_sv)
        if len(history) > self.window_size:
            del history[: len(history) - self.window_size]
        return tau
