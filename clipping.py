"""
Spectral gradient clipping with a constant threshold (fast / randomized SVD).
"""
import torch


@torch.no_grad()
def svd_randomized_torch(
    A: torch.Tensor,
    k: int,
    p: int = 5,
    n_iter: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Randomized SVD via torch.svd_lowrank."""
    m, n = A.shape
    k_eff = min(k, m, n)
    q = min(k_eff + p, m, n)
    U, S, V = torch.svd_lowrank(A, q=q, niter=n_iter)
    return U[:, :k_eff], S[:k_eff], V[:, :k_eff]


@torch.no_grad()
def clip_singular_values_randomized_svd(
    B: torch.Tensor,
    tau: float,
    rank: int,
    p: int = 5,
    n_iter: int = 1,
) -> torch.Tensor:
    """Clip singular values of B above tau using randomized SVD."""
    input_dtype = B.dtype
    B_work = B.float() if input_dtype in (torch.float16, torch.bfloat16) else B

    k = min(rank, B_work.shape[0], B_work.shape[1])
    U, S, V = svd_randomized_torch(B_work, k=k, p=p, n_iter=n_iter)
    S_clipped = torch.clamp(S, max=tau)
    delta = S_clipped - S
    clipped = B_work + (U * delta) @ V.T
    if clipped.dtype != input_dtype:
        clipped = clipped.to(input_dtype)
    return clipped


class SpectralClipper:
    """
    Spectral gradient clipper with a fixed (constant) threshold.

    For each parameter gradient, treats it as a 2D matrix and clamps any
    singular value above `threshold` down to `threshold`, leaving the rest
    of the spectrum (and the singular vectors) untouched.

    Embedding-style parameters are skipped by default.
    """

    def __init__(
        self,
        model,
        threshold: float,
        r_max: int = 5,
        randomized_p: int = 5,
        randomized_n_iter: int = 1,
    ):
        self.threshold = float(threshold)
        self.r_max = r_max
        self.randomized_p = randomized_p
        self.randomized_n_iter = randomized_n_iter

        self.param_names = set()
        for param_name, _ in model.named_parameters():
            is_embedding = (
                ('word_embeddings' in param_name or
                 'position_embeddings' in param_name or
                 'token_type_embeddings' in param_name or
                 'embed_tokens' in param_name or
                 'lm_head' in param_name or
                 'wpe' in param_name or
                 'wte' in param_name) and
                'LayerNorm' not in param_name
            )
            if not is_embedding:
                self.param_names.add(param_name)

    @torch.no_grad()
    def clip(self, model):
        for param_name, param in model.named_parameters():
            if param.grad is None or param_name not in self.param_names:
                continue

            grad = param.grad.data
            original_shape = grad.shape
            grad_2d = grad.reshape(grad.shape[0], -1)

            clipped_grad = clip_singular_values_randomized_svd(
                grad_2d,
                tau=self.threshold,
                rank=self.r_max,
                p=self.randomized_p,
                n_iter=self.randomized_n_iter,
            )
            param.grad.data = clipped_grad.reshape(original_shape)
