import torch


def _two_sided_pareto(shape, alpha, scale, device):
    s = torch.distributions.Pareto(scale=scale, alpha=alpha).sample(shape).to(device)
    signs = torch.randint(0, 2, shape, device=device) * 2 - 1
    return s * signs


def generate_lowrank_noise(shape, alpha, scale, device, rank):
    shape = tuple(shape)
    m = int(shape[0])
    n = int(torch.tensor(shape[1:]).prod().item()) if len(shape) > 1 else 1
    rank = min(rank, m, n)
    noise = torch.zeros(m, n, device=device)
    for _ in range(rank):
        s = _two_sided_pareto((1,), alpha, scale, device).item()
        u = torch.randn(m, device=device)
        v = torch.randn(n, device=device)
        noise += s * torch.outer(u / u.norm(), v / v.norm())
    return noise.reshape(shape) if len(shape) > 1 else noise.reshape(shape[0])


def add_lowrank_noise_to_gradients(model, noise_cfg, device, rank):
    for p in model.parameters():
        if p.grad is None:
            continue
        p.grad.add_(generate_lowrank_noise(
            p.grad.shape, noise_cfg.alpha, noise_cfg.pareto_scale, device, rank,
        ))
