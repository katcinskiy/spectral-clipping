"""Trace regression: SGD with three families of spectral clipping thresholds."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

CONST_TAUS = (0.05, 0.1, 0.5, 1.0, 5.0)


def make_problem(n: int, N: int, sigma: float, rng: np.random.Generator):
    A = np.zeros((N, n, n))
    for i in range(N):
        Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
        eigs = rng.uniform(0.5, 1.5, size=n)
        A[i] = Q @ np.diag(eigs) @ Q.T

    X_true = rng.standard_normal((n, n))
    b = np.einsum('ijk,jk->i', A, X_true) + sigma * rng.standard_normal(N)
    L = 2.0 * sum(np.linalg.norm(A_i, ord='fro') ** 2 for A_i in A) / N
    return A, b, L


def full_gradient(X, A, b):
    residuals = np.einsum('ijk,jk->i', A, X) - b
    return 2.0 * np.einsum('i,ijk->jk', residuals, A) / len(b)


def stochastic_gradient(X, A_i, b_i):
    return 2.0 * (np.sum(A_i * X) - b_i) * A_i


def clip_gradient(G, tau):
    U, s, Vt = np.linalg.svd(G)
    return (U * np.minimum(s, tau)) @ Vt, s[0]


def estimate_delta(X, A, b, tau, mc_samples, rng):
    full = full_gradient(X, A, b)
    accum = np.zeros_like(full)
    for _ in range(mc_samples):
        i = rng.integers(len(b))
        accum += clip_gradient(stochastic_gradient(X, A[i], b[i]), tau)[0]
    return float(np.linalg.norm(full - accum / mc_samples, ord='fro') ** 2)


def sgd_const_tau(A, b, tau, K, lr, mc_samples, rng):
    X = np.zeros((A.shape[1], A.shape[1]))
    deltas = np.empty(K)
    for k in tqdm(range(K), desc=f"const tau={tau}", leave=False):
        deltas[k] = estimate_delta(X, A, b, tau, mc_samples, rng)
        i = rng.integers(len(b))
        Gc, _ = clip_gradient(stochastic_gradient(X, A[i], b[i]), tau)
        X -= lr * Gc
    return deltas


def sgd_ema(A, b, K, lr, beta, eps, mc_samples, rng):
    X = np.zeros((A.shape[1], A.shape[1]))
    tau, tau_hat = 0.0, 0.0
    deltas = np.empty(K)
    for k in tqdm(range(K), desc="EMA", leave=False):
        deltas[k] = estimate_delta(X, A, b, tau_hat, mc_samples, rng)
        i = rng.integers(len(b))
        Gc, sv_max = clip_gradient(stochastic_gradient(X, A[i], b[i]), tau_hat)
        X -= lr * Gc
        tau = beta * tau + (1 - beta) * (sv_max + eps)
        tau_hat = tau / (1 - beta ** (k + 1))
    return deltas


def sgd_quantile(A, b, K, lr, window_size, q, mc_samples, rng):
    X = np.zeros((A.shape[1], A.shape[1]))
    sv_history: list[float] = []
    deltas = np.empty(K)
    for k in tqdm(range(K), desc=f"quantile q={q}", leave=False):
        tau = float(np.quantile(sv_history[-window_size:], q)) if sv_history else np.inf
        deltas[k] = estimate_delta(X, A, b, tau, mc_samples, rng)
        i = rng.integers(len(b))
        Gc, sv_max = clip_gradient(stochastic_gradient(X, A[i], b[i]), tau)
        sv_history.append(sv_max)
        X -= lr * Gc
    return deltas


def aggregate_to_rows(method: str, deltas_per_seed: dict[int, np.ndarray]) -> list[dict]:
    K = next(iter(deltas_per_seed.values())).shape[0]
    cumavg = np.stack([np.cumsum(d) / np.arange(1, K + 1) for d in deltas_per_seed.values()])
    return [
        {
            "method": method,
            "iteration": k,
            "mean": float(cumavg[:, k].mean()),
            "std": float(cumavg[:, k].std()),
            "min": float(cumavg[:, k].min()),
            "max": float(cumavg[:, k].max()),
        }
        for k in range(K)
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[11, 12, 13])
    p.add_argument("--K", type=int, default=100_000)
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--N", type=int, default=10)
    p.add_argument("--sigma", type=float, default=0.01)
    p.add_argument("--mc-samples", type=int, default=10)
    p.add_argument("--ema-beta", type=float, default=0.9)
    p.add_argument("--quantile-q", type=float, default=0.85)
    p.add_argument("--quantile-window", type=int, default=100)
    p.add_argument("--lr-scale", type=float, default=0.5)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    if args.out is None:
        args.out = Path(f"tracereg_sigma{args.sigma}_seeds{len(args.seeds)}.csv")

    ema, quantile = {}, {}
    constant: dict[float, dict[int, np.ndarray]] = {tau: {} for tau in CONST_TAUS}

    for seed in args.seeds:
        print(f"=== seed={seed} ===")
        rng = np.random.default_rng(seed)
        A, b, L = make_problem(args.n, args.N, args.sigma, rng)
        lr = args.lr_scale / L
        print(f"  L={L:.4f}  lr={lr:.6f}")

        ema[seed] = sgd_ema(A, b, args.K, lr, args.ema_beta, 1e-8, args.mc_samples, rng)
        quantile[seed] = sgd_quantile(
            A, b, args.K, lr, args.quantile_window, args.quantile_q, args.mc_samples, rng,
        )
        for tau in CONST_TAUS:
            constant[tau][seed] = sgd_const_tau(A, b, tau, args.K, lr, args.mc_samples, rng)

    rows: list[dict] = []
    rows += aggregate_to_rows("EMA", ema)
    rows += aggregate_to_rows("Quantile", quantile)
    for tau, by_seed in constant.items():
        rows += aggregate_to_rows(f"constant_tau_{tau}", by_seed)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()
