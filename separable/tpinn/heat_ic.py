import argparse
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from tpinn import ref
from tpinn.models import copy_directions, heat, init_mlp, mlp
from tpinn.train import train

A_COEF, CORR = 0.4, 0.05


def kernel(x, y, xp=np):
    return 16 * x * (1 - x) * y * (1 - y) * xp.exp(-((x - y) ** 2) / (2 * CORR))


def truth(P, Q, nm=256):
    j = np.arange(1, nm + 1)
    mu = A_COEF * (np.pi * j) ** 2
    U = lambda Z: np.sqrt(2) * np.sin(np.pi * Z[:, :1] * j) * np.exp(-mu * Z[:, 1:])
    return np.einsum("na,ac,nc->n", U(P), ref.sine_kernel(kernel, nm), U(Q))


def sampler(n):
    def sample(key):
        kx, k0, ks = jax.random.split(key, 3)
        x0 = jax.random.uniform(k0, (n, 2, 2)).at[:, :, 1].set(0.0)
        mixed = jax.random.uniform(ks, (n, 2, 2)).at[:, 0, 1].set(0.0)
        return {"x": jax.random.uniform(kx, (n, 2, 2)), "x0": x0, "xm": mixed, "w": jnp.ones(n)}

    return sample


def make_lr(cfg):
    A = heat(A_COEF)
    gamma = A_COEF * np.pi**2
    dirs = list(jnp.eye(2))

    def init(key):
        return {"net": init_mlp(key, [2] + [cfg.width] * cfg.depth + [cfg.rank]), "c": jnp.ones(cfg.rank) / cfg.rank}

    def phi(params, t):
        return 4 * t[0] * (1 - t[0]) * mlp(params["net"], t)

    def both(params, X):
        P, AP = jax.vmap(lambda t: A(lambda y: phi(params, y), t, dirs))(X.reshape(-1, 2))
        return P.reshape(X.shape[:2] + (-1,)), AP.reshape(X.shape[:2] + (-1,)) / gamma

    def field(params, X):
        return jnp.prod(both(params, X)[0], 1) @ params["c"]

    def blocks(params, batch):
        _, AP = both(params, batch["x"])
        P0, _ = both(params, batch["x0"])
        Pm, APm = both(params, batch["xm"])
        K = kernel(batch["x0"][:, 0, 0], batch["x0"][:, 1, 0], jnp)
        return [jnp.prod(AP, 1) @ params["c"], jnp.prod(P0, 1) @ params["c"] - K, (Pm[:, 0] * APm[:, 1]) @ params["c"]]

    return init, field, blocks


def make_v(cfg):
    A = heat(A_COEF)
    gamma = A_COEF * np.pi**2

    def init(key):
        return init_mlp(key, [4] + [cfg.width] * cfg.depth + [1])

    def point(params, x):
        return 16 * x[0, 0] * (1 - x[0, 0]) * x[1, 0] * (1 - x[1, 0]) * mlp(params, x.ravel())[0]

    def second(params, x):
        return A(lambda y: point(params, y), x, copy_directions(2, 2, 1))[1] / gamma

    def full(params, x):
        return A(lambda y: second(params, y), x, copy_directions(2, 2, 0))[1] / gamma

    def blocks(params, batch):
        K = kernel(batch["x0"][:, 0, 0], batch["x0"][:, 1, 0], jnp)
        return [
            jax.vmap(lambda x: full(params, x))(batch["x"]),
            jax.vmap(lambda x: point(params, x))(batch["x0"]) - K,
            jax.vmap(lambda x: second(params, x))(batch["xm"]),
        ]

    return init, jax.vmap(point, (None, 0)), blocks


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="lr", choices=["lr", "v"])
    ap.add_argument("--width", default=64, type=int)
    ap.add_argument("--depth", default=4, type=int)
    ap.add_argument("--rank", default=32, type=int)
    ap.add_argument("--n", default=4000, type=int)
    ap.add_argument("--adam", default=20000, type=int)
    ap.add_argument("--lbfgs", default=5000, type=int)
    ap.add_argument("--lm", default=0, type=int)
    ap.add_argument("--lr", default=1e-3, type=float)
    ap.add_argument("--const-lr", dest="const_lr", default=0, type=int)
    ap.add_argument("--fixed", default=0, type=int)
    ap.add_argument("--balance", default=0, type=int)
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--tag", default="base")
    ap.add_argument("--out", default=os.environ.get("TPINN_OUT", "out"))
    cfg = ap.parse_args(argv)
    init, field, blocks = {"lr": make_lr, "v": make_v}[cfg.arch](cfg)
    key = jax.random.PRNGKey(cfg.seed)
    params, w, hist, times = train(jax.random.fold_in(key, 1), init(key), blocks, sampler(cfg.n), cfg, np.ones(3))
    rng = np.random.default_rng(0)
    pts = rng.random((500, 2))
    idx = rng.integers(0, 500, (20000, 2))
    X = np.stack([pts[idx[:, 0]], pts[idx[:, 1]]], 1)
    D = np.stack([pts, pts], 1)
    rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))
    f = jax.jit(field)
    out = {
        "cfg": vars(cfg), "times": times, "hist": hist,
        "err_full": rel(np.asarray(f(params, jnp.asarray(X))), truth(X[:, 0], X[:, 1])),
        "err_diag": rel(np.asarray(f(params, jnp.asarray(D))), truth(pts, pts)),
        "ref_err": rel(truth(X[:, 0], X[:, 1], 128), truth(X[:, 0], X[:, 1])),
    }
    Path(cfg.out).mkdir(parents=True, exist_ok=True)
    (Path(cfg.out) / f"heatic_{cfg.arch}_{cfg.tag}_s{cfg.seed}.json").write_text(json.dumps(out))
    print(json.dumps({q: out[q] for q in ["err_full", "err_diag", "ref_err", "times"]}))


if __name__ == "__main__":
    main()
