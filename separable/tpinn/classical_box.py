import argparse
import json
import os
import time
from math import comb
from pathlib import Path

import numpy as np
from scipy.fft import dstn

from tpinn import ref

LETTERS = "abcdefghijklmnopqrstuvwxyz"


def nodes(level):
    return np.arange(1, 2**level) / 2**level


def eig(level):
    h = 2.0**-level
    return 4 / h**2 * np.sin(np.arange(1, 2**level) * np.pi * h / 2) ** 2


def solve(levels, d, lam, c2, P, rhs="gauss"):
    xs = [nodes(l) for l in levels]
    gs = [ref.kernel_1d(rhs, lam)(xs[i][:, None], xs[d + i][None, :]) for i in range(d)]
    spec = ",".join(LETTERS[i] + LETTERS[d + i] for i in range(d)) + "->" + LETTERS[:2 * d]
    T = dstn(np.einsum(spec, *gs), type=1, norm="ortho")
    lx, ly = c2, c2
    for i in range(d):
        shape = [1] * (2 * d)
        shape[i] = -1
        lx = lx + eig(levels[i]).reshape(shape)
        shape = [1] * (2 * d)
        shape[d + i] = -1
        ly = ly + eig(levels[d + i]).reshape(shape)
    T = dstn(T / (lx * ly), type=1, norm="ortho")
    coords = np.concatenate([P[:, 0], P[:, 1]], 1)
    return np.concatenate([evaluate(T, xs, coords[c:c + 16]) for c in range(0, len(P), 16)]), T.size


def evaluate(T, xs, coords):
    R = ref.hat(xs[0], coords[:, 0]) @ T.reshape(len(xs[0]), -1)
    for a in range(1, len(xs)):
        R = np.einsum("pn,pnr->pr", ref.hat(xs[a], coords[:, a]), R.reshape(len(coords), len(xs[a]), -1))
    return R[:, 0]


def compositions(total, parts):
    if parts == 1:
        yield (total,)
        return
    for first in range(1, total - parts + 2):
        for rest in compositions(total - first, parts - 1):
            yield (first,) + rest


def combine(groups, n, d, lam, c2, P, rhs="gauss"):
    D = len(groups)
    out, dofs = 0.0, 0
    for q in range(D):
        for parts in compositions(n + D - 1 - q, D):
            levels = [l for g, l in zip(groups, parts) for _ in range(g)]
            u, size = solve(levels, d, lam, c2, P, rhs)
            out, dofs = out + (-1) ** q * comb(D - 1, q) * u, dofs + size
    return out, dofs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", default=4, type=int)
    ap.add_argument("--corr", default=0.1, type=float)
    ap.add_argument("--c2", default=0.0, type=float)
    ap.add_argument("--rhs", default="gauss", choices=["gauss", "exp"])
    ap.add_argument("--pairs", default=500, type=int)
    ap.add_argument("--sg", default="")
    ap.add_argument("--st", default="")
    ap.add_argument("--dense", default="")
    ap.add_argument("--out", default=os.environ.get("TPINN_OUT", "out"))
    cfg = ap.parse_args(argv)
    d = cfg.dim
    P = np.random.default_rng(0).random((cfg.pairs, 2, d))
    t0 = time.perf_counter()
    truth = ref.box_highd(d, cfg.corr, P, cfg.c2, rhs=cfg.rhs)
    rows = [{"method": "reference", "size": 128, "dofs": 0, "err_full": 0.0, "time": time.perf_counter() - t0}]
    runs = [("sparse_grid", [1] * (2 * d), cfg.sg), ("sparse_tensor", [d, d], cfg.st)]
    for name, groups, levels in runs:
        for n in [int(v) for v in levels.split(",") if v]:
            t0 = time.perf_counter()
            u, dofs = combine(groups, n, d, cfg.corr, cfg.c2, P, cfg.rhs)
            rows.append({"method": name, "size": n, "dofs": dofs, "err_full": float(np.linalg.norm(u - truth) / np.linalg.norm(truth)), "time": time.perf_counter() - t0})
            print(json.dumps(rows[-1]), flush=True)
    for n in [int(v) for v in cfg.dense.split(",") if v]:
        t0 = time.perf_counter()
        u, dofs = solve([n] * (2 * d), d, cfg.corr, cfg.c2, P, cfg.rhs)
        rows.append({"method": "dense", "size": n, "dofs": dofs, "err_full": float(np.linalg.norm(u - truth) / np.linalg.norm(truth)), "time": time.perf_counter() - t0})
        print(json.dumps(rows[-1]), flush=True)
    out = Path(cfg.out) / "classical_box"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"box_{cfg.rhs}_d{d}_l{cfg.corr}_c{cfg.c2}.json").write_text(json.dumps({"cfg": vars(cfg), "rows": rows}))


if __name__ == "__main__":
    main()
