import argparse
import json
import os
import time
from functools import partial
from pathlib import Path

import numpy as np

from tpinn import ref
from tpinn.run import NCHEB, NT


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def monte_carlo(k, c2, m, p, amp, pts, samples, rng, batch=500):
    t, D = ref.cheb(48)
    L = c2 * np.eye(47) - (D @ D)[1:-1, 1:-1]
    M = ref.bary(t, pts)[:, 1:-1] @ np.linalg.inv(L)
    j = np.arange(1, m + 1) * 1.0
    B = np.sin(np.pi * t[1:-1, None] * j) * np.sqrt(amp * j**-p)
    letters = "abcdefghij"[:k]
    spec = ",".join("z" + c for c in letters) + "->" + letters
    acc = 0.0
    for _ in range(samples // batch):
        U = np.exp(rng.standard_normal((batch, m)) @ B.T) @ M.T
        acc = acc + np.einsum(spec, *[U] * k, optimize=True)
    return acc / (samples // batch * batch)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", default=2, type=int)
    ap.add_argument("--c2", default=0.0, type=float)
    ap.add_argument("--modes", default=8, type=int)
    ap.add_argument("--decay", default=2.0, type=float)
    ap.add_argument("--amp", default=1.0, type=float)
    ap.add_argument("--levels", default="3,4,5,6,7,8,9,10")
    ap.add_argument("--dense", default="3,4,5,6,7")
    ap.add_argument("--mc", default="1000,10000,100000,1000000")
    ap.add_argument("--budget", default=4e8, type=float)
    ap.add_argument("--tuples", default=0, type=int)
    ap.add_argument("--nt", default=0, type=int)
    ap.add_argument("--out", default=os.environ.get("TPINN_OUT", "out"))
    cfg = ap.parse_args(argv)
    k = cfg.k
    C = partial(ref.moment_logn, m=cfg.modes, p=cfg.decay, scale=cfg.amp)
    nt = cfg.nt or NT[k]
    pts = (np.arange(nt) + 0.5) / nt
    if cfg.tuples:
        idx = np.random.default_rng(0).integers(0, nt, (5000, k))
        truth, quad = ref.lognormal_tuples(k, cfg.c2, cfg.modes, cfg.decay, cfg.amp, pts, idx)
        pick = lambda T: T[tuple(idx.T)]
    else:
        truth, pick = ref.Reference(C, k, cfg.c2, NCHEB[k]).grid(pts), lambda T: T
    rows = []

    def record(method, size, dofs, fn):
        t0 = time.perf_counter()
        sol = pick(fn())
        rows.append({"method": method, "size": size, "dofs": int(dofs), "err_full": rel(sol, truth), "time": time.perf_counter() - t0})
        print(json.dumps(rows[-1]), flush=True)

    for level in map(int, cfg.levels.split(",")):
        dofs = sum(np.prod([2.0**l - 1 for l in ls]) for q in range(k) for ls in ref.compositions(level + k - 1 - q, k))
        if 2.0**level * nt ** (k - 1) < cfg.budget and dofs < cfg.budget:
            record("sparse", level, dofs, lambda: ref.combination(C, k, cfg.c2, level, pts)[0])
    for level in map(int, cfg.dense.split(",")):
        if (2.0**level - 1) ** k < cfg.budget:
            record("dense", level, (2**level - 1) ** k, lambda: ref.fd_tensor(C, (level,) * k, cfg.c2, pts)[0])
    rng = np.random.default_rng(0)
    for samples in map(int, cfg.mc.split(",")):
        record("mc", samples, samples, lambda: monte_carlo(k, cfg.c2, cfg.modes, cfg.decay, cfg.amp, pts, samples, rng))
    Path(cfg.out).mkdir(parents=True, exist_ok=True)
    (Path(cfg.out) / f"classical_k{k}_a{cfg.amp}_m{cfg.modes}.json").write_text(json.dumps({"cfg": vars(cfg), "rows": rows}))


if __name__ == "__main__":
    main()
