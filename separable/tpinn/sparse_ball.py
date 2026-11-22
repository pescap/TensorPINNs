import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import splu
from skfem import Basis, BilinearForm, ElementTetP2
from skfem.helpers import dot, grad

from tpinn import fem, geom, ref


def level(n, pts):
    m1, m2 = fem.curved_ball(n)
    basis = Basis(m2, ElementTetP2())
    K = BilinearForm(lambda u, v, _: dot(grad(u), grad(v))).assemble(basis)
    M = BilinearForm(lambda u, v, _: u * v).assemble(basis)
    I = basis.complement_dofs(basis.get_dofs())
    P = Basis(m1, ElementTetP2()).probes(pts.T).tocsr()[:, I]
    return {"x": basis.doflocs.T, "lu": splu(K[I][:, I].tocsc()), "M": M, "I": I, "P": P}


def rows(a, b, kw):
    return ref.ball_moment(np.stack(np.broadcast_arrays(a[:, None, :], b[None, :, :]), -2).reshape(-1, 2, 3), np, **kw).reshape(len(a), len(b))


def data(xa, xb, kw, workers, block=32):
    with ProcessPoolExecutor(workers) as ex:
        parts = ex.map(partial(rows, b=xb, kw=kw), [xa[i:i + block] for i in range(0, len(xa), block)])
        return np.vstack(list(parts))


def solve(la, lb, kw, workers):
    C = data(la["x"], lb["x"], kw, workers)
    F = (la["M"] @ (lb["M"] @ C.T).T)[la["I"]][:, lb["I"]]
    U = la["lu"].solve(F)
    U = lb["lu"].solve(U.T).T
    return la["P"] @ U @ lb["P"].T


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default=4, type=int)
    ap.add_argument("--decay", default=2.0, type=float)
    ap.add_argument("--amp", default=1.0, type=float)
    ap.add_argument("--base", default=2, type=int)
    ap.add_argument("--top", default=5, type=int)
    ap.add_argument("--tuples", default=5000, type=int)
    ap.add_argument("--workers", default=16, type=int)
    ap.add_argument("--out", default=os.environ.get("TPINN_OUT", "out"))
    cfg = ap.parse_args(argv)
    kw = dict(m=cfg.modes, p=cfg.decay, amp=cfg.amp)
    rng = np.random.default_rng(0)
    pts = geom.ball_points(rng, 500, 0.95)
    idx = rng.integers(0, 500, (20000, 2))[:cfg.tuples]
    exact = ref.ball_exact(pts[idx], np, **kw)
    t0 = time.perf_counter()
    levels = {n: level(n, pts) for n in range(cfg.base, cfg.top + 1)}
    dofs = {n: len(l["I"]) for n, l in levels.items()}
    cache, times, out = {}, {}, []
    for L in range(2 * cfg.base, cfg.base + cfg.top + 1):
        for a in range(cfg.base, cfg.top + 1):
            b = L - a
            if cfg.base <= b <= a and (a, b) not in cache:
                t = time.perf_counter()
                cache[(a, b)] = solve(levels[a], levels[b], kw, cfg.workers)
                times[(a, b)] = time.perf_counter() - t
        get = lambda a, b: cache[(a, b)] if a >= b else cache[(b, a)].T
        S = sum(get(a, L - a) for a in range(cfg.base, cfg.top + 1) if cfg.base <= L - a <= cfg.top)
        if L > 2 * cfg.base:
            S = S - sum(get(a, L - 1 - a) for a in range(cfg.base, cfg.top + 1) if cfg.base <= L - 1 - a <= cfg.top)
        used = [(a, L - a) for a in range(cfg.base, cfg.top + 1) if cfg.base <= L - a <= cfg.top]
        row = {"method": "sparse_fem", "k": 2, "level": L, "finest": max(max(p) for p in used),
               "dofs": int(sum(dofs[a] * dofs[b] for a, b in used)), "err_full": float(np.linalg.norm(S[idx[:, 0], idx[:, 1]] - exact) / np.linalg.norm(exact)),
               "time": float(sum(v for (a, b), v in times.items() if a + b <= L)), "setup": float(time.perf_counter() - t0)}
        out.append(row)
        print(json.dumps(row), flush=True)
    d = Path(cfg.out) / "classical_ball"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"sparse_a{cfg.amp}.json").write_text(json.dumps({"cfg": vars(cfg), "rows": out, "levels": dofs}))


if __name__ == "__main__":
    main()
