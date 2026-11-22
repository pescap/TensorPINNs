import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from tpinn import fem, geom, ref


def tuples(V, w, idx):
    out = w * 1.0
    for j in range(idx.shape[1]):
        out = out * V[idx[:, j]]
    return out.sum(1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default=4, type=int)
    ap.add_argument("--decay", default=2.0, type=float)
    ap.add_argument("--amp", default=1.0, type=float)
    ap.add_argument("--refs", default="2,3,4")
    ap.add_argument("--mc", default="1000,10000,100000")
    ap.add_argument("--tuples", default=5000, type=int)
    ap.add_argument("--curved", default=0, type=int)
    ap.add_argument("--out", default=os.environ.get("TPINN_OUT", "out"))
    cfg = ap.parse_args(argv)
    kw = dict(m=cfg.modes, p=cfg.decay, amp=cfg.amp)
    pts = geom.ball_points(np.random.default_rng(0), 500, 0.95)
    idx = {k: np.random.default_rng(0) for k in (2, 3, 4)}
    for k, g in idx.items():
        geom.ball_points(g, 500, 0.95)
        idx[k] = g.integers(0, 500, (20000, k))[:cfg.tuples]
    exact = {k: ref.ball_exact(pts[i], np, **kw) for k, i in idx.items()}
    xi, w = ref.hermite(4, cfg.modes, cfg.decay, cfg.amp)
    rows = []
    for n in [int(v) for v in cfg.refs.split(",")]:
        t0 = time.perf_counter()
        V, dofs = fem.ball_solutions(n, pts, xi, curved=bool(cfg.curved), **kw)
        t = time.perf_counter() - t0
        for k, i in idx.items():
            e = float(np.linalg.norm(tuples(V, w, i) - exact[k]) / np.linalg.norm(exact[k]))
            rows.append({"method": "sc_fem_curved" if cfg.curved else "sc_fem", "k": k, "refs": n, "dofs": dofs, "nodes": len(w), "err_full": e, "time": t})
            print(json.dumps(rows[-1]), flush=True)
    n = max(int(v) for v in cfg.refs.split(","))
    rng = np.random.default_rng(1)
    for M in [int(v) for v in cfg.mc.split(",") if v]:
        t0 = time.perf_counter()
        V, dofs = fem.ball_solutions(n, pts, rng.standard_normal((M, cfg.modes)), **kw)
        t = time.perf_counter() - t0
        for k, i in idx.items():
            e = float(np.linalg.norm(tuples(V, np.full(M, 1.0 / M), i) - exact[k]) / np.linalg.norm(exact[k]))
            rows.append({"method": "mc_fem", "k": k, "refs": n, "dofs": dofs, "nodes": M, "err_full": e, "time": t})
            print(json.dumps(rows[-1]), flush=True)
    out = Path(cfg.out) / "classical_ball"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"ball_a{cfg.amp}_c{cfg.curved}.json").write_text(json.dumps({"cfg": vars(cfg), "rows": rows}))


if __name__ == "__main__":
    main()
