import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import teneva

from tpinn import ref, run


def tt_moment(k, m, p, amp, c2, n, budget, seed=0):
    t, D = ref.cheb(n)
    nodes = t[1:-1]
    Linv = np.linalg.inv(c2 * np.eye(n - 1) - (D @ D)[1:-1, 1:-1])
    f = lambda I: ref.moment_logn(nodes[np.asarray(I)], np, m, p, amp)
    Y = teneva.cross(f, teneva.rand([n - 1] * k, 2, seed=seed), m=budget, dr_max=2, e=1e-13)
    Y = teneva.truncate(Y, 1e-12)
    return [np.einsum("anb,mn->amb", G, Linv) for G in Y], t, max(G.shape[2] for G in Y)


def evaluate(cores, t, X, chunk=100000):
    out = []
    for c in range(0, len(X), chunk):
        v = np.ones((len(X[c:c + chunk]), 1))
        for i, G in enumerate(cores):
            v = np.einsum("sa,anb,sn->sb", v, G, ref.bary(t, X[c:c + chunk, i])[:, 1:-1])
        out.append(v[:, 0])
    return np.concatenate(out)


def cases():
    for amp in (1.0, 2.0):
        for k in range(2, 7):
            yield k, 8, amp, False
    for k in (4, 6, 8, 10):
        yield k, 4, 1.0, True


def main(argv):
    out = Path(os.environ.get("TPINN_OUT", "out")) / "classical_tt"
    out.mkdir(parents=True, exist_ok=True)
    done = {(r["k"], r["modes"], r["amp"], r["n"], r["budget"]) for r in map(json.loads, open(out / "tt.jsonl"))} if (out / "tt.jsonl").exists() else set()
    for k, m, amp, tuples in cases():
        if all((k, m, amp, n, b) in done for n in (24, 32, 48) for b in (10**5, 10**6)):
            continue
        cfg = run.parse(["--problem", "logn", "--k", str(k), "--modes", str(m), "--amp", str(amp)] + (["--tuples", "1"] if tuples else []))
        _, _, _, X, _, truth = (run.interval_tuples if tuples else run.interval)(cfg)
        T = truth()["full"]
        for n in (24, 32, 48):
            for budget in (10**5, 10**6):
                if (k, m, amp, n, budget) in done:
                    continue
                t0 = time.perf_counter()
                cores, t, rank = tt_moment(k, m, 2.0, amp, 0.0, n, budget)
                build = time.perf_counter() - t0
                err = float(np.linalg.norm(evaluate(cores, t, X[..., 0]) - T) / np.linalg.norm(T))
                row = {"k": k, "modes": m, "amp": amp, "n": n, "budget": budget, "rank": int(rank), "err_full": err, "time": build}
                print(json.dumps(row), flush=True)
                with open(out / "tt.jsonl", "a") as fh:
                    fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
