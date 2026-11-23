import json
import sys

import numpy as np

from tpinn.ref import box_kernel_moment

RANKS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)


def tail(d, c2, lam, m, n=32):
    g = (np.arange(m) + 0.5) / m
    pts = np.stack(np.meshgrid(*[g] * d, indexing="ij"), -1).reshape(-1, d)
    S = box_kernel_moment(d, c2, lam, n, pts) / m**d
    ev = np.sort(np.abs(np.linalg.eigvalsh((S + S.T) / 2)))[::-1]
    tot = np.sum(ev**2)
    return {R: float(np.sqrt(np.sum(ev[R:] ** 2) / tot)) for R in RANKS if R < len(ev)}


def main(argv):
    for d, m in ((2, 80), (3, 24)):
        for c2 in (0.0, -4 * np.pi**2):
            for lam in (0.05, 0.01):
                print(json.dumps({"d": d, "c2": c2, "corr": lam, "m": m, "tail": tail(d, c2, lam, m)}), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
