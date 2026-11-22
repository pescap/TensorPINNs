import json
import sys

import numpy as np
from scipy.stats import qmc

from tpinn.ref import moment_logn


def kurtosis(k, m, p, amp, n=2**22, chunk=2**18, seed=0):
    x = qmc.Sobol(k, seed=seed).random(n)
    c2, c4 = 0.0, 0.0
    for q in range(0, n, chunk):
        C = moment_logn(x[q:q + chunk], np, m, p, amp)
        c2, c4 = c2 + np.sum(C**2), c4 + np.sum(C**4)
    return c4 * n / c2**2


def cases():
    for amp in (1.0, 2.0):
        for k in range(2, 7):
            yield k, 8, 2.0, amp
    for k in (4, 6, 8, 10):
        yield k, 4, 2.0, 1.0


def main(argv):
    n = int(argv[0]) if argv else 2**22
    for k, m, p, amp in cases():
        est = [kurtosis(k, m, p, amp, n, seed=s) for s in range(4)]
        print(json.dumps({"k": k, "modes": m, "decay": p, "amp": amp, "kurtosis": float(np.median(est)),
                          "spread": float(np.max(est) / np.min(est))}), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
