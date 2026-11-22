import collections
import glob
import json
import sys
from pathlib import Path

import numpy as np


def runs(root):
    for f in sorted(glob.glob(f"{root}/*/*.json")):
        o = json.load(open(f))
        if "err_full" in o and Path(f).parent.name not in ("dry", "holeref", "tupref") and not Path(f).parent.name.startswith("smoke"):
            yield Path(f).parent.name, o


def main(root):
    groups = collections.defaultdict(list)
    for sweep, o in runs(root):
        c = o["cfg"]
        key = (sweep, c.get("problem", "heatic"), c.get("rhs", "") if c.get("problem") in ("helm", "heat") else "", c["arch"], c.get("bc", "hard"), c["tag"] or "base")
        groups[key].append((c.get("k", 2), o))
    print("| barrido | problema | arq. | BC | etiqueta | k | n | error tensor | error diag. | cota rel. | tiempo (s) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for key in sorted(groups):
        by_k = collections.defaultdict(list)
        for k, o in groups[key]:
            by_k[k].append(o)
        for k in sorted(by_k):
            rs = by_k[k]
            e = np.array([r["err_full"] for r in rs])
            d = np.array([r["err_diag"] for r in rs])
            b = [r["bound_rel"] for r in rs if r.get("bound_rel")]
            t = np.mean([sum(v for q, v in r["times"].items() if q != "compile") for r in rs])
            sweep, prob, rhs, arch, bc, tag = key
            print(f"| {sweep} | {prob}{' ' + rhs if rhs else ''} | {arch} | {bc} | {tag} | {k} | {len(rs)} | {e.mean():.1e} ± {e.std():.0e} | {d.mean():.1e} | {np.mean(b):.1e} | {t:.0f} |" if b else
                  f"| {sweep} | {prob}{' ' + rhs if rhs else ''} | {arch} | {bc} | {tag} | {k} | {len(rs)} | {e.mean():.1e} ± {e.std():.0e} | {d.mean():.1e} | - | {t:.0f} |")
    print("\n| clásico | k | método | tamaño | gdl | error tensor | tiempo (s) |")
    print("|---|---|---|---|---|---|---|")
    for f in sorted(glob.glob(f"{root}/classical/*.json")):
        o = json.load(open(f))
        for r in o["rows"]:
            print(f"| {Path(f).stem} | {o['cfg']['k']} | {r['method']} | {r['size']} | {r['dofs']:.1e} | {r['err_full']:.1e} | {r['time']:.3g} |")


if __name__ == "__main__":
    main(sys.argv[1])
