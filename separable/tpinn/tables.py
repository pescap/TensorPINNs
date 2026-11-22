import collections
import glob
import json
import sys
from pathlib import Path

import numpy as np

V1 = {"v": [2.00e-5, 1.49e-4, 6.07e-4, 6.15e-4], "mo": [2.03e-4, 1.10e-2, 4.86e-2, 2.34e-1]}


def sci(x, digits=1):
    if x is None:
        return "--"
    m, e = f"{x:.{digits}e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def sec(x):
    return "--" if x is None else f"{x:.0f}"


def load(root, sweep):
    out = collections.defaultdict(list)
    for f in glob.glob(f"{root}/{sweep}/*.json"):
        o = json.load(open(f))
        if "err_full" not in o:
            continue
        c = o["cfg"]
        prob = c.get("problem", "heatic")
        out[(prob, c.get("rhs", "") if prob in ("helm", "heat") else "", c["arch"], c["tag"] or "base", c.get("k", 2))].append(o)
    return out


def err(runs, key):
    return float(np.mean([o["err_full"] for o in runs[key]])) if runs.get(key) else None


def wall(runs, key):
    return float(np.mean([sum(o["times"].values()) for o in runs[key]])) if runs.get(key) else None


def best(rows, method):
    rows = [r for r in rows if r["method"] == method and np.isfinite(r["err_full"]) and (method != "mc" or r["size"] == 10**6)]
    return min(rows, key=lambda r: r["err_full"]) if rows else None


def cell(row, ok=lambda r: True):
    return f"{sci(row['err_full'])} ({sec(row['time'])}\\,s)" if row and ok(row) else "--"


def tabular(header, rows):
    cols = "l" + "c" * (len(header) - 1)
    lines = [f"\\begin{{tabular}}{{{cols}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    lines += [" & ".join(r) + " \\\\" for r in rows]
    return "\n".join(lines + ["\\bottomrule", "\\end{tabular}"]) + "\n"


def poisson(root):
    p = load(root, "paper")
    rows = []
    for k in range(1, 5):
        key = lambda a: ("sep", "", a, "paper", k)
        rows.append([str(k)] + [sci(err(p, key(a))) for a in ("v", "mo", "lr")] + [sec(wall(p, key(a))) for a in ("v", "mo", "lr")]
                    + [sci(V1["v"][k - 1], 2), sci(V1["mo"][k - 1], 2)])
    head = ["$\\tk$", "V", "MO", "LR", "V (s)", "MO (s)", "LR (s)", "V, first version", "MO, first version"]
    return tabular(head, rows)


def classical(root, name):
    f = Path(root) / "classical" / f"{name}.json"
    return json.load(open(f))["rows"] if f.exists() else []


def tt(root, k, m):
    f = Path(root) / "classical_tt" / "tt.jsonl"
    rows = [r for r in map(json.loads, open(f)) if r["k"] == k and r["modes"] == m and r["amp"] == 1.0] if f.exists() else []
    return min(rows, key=lambda r: r["err_full"]) if rows else None


def lognormal(root):
    pilot, peak = load(root, "pilot"), load(root, "peak")
    ks = range(2, 7)
    tpinn = [("LR, uniform", lambda k: err(pilot, ("logn", "", "lr", "a1.0_r32", k))),
             ("LR, peak", lambda k: err(peak, ("logn", "", "lr", "a1.0_r32_peak", k))),
             ("MO", lambda k: min([e for e in (err(pilot, ("logn", "", "mo", "a1.0", k)), err(peak, ("logn", "", "mo", "a1.0_peak", k))) if e is not None], default=None)),
             ("V", lambda k: err(pilot, ("logn", "", "v", "a1.0", k)))]
    rows = [[name] + [sci(f(k)) for k in ks] for name, f in tpinn]
    for label, method in (("Sparse tensor", "sparse"), ("Dense", "dense"), ("MC, $10^6$", "mc")):
        rows.append([label] + [cell(best(classical(root, f"classical_k{k}_a1.0_c0.0"), method)) for k in ks])
    rows.append(["TT-cross"] + [cell(tt(root, k, 8)) for k in ks])
    top = tabular(["$m=8$"] + [f"$\\tk={k}$" for k in ks], rows)
    ks = (4, 6, 8, 10)
    rows = [["LR, peak"] + [sci(err(peak, ("logn", "", "lr", "m4_r32_peak", k))) for k in ks]]
    for label, method in (("Sparse tensor", "sparse"), ("MC, $10^6$", "mc")):
        rows.append([label] + [cell(best(classical(root, f"classical_k{k}_a1.0_m4"), method)) for k in ks])
    rows.append(["TT-cross"] + [cell(tt(root, k, 4)) for k in ks])
    return top + "\n\\medskip\n" + tabular(["$m=4$"] + [f"$\\tk={k}$" for k in ks], rows)


def kernels(root):
    stage2, grid, ic, paper = load(root, "stage2"), load(root, "gridall"), load(root, "heatic"), load(root, "paper")
    rows = []
    cases = [("Helmholtz", "helm", "sep", "separable"), ("Helmholtz", "helm", "gauss", "Gaussian"), ("Helmholtz", "helm", "exp", "exponential"),
             ("Heat", "heat", "gauss", "Gaussian"), ("Heat", "heat", "exp", "exponential")]
    for op, prob, rhs, label in cases:
        e = [err(stage2, (prob, rhs, "v", rhs, 2)), err(stage2, (prob, rhs, "mo", rhs, 2)), err(grid, (prob, rhs, "lr", f"{rhs}_r64_g", 2))]
        t = [wall(paper, (prob, rhs, a, f"paper_{rhs}", 2)) for a in ("v", "mo", "lr")]
        rows.append([f"{op}, {label}"] + [sci(x) for x in e] + [sec(x) for x in t])
    e = [err(ic, ("heatic", "", "v", "base", 2)), None, err(ic, ("heatic", "", "lr", "r16", 2))]
    t = [wall(paper, ("heatic", "", "v", "paper", 2)), None, wall(paper, ("heatic", "", "lr", "paper", 2))]
    rows.append(["Heat, random initial condition"] + [sci(x) for x in e] + [sec(x) for x in t])
    return tabular(["Case", "V", "MO", "LR", "V (s)", "MO (s)", "LR (s)"], rows)


def ball(root):
    runs = {s: load(root, s) for s in ("ball", "ballg", "ballk", "paper")}
    fem = json.load(open(Path(root) / "classical_ball" / "ball_a1.0_c1.json"))["rows"]
    lr = {2: ("ballg", "r64_g"), 3: ("ballk", "r64_peak"), 4: ("ballk", "r64_peak")}
    mo = {2: ("ball", "base"), 3: ("ballk", "peak"), 4: ("ballk", "peak")}
    rows = []
    for k in (2, 3, 4):
        key = lambda a, s: ("ball", "", a, s, k)
        sc = [r for r in fem if r["k"] == k and r["method"] == "sc_fem_curved"]
        sc = max(sc, key=lambda r: r["dofs"]) if sc else None
        mc = next((r for r in fem if r["k"] == k and r["method"] == "mc_fem"), None)
        rows.append([str(k),
                     f"{sci(err(runs[lr[k][0]], key('lr', lr[k][1])))} ({sec(wall(runs['paper'], key('lr', 'paper')))}\\,s)",
                     f"{sci(err(runs[mo[k][0]], key('mo', mo[k][1])))} ({sec(wall(runs['paper'], key('mo', 'paper')))}\\,s)",
                     f"{sci(err(runs['ball'], key('v', 'base')))} ({sec(wall(runs['paper'], key('v', 'paper')))}\\,s)" if k == 2 else "--",
                     cell(sc), cell(mc)])
    return tabular(["$\\tk$", "LR", "MO", "V", "SC-FEM", "MC-FEM"], rows)


def main(root, dest):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for name, make in (("poisson", poisson), ("lognormal", lognormal), ("kernels", kernels), ("ball", ball)):
        (dest / f"{name}.tex").write_text(make(root))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
