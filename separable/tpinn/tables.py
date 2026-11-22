import collections
import glob
import json
import sys
from pathlib import Path

import numpy as np



def sci(x, digits=1):
    if x is None:
        return "--"
    m, e = f"{x:.{digits}e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def sec(x):
    return "--" if x is None else "$<$1" if x < 0.5 else f"{x:.0f}"


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


def pm(runs, key, field="err_full"):
    if not runs.get(key):
        return "--"
    v = np.array([o[field] for o in runs[key]])
    m, e = f"{v.mean():.1e}".split("e")
    d = v.std() / 10 ** int(e)
    spread = f"{{\\scriptstyle\\pm{d:.{1 if d >= 0.05 else 2}f}}}" if len(v) > 1 else ""
    return f"${m}{spread}\\times10^{{{int(e)}}}$"


def wall(runs, key):
    return float(np.mean([sum(o["times"].values()) for o in runs[key]])) if runs.get(key) else None


def best(rows, method):
    rows = [r for r in rows if r["method"] == method and np.isfinite(r["err_full"]) and (method != "mc" or r["size"] == 10**6)]
    return min(rows, key=lambda r: r["err_full"]) if rows else None


def cell(row, timed=True):
    if not row:
        return "--"
    e = sci(row["err_full"]) if row["err_full"] > 1e-10 else "$<10^{-10}$"
    return f"{e} ({sec(row['time'])}\\,s)" if timed else e


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
        rows.append([str(k)] + [pm(p, key(a)) for a in ("v", "mo", "lr")] + [sec(wall(p, key(a))) for a in ("v", "mo", "lr")])
    return tabular(["$\\tk$", "V", "MO", "LR", "V (s)", "MO (s)", "LR (s)"], rows)


def classical(root, name):
    f = Path(root) / "classical" / f"{name}.json"
    return json.load(open(f))["rows"] if f.exists() else []


def dense(root, k):
    f = Path(root) / "classical_dense" / f"classical_k{k}_a1.0_m8.json"
    return [r for r in json.load(open(f))["rows"] if r["method"] == "dense"] if f.exists() else []


def tt(root, k, m):
    f = Path(root) / "classical_tt" / "tt.jsonl"
    rows = [r for r in map(json.loads, open(f)) if r["k"] == k and r["modes"] == m and r["amp"] == 1.0] if f.exists() else []
    return min(rows, key=lambda r: r["err_full"]) if rows else None


def lognormal(root):
    pilot, peak = load(root, "pilot"), load(root, "peak")
    ks = range(2, 7)
    mo = lambda k: min([(err(r, ("logn", "", "mo", t, k)), i) for i, (r, t) in enumerate(((pilot, "a1.0"), (peak, "a1.0_peak"))) if err(r, ("logn", "", "mo", t, k)) is not None], default=(None, 0))[1]
    tpinn = [("LR, uniform", lambda k: pm(pilot, ("logn", "", "lr", "a1.0_r32", k))),
             ("LR, peak", lambda k: pm(peak, ("logn", "", "lr", "a1.0_r32_peak", k))),
             ("MO", lambda k: pm((pilot, peak)[mo(k)], ("logn", "", "mo", ("a1.0", "a1.0_peak")[mo(k)], k))),
             ("V", lambda k: pm(pilot, ("logn", "", "v", "a1.0", k)))]
    rows = [[name] + [f(k) for k in ks] for name, f in tpinn]
    for label, method in (("Sparse tensor", "sparse"), ("Dense", "dense"), ("MC, $10^6$", "mc")):
        rows.append([label] + [cell(best(classical(root, f"classical_k{k}_a1.0_c0.0") + dense(root, k), method), method != "mc") for k in ks])
    rows.append(["TT-cross"] + [cell(tt(root, k, 8)) for k in ks])
    top = tabular(["$J=8$"] + [f"$\\tk={k}$" for k in ks], rows)
    ks = (4, 6, 8, 10)
    key = lambda k: ("logn", "", "lr", "m4_r32_peak", k)
    six = {key(k): load(root, "peaka30")[key(k)] + load(root, "peakis")[key(k)] for k in (8, 10)}
    ok = {q: [o for o in v if o["err_full"] < 0.5] for q, v in six.items()}
    mark = lambda k: "$^\\dagger$" if len(ok.get(key(k), [])) < len(six.get(key(k), [])) else ""
    rows = [["LR, peak"] + [pm(ok if k > 6 else peak, key(k)) + mark(k) for k in ks],
            ["LR, peak, $\\rho_2$ tuples"] + [pm(ok, key(k), "err_peak") + mark(k) for k in ks]]
    for label, method in (("Sparse tensor", "sparse"), ("Dense", "dense"), ("MC, $10^6$", "mc")):
        rows.append([label] + [cell(best(classical(root, f"classical_k{k}_a1.0_m4"), method), method != "mc") for k in ks])
    rows.append(["TT-cross"] + [cell(tt(root, k, 4)) for k in ks])
    return top + "\n\\medskip\n" + tabular(["$J=4$"] + [f"$\\tk={k}$" for k in ks], rows)


def kernels(root):
    stage2, grid, ic, paper = load(root, "stage2"), load(root, "gridall"), load(root, "heatic"), load(root, "paper")
    rows = []
    cases = [("Helmholtz", "helm", "sep", "separable"), ("Helmholtz", "helm", "gauss", "Gaussian"), ("Helmholtz", "helm", "exp", "exponential"),
             ("Heat", "heat", "gauss", "Gaussian"), ("Heat", "heat", "exp", "exponential")]
    for op, prob, rhs, label in cases:
        e = [pm(stage2, (prob, rhs, "v", rhs, 2)), pm(stage2, (prob, rhs, "mo", rhs, 2)), pm(grid, (prob, rhs, "lr", f"{rhs}_r64_g", 2))]
        t = [wall(paper, (prob, rhs, a, f"paper_{rhs}", 2)) for a in ("v", "mo", "lr")]
        rows.append([f"{op}, {label}"] + e + [sec(x) for x in t])
    e = [pm(ic, ("heatic", "", "v", "base", 2)), "--", pm(ic, ("heatic", "", "lr", "r16", 2))]
    t = [wall(paper, ("heatic", "", "v", "paper", 2)), None, wall(paper, ("heatic", "", "lr", "paper", 2))]
    rows.append(["Heat, random initial condition"] + e + [sec(x) for x in t])
    return tabular(["Case", "V", "MO", "LR", "V (s)", "MO (s)", "LR (s)"], rows)


def ball(root):
    runs = {s: load(root, s) for s in ("ball", "ballg", "ballk", "paper", "ballw")}
    fem = json.load(open(Path(root) / "classical_ball" / "ball_a1.0_c1.json"))["rows"]
    f = Path(root) / "classical_ball" / "sparse_a1.0.json"
    st = max(json.load(open(f))["rows"], key=lambda r: r["level"]) if f.exists() else None
    lr = {2: ("ballg", "r64_g"), 3: ("ballk", "r64_peak"), 4: ("ballk", "r64_peak")}
    mo = {2: ("ball", "base"), 3: ("ballk", "peak"), 4: ("ballk", "peak")}
    key = lambda a, s, k: ("ball", "", a, s, k)
    t = lambda a, k: f" ({sec(wall(runs['paper'], key(a, 'paper', k)))}\\,s)"
    sc = lambda k: max((r for r in fem if r["k"] == k and r["method"] == "sc_fem_curved"), key=lambda r: r["dofs"])
    mc = lambda k: next(r for r in fem if r["k"] == k and r["method"] == "mc_fem")
    ks = (2, 3, 4)
    rows = [["LR"] + [pm(runs["ballw"], key("lr", lr[k][1], k)) + t("lr", k) for k in ks],
            ["LR, whole ball"] + [pm(runs["ballw"], key("lr", lr[k][1], k), "err_whole") for k in ks],
            ["MO"] + [pm(runs[mo[k][0]], key("mo", mo[k][1], k)) + t("mo", k) for k in ks],
            ["V", pm(runs["ball"], key("v", "base", 2)) + t("v", 2), "--", "--"],
            ["ST-FEM", cell(st and {**st, "time": st["setup"]}), "--", "--"],
            ["SC-FEM"] + [cell(sc(k)) for k in ks],
            ["MC-FEM"] + [cell(mc(k)) for k in ks]]
    return tabular([""] + [f"$\\tk={k}$" for k in ks], rows)


def main(root, dest):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for name, make in (("poisson", poisson), ("lognormal", lognormal), ("kernels", kernels), ("ball", ball)):
        (dest / f"{name}.tex").write_text(make(root))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
