import glob
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm

from tpinn import tables

ARCH = {"v": ("V-TPINN", "#1baf7a", "^"), "mo": ("MO-TPINN", "#eb6834", "s"), "lr": ("LR-TPINN", "#2a78d6", "o")}
CLASSIC = {"sparse": ("sparse tensor", "#eda100", "D"), "dense": ("dense FD", "#e87ba4", "v"), "mc": ("MC, $10^6$", "#008300", "x"),
           "tt": ("TT-cross", "#6250d6", "P")}
FIELD = LinearSegmentedColormap.from_list("field", ["#cde2fb", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"])
ERROR = LinearSegmentedColormap.from_list("error", ["#fdeee6", "#f5a27f", "#eb6834", "#a93f15", "#5e1f06"])
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e4e3df", "grid.linewidth": 0.5, "lines.linewidth": 1.5, "lines.markersize": 5})


def diag(root, name):
    f = Path(root) / "paper" / f"{name}.npz"
    return np.load(f) if f.exists() else None


def sol1(root, out):
    fig, axes = plt.subplots(2, 4, figsize=(10, 4.2), sharex=True)
    for c, k in enumerate(range(1, 5)):
        for arch, (label, color, marker) in ARCH.items():
            d = diag(root, f"sep_{arch}_k{k}_hard_paper_s0")
            if d is None:
                continue
            x = d["points"].reshape(len(d["points"]), -1)[:, 0]
            axes[0, c].plot(x, d["pred"], color=color, label=label)
            axes[1, c].semilogy(x, np.abs(d["pred"] - d["true"]) + 1e-16, color=color, label=label)
            truth = d["true"]
        axes[0, c].plot(x, truth, "--", color="#1a1a19", lw=1, label="exact")
        axes[0, c].set_title(f"$k={k}$")
        axes[1, c].set_xlabel("$x$")
    axes[0, 0].set_ylabel(r"diag$(\Sigma^k_\theta)$")
    axes[1, 0].set_ylabel("pointwise error")
    axes[0, 0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(Path(out) / "sol1_v2.pdf")


def field_row(axes, d, cols):
    P = d["points"][:, 0, :]
    err = np.abs(d["pred"] - d["true"])
    vals = {"ref": d["true"], "pred": d["pred"]}
    for ax, what in zip(axes, cols):
        if what == "err":
            im = ax.tricontourf(P[:, 0], P[:, 1], err + 1e-12, levels=np.logspace(np.log10(err.max()) - 3, np.log10(err.max()), 13), cmap=ERROR, norm=LogNorm(), extend="min")
        else:
            im = ax.tricontourf(P[:, 0], P[:, 1], vals[what], levels=20, cmap=FIELD)
        plt.colorbar(im, ax=ax, shrink=0.85, format="%.0e" if what == "err" else None)
        ax.set_aspect("equal")
        ax.grid(False)


def solhelm(root, out):
    fig, axes = plt.subplots(2, 3, figsize=(9, 5.4))
    for c, (arch, (label, _, _)) in enumerate(ARCH.items()):
        d = diag(root, f"helm_{arch}_k2_hard_paper_sep_s0")
        if d is None:
            continue
        field_row([axes[0, c], axes[1, c]], d, ["pred", "err"])
        axes[0, c].set_title(label)
    axes[0, 0].set_ylabel("diag$(\\Sigma_\\theta)$\n$y$")
    axes[1, 0].set_ylabel("pointwise error\n$y$")
    fig.tight_layout()
    fig.savefig(Path(out) / "solhelm_v2.pdf")


def kernel_diag(root, out, prob, name, xlabel, ylabel):
    fig, axes = plt.subplots(2, 3, figsize=(9, 5.4))
    for r, rhs in enumerate(("gauss", "exp")):
        d = diag(root, f"{prob}_lr_k2_hard_paper_{rhs}_s0")
        if d is None:
            continue
        field_row(axes[r], d, ["ref", "pred", "err"])
        axes[r, 0].set_ylabel(f"{'Gaussian' if rhs == 'gauss' else 'exponential'}\n{ylabel}")
    for ax, t in zip(axes[0], ("reference", "LR-TPINN", "pointwise error")):
        ax.set_title(t)
    for ax in axes[1]:
        ax.set_xlabel(xlabel)
    fig.tight_layout()
    fig.savefig(Path(out) / name)


def moments(root, out):
    pilot, peak, bench = tables.load(root, "pilot"), tables.load(root, "peak"), tables.load(root, "bench")
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3))
    ax = axes[0]
    ks = list(range(2, 7))
    curves = [("LR-TPINN, peak", "#2a78d6", "o", "-", lambda k: tables.err(peak, ("logn", "", "lr", "a1.0_r32_peak", k))),
              ("LR-TPINN, uniform", "#2a78d6", "o", ":", lambda k: tables.err(pilot, ("logn", "", "lr", "a1.0_r32", k))),
              ("MO-TPINN", "#eb6834", "s", "-", lambda k: tables.err(pilot, ("logn", "", "mo", "a1.0", k))),
              ("V-TPINN", "#1baf7a", "^", "-", lambda k: tables.err(pilot, ("logn", "", "v", "a1.0", k)))]
    for method in ("sparse", "dense", "mc"):
        label, color, marker = CLASSIC[method]
        curves.append((label, color, marker, "--", lambda k, m=method: (tables.best(tables.classical(root, f"classical_k{k}_a1.0_c0.0"), m) or {}).get("err_full")))
    label, color, marker = CLASSIC["tt"]
    curves.append((label, color, marker, "--", lambda k: (tables.tt(root, k, 8) or {}).get("err_full")))
    for label, color, marker, style, f in curves:
        pts = [(k, f(k)) for k in ks if f(k) is not None]
        ax.semilogy([p[0] for p in pts], [p[1] for p in pts], style, color=color, marker=marker, label=label)
    ax.set(xlabel="$k$", ylabel="relative error (full tensor)", xticks=ks)
    ax.legend(frameon=False, fontsize=6.5, ncol=2)
    ax = axes[1]
    for arch, (label, color, marker) in ARCH.items():
        pts = [(k, 1000 * bench[("sep", "", arch, "bench", k)][0]["times"]["adam"] / 1900) for k in range(1, 7) if bench.get(("sep", "", arch, "bench", k))]
        ax.semilogy([p[0] for p in pts], [p[1] for p in pts], color=color, marker=marker, label=label)
    ax.set(xlabel="$k$", ylabel="ms per Adam step", xticks=range(1, 7))
    ax.legend(frameon=False)
    ax = axes[2]
    kur = {(r["k"], r["amp"]): r["kurtosis"] for r in map(json.loads, open(Path(root) / "kurtosis.jsonl")) if r["modes"] == 8}
    for label, runs, tag, color, marker in (("uniform", pilot, "a{}_r32", "#2a78d6", "o"), ("peak", peak, "a{}_r32_peak", "#eb6834", "s")):
        pts = sorted((kap, tables.err(runs, ("logn", "", "lr", tag.format(amp), k))) for (k, amp), kap in kur.items())
        pts = [p for p in pts if p[1] is not None]
        ax.loglog([p[0] for p in pts], [p[1] for p in pts], marker, color=color, label=f"LR-TPINN, {label}")
    ax.set(xlabel=r"kurtosis Kurt$(C^k)$", ylabel="relative error (full tensor)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(Path(out) / "moments.pdf")


def main(root, out):
    Path(out).mkdir(parents=True, exist_ok=True)
    sol1(root, out)
    solhelm(root, out)
    kernel_diag(root, out, "helm", "kernels_diag_v2.pdf", "$x$", "$y$")
    kernel_diag(root, out, "heat", "heat_diag_v2.pdf", "$x$", "$t$")
    moments(root, out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
