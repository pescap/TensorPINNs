"""
Combined publication figure: low-rank hard-BC V-TPINN 2nd-moment diagonal vs MC+FEM on the
square-with-a-hole domain, for two stiffnesses (kappa=4 top, kappa=2 bottom).
Each row: MC+FEM E[u^2] | TPINN Sigma(p,p) | node-wise scatter.  Vector PDF.
"""
import argparse, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="figs/complexgeom_tpinn.pdf")
ap.add_argument("--r0", default=0.3, type=float)
# rows: (tag, refsuf, kappa-label)
ap.add_argument("--row1", default="k4c30s,_hq2,4")
ap.add_argument("--row2", default="k2c35s,_k2hq,2")
args = ap.parse_args()

plt.rcParams.update({"font.size": 12, "font.family": "serif",
                     "mathtext.fontset": "cm", "axes.linewidth": 0.8})
rows = [args.row1.split(","), args.row2.split(",")]
fig, ax = plt.subplots(2, 3, figsize=(13.5, 8.4))

for i, (tag, rs, kap) in enumerate(rows):
    coords = np.load(f"results/mcfem_coords{rs}.npy")
    mean = np.load(f"results/mcfem_mean{rs}.npy"); var = np.load(f"results/mcfem_var{rs}.npy")
    pred = np.load(f"results/lowrank_pred_{tag}.npy")
    ref = mean**2 + var
    x, y = coords[:, 0], coords[:, 1]
    tri = mtri.Triangulation(x, y)
    cx = x[tri.triangles].mean(1); cy = y[tri.triangles].mean(1)
    tri.set_mask(cx**2 + cy**2 < (args.r0*1.03)**2)
    rel = np.linalg.norm(pred-ref)/np.linalg.norm(ref); corr = np.corrcoef(pred, ref)[0, 1]
    vmax = ref.max()
    for a, val, ttl in [(ax[i, 0], ref, rf"MC+FEM  $\mathbb{{E}}[u^2]$  ($\kappa={kap}$)"),
                        (ax[i, 1], pred, rf"TPINN  $\Sigma_\theta(p,p)$  ($\kappa={kap}$)")]:
        tc = a.tripcolor(tri, val, shading="gouraud", cmap="viridis", vmin=0, vmax=vmax)
        a.set_aspect("equal"); a.set_title(ttl); a.set_xticks([-1, 0, 1]); a.set_yticks([-1, 0, 1])
        plt.colorbar(tc, ax=a, fraction=0.046, pad=0.04)
    a = ax[i, 2]
    a.scatter(ref, pred, s=7, alpha=0.45, edgecolors="none", color="#2c7fb8")
    lim = max(ref.max(), pred.max()); a.plot([0, lim], [0, lim], "--", color="0.3", lw=1)
    a.set_xlabel(r"MC+FEM $\mathbb{E}[u^2]$"); a.set_ylabel(r"TPINN $\Sigma_\theta(p,p)$")
    a.set_title(fr"$\kappa={kap}$:  $\rho={corr:.3f}$,  rel. err ${rel*100:.1f}\%$")
    a.set_aspect("equal"); a.set_xlim(0, lim); a.set_ylim(0, lim)
    print(f"row kappa={kap} [{tag} vs {rs}]: corr {corr:.4f}, rel {rel:.3e}")

plt.tight_layout()
plt.savefig(args.out, bbox_inches="tight")
print(f"[fig] {args.out}")
