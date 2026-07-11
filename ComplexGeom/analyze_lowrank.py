"""
Diagnose + plot a low-rank V-TPINN complex-geometry run against its MC+FEM reference.
Reports overall corr/rel.err and a spatial breakdown (near-hole ring vs interior vs outer),
and writes a 3-panel figure: MC+FEM E[u^2] | TPINN diagonal | scatter.  Runs on CPU.
  python analyze_lowrank.py --tag k2c35s --refsuf _k2
"""
import argparse, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--refsuf", default="")
ap.add_argument("--r0", default=0.3, type=float)
args = ap.parse_args()
RS = args.refsuf

coords = np.load(f"results/mcfem_coords{RS}.npy")
mean = np.load(f"results/mcfem_mean{RS}.npy"); var = np.load(f"results/mcfem_var{RS}.npy")
pred = np.load(f"results/lowrank_pred_{args.tag}.npy")
ref = mean**2 + var

rel = np.linalg.norm(pred-ref)/np.linalg.norm(ref); corr = np.corrcoef(pred, ref)[0,1]
print(f"[{args.tag}] OVERALL diag rel.err {rel:.3e} | corr {corr:.4f}")

# spatial breakdown by distance from the hole boundary
rr = np.sqrt(coords[:,0]**2 + coords[:,1]**2)
bands = [("near-hole  r<r0+0.25", rr < args.r0+0.25),
         ("mid        r0+0.25..0.7", (rr>=args.r0+0.25) & (rr<0.7)),
         ("outer      r>=0.7", rr>=0.7)]
for name, msk in bands:
    if msk.sum() < 5: continue
    c = np.corrcoef(pred[msk], ref[msk])[0,1]
    r_ = np.linalg.norm(pred[msk]-ref[msk])/np.linalg.norm(ref[msk])
    ratio = pred[msk].mean()/ref[msk].mean()
    print(f"  {name}: corr {c:.3f} | rel {r_:.3f} | mean-ratio {ratio:.2f} | n={msk.sum()}")

fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
vmax = ref.max()
for a, val, ttl in [(ax[0], ref, "MC+FEM  E[u^2]=mean^2+var"), (ax[1], pred, f"TPINN diag [{args.tag}]")]:
    sc = a.scatter(coords[:,0], coords[:,1], c=val, s=8, cmap="viridis", vmin=0, vmax=vmax)
    a.set_aspect("equal"); a.set_title(ttl); plt.colorbar(sc, ax=a, fraction=0.046)
ax[2].scatter(ref, pred, s=6, alpha=0.5)
lim = max(ref.max(), pred.max()); ax[2].plot([0,lim],[0,lim],"r--",lw=1)
ax[2].set_xlabel("MC+FEM ref"); ax[2].set_ylabel("TPINN pred")
ax[2].set_title(f"corr {corr:.3f} | rel.err {rel:.3f}"); ax[2].set_aspect("equal")
plt.tight_layout(); out = f"results/fig_lowrank_{args.tag}.png"; plt.savefig(out, dpi=130)
print(f"[fig] {out}")
