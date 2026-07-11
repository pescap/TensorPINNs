# Complex geometry and low-rank TPINNs

Reproduces the complex-geometry second-moment experiment and the exact-reference low-rank
validation of the Neurocomputing revision.

Domain `D = [-1,1]^2 \ disk(r0=0.3)` (square with a circular hole), coercive screened-Poisson
operator `A = -Delta + kappa^2`, lognormal random source. The deterministic second-moment equation
`A^(2) Sigma = C^2` lives on the 4-D product domain `D x D`.

## Scripts

- `mcfem_helmholtz.py` — Monte-Carlo + FEM reference (dolfinx). Env vars `KAPPA`, `NMC`, `SUF`
  (output suffix). Run in the `dolfinx/dolfinx:stable` image. Writes
  `results/mcfem_{mean,var,coords}<SUF>.npy`.
- `lowrank_vtpinn.py` — low-rank (canonical-polyadic) hard-BC V-TPINN for the second moment, single
  loss, symmetric tied factors `Sigma = sum_r phi_r(p1) phi_r(p2)` (SPD diagonal), Adam + L-BFGS.
  Compares the diagonal `Sigma(p,p)` to `E[u]^2 + Var(u)`. Args: `--kappa --R --cw --refsuf --lbfgs
  --sym ...`. Reaches rel.err ~3-4%, correlation >= 0.999 (both kappa=2 and kappa=4).
- `lowrank_check_1d_v2.py` — 1-D exact-reference check: same ansatz on `D=[0,1]`, compared to the
  exact tensor finite-difference solution `Sigma = L^-1 C^2 L^-T`. Full-field rel.err ~2e-4,
  correlation 1.0000. Writes the validation figure `results/lowrank_1d_exact.pdf`.
- `pub_fig_both.py` — publication figure: MC+FEM vs low-rank TPINN diagonal for kappa=4 and kappa=2.
- `analyze_lowrank.py` — spatial-breakdown diagnostics (near-hole vs interior).

## Environment

TPINN scripts: `pescapil/deepxde:latest` (DeepXDE 1.8.4, TensorFlow 2.9.2). MC+FEM reference:
`dolfinx/dolfinx:stable` (v0.11). Global seed 1234. Pick a free GPU with `nvidia-smi` and pass
`--gpus '"device=<i>"'` (add `-e TF_FORCE_GPU_ALLOW_GROWTH=true` on the shared node).
