# Reproducing the TensorPINNs experiments

All experiments run inside a Docker image with DeepXDE, TensorFlow and (for the Monte-Carlo
finite-element reference) FEniCSx/dolfinx. A global random seed of `1234` is used throughout.

## Environment

**DeepXDE / TPINN experiments** — image `pescapil/deepxde:latest` (Python 3.8, DeepXDE 1.8.4,
TensorFlow 2.9.2). The image's entrypoint launches Jupyter, so override it with `python`:

```bash
docker pull pescapil/deepxde:latest
docker run --rm --gpus '"device=0"' -e MPLBACKEND=Agg -e DDE_BACKEND=tensorflow.compat.v1 \
  -v "$PWD":/work -w /work --entrypoint python pescapil/deepxde:latest <script.py> [args]
```

Pick an idle GPU first with `nvidia-smi` (this is a shared node with no scheduler).

**Monte-Carlo finite-element baseline** — image `dolfinx/dolfinx:stable` (dolfinx 0.11):

```bash
docker run --rm -v "$PWD":/work -w /work dolfinx/dolfinx:stable python3 <mcfem_script.py>
```

## Running each case

| Paper section | Script |
|---|---|
| Poisson 1D (V-TPINN / MO-TPINN) | `Poisson/tensoruq.py`, `Poisson/mo-tensoruq.py` (`--k 1..4`) |
| Helmholtz 2D | `Helmholtz/helmholtz1.py`, `helmholtz2.py`, `mo-helmholtz.py`, `*-kernel*` |
| Heat 1D | `Heat/heat1.ipynb`, `heat2.ipynb`, `heat2-mo.ipynb`, `heat2-*-kernel*.ipynb` |

Each script sets `dde.config.set_random_seed(1234)`, prints the training/test loss, the relative
error and the parameter count, and saves the predicted vs.\ exact diagonal to `results/`.

## Notes / fixes for reproducibility

- Parameter counting uses `tensorflow.compat.v1.trainable_variables()` (the older
  `model.net.trainable_variables` attribute was removed in recent DeepXDE).
- Multi-output (MO) scripts slice the last network output (`[:, k-1:k]`) before reshaping the
  diagonal grid for the final error/plot; single-output (V-TPINN) scripts do not.
- Manufactured solutions: Helmholtz uses `u = sin(k0 x) sin(k0 y)`, `f = k0^2 sin(k0 x) sin(k0 y)`
  with `k0 = 2*pi*n` (amplitude 1); Heat uses `u = sin(n*pi*x/L) exp(-n^2*pi^2*a*t/L^2)`.
