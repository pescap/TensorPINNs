# Separable TPINNs (JAX)

JAX implementation of the three TPINN architectures used in the revised manuscript:
vanilla (V), multi-output (MO) and separable low-rank (LR), together with the classical
baselines, the references and the scripts that regenerate every table and figure.

LR-TPINNs represent the k-th moment as a symmetric CP tensor with factors tied across the
k copies of the domain, `Sigma(x_1, ..., x_k) = sum_r c_r phi_r(x_1) ... phi_r(x_k)`, so the
tensor operator acts on the factor network only.

## Install

```
python -m venv venv
venv/bin/pip install -r requirements.txt
```

For GPUs, install the CUDA build of JAX (`jax[cuda12]==0.10.2`). All runs use double precision.

## Run

One experiment writes a JSON (errors, timings, losses) and an NPZ (diagonal) to `TPINN_OUT`:

```
TPINN_OUT=out/demo venv/bin/python -m tpinn.run --problem logn --arch lr --k 4 --rank 32 --peak 1
```

Main options: `--problem {sep,logn,helm,heat,box,ball,hole}`, `--arch {v,mo,lr}`, `--k`,
`--rank`, `--grid M` (tensor-product collocation, k=2), `--peak 1` (peak sampling),
`--bc {hard,soft}` (soft: penalty on tuples with one point on the boundary, weight `--wbc`,
or adaptive weights with `--balance 1`; sweeps `soft` and `softbal`). `python -m tpinn.heat_ic` runs the heat equation with a random initial
condition.

Classical baselines, each writing to `out/<name>/`:

| Module | Method |
|---|---|
| `tpinn.classical` | dense tensor FD, sparse-tensor combination technique, Monte Carlo (1D lognormal) |
| `tpinn.classical_tt` | TT-cross on the data moment with the inverse applied per core |
| `tpinn.classical_box` | sparse grid and sparse tensor on `[0,1]^d` |
| `tpinn.classical_ball` | stochastic collocation and Monte Carlo with isoparametric P2 FEM on the ball |
| `tpinn.sparse_ball` | sparse-tensor FEM (combination technique) on the ball, k=2; `--top 5` adds the fourth mesh (8 h on 32 cores) |

## Sweeps

`slurm/tasks.py <name>` prints one command line per task (`paper`, `pilot`, `peak`, `ball`,
...). `slurm/array.sbatch` runs one line per array index from the submission directory,
which holds `venv/`, `logs/` and `out/`; `TPINN_CODE` is the code directory relative to it:

```
mkdir -p logs
venv/bin/python separable/slurm/tasks.py paper > tasks.txt
TPINN_CODE=separable sbatch --export=ALL --array=1-$(wc -l < tasks.txt) separable/slurm/array.sbatch $PWD/tasks.txt
```

Without SLURM, run each line of `tasks.txt` with `python -m tpinn.run`.

The table generator reads the `peakis` tasks from two output folders, `out/peakis` and `out/peaka30`
(the same tasks on two GPU models), and the finer dense runs from `out/classical_dense`.

## Tables and figures

```
venv/bin/python -m tpinn.tables out tables
venv/bin/python -m tpinn.paper_figs out figs
venv/bin/python -m tpinn.report out > RESULTS.md
```

## Tests

```
venv/bin/python -m pytest tests
```
