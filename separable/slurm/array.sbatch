#!/bin/bash
#SBATCH -p compute-gpu
#SBATCH --gres=gpu:a30mig:1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 24:00:00
#SBATCH -o logs/%x_%A_%a.out
ROOT=${TPINN_ROOT:-$SLURM_SUBMIT_DIR}
export TPINN_OUT=${TPINN_OUT:-$ROOT/out/$SLURM_JOB_NAME} XLA_PYTHON_CLIENT_PREALLOCATE=false
export TPINN_CACHE=${TPINN_CACHE:-$ROOT/cache}
cd "$ROOT/${TPINN_CODE:-code}"
exec "$ROOT/venv/bin/python" -m ${TPINN_MODULE:-tpinn.run} $(sed -n "${SLURM_ARRAY_TASK_ID}p" "$1")
