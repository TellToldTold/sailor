#!/usr/bin/env bash
# Sailor model profile on one Alps (Clariden) GH200 node with Megatron-Core: profile_sweep.py (TP 1/2/4 x MBS 1-8)
# in the alps6 container (alps6.toml). Submit from this directory:
#
#     sbatch profile_sweep.slurm <MODEL> [profile_sweep.py options]     e.g. sbatch profile_sweep.slurm LLAMA-3-8
#
# Profiles and per-run logs: $SCRATCH/sailor-profiles/<MODEL>/<job>/. Then ../gather_profs.py (docs/PROFILING.md).

#SBATCH --job-name=sailor-prof
#SBATCH --account=infra02
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export MODEL="$1"; shift
export OUT="${SCRATCH}/sailor-profiles/${MODEL}/${SLURM_JOB_ID}" EXTRA="$*"
echo "job ${SLURM_JOB_ID} on $(hostname) git=$(git rev-parse --short HEAD)$(git diff --quiet HEAD || echo -dirty) -> ${OUT}"

srun --environment="$(realpath alps6.toml)" bash -c '
  cd "${SLURM_SUBMIT_DIR}"
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  python profile_sweep.py --model "${MODEL}" --gpu-type GH-96 --results-dir "${OUT}" ${EXTRA}
'
