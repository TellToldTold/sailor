#!/usr/bin/env bash
# A Megatron-Core training run of one plan, with nsys traces for moe-comms' moetrace: validation_run.py (docstring).
# Submit from this directory, nodes = world / 4:
#
#     sbatch --nodes=1 validation_run.slurm QWEN3-30B-A3B --tp 1 --ep 4 --mbs 1 --gbs 16
#     sbatch --nodes=2 validation_run.slurm MIXTRAL-8X7B --tp 2 --pp 2 --ep 2 --mbs 1 --gbs 16 --layers 8
#
# Output: $SCRATCH/sailor-validation/<job>/. Then, from moe-comms:
#     python -m moetrace extract <dir> && python -m moetrace view <dir>/tables -o page.html

#SBATCH --job-name=sailor-val
#SBATCH --account=infra02
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
echo "job ${SLURM_JOB_ID} on ${SLURM_NNODES} node(s) (${SLURM_JOB_NODELIST}) git=$(git rev-parse --short HEAD)$(git diff --quiet HEAD || echo -dirty)"
python3 validation_run.py "$@"
