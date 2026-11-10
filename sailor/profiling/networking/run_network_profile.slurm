#!/usr/bin/env bash
# Network profiling on Alps (Clariden), in the container the profiles and training runs use (../../mcore/alps6.toml).
# Submit from this directory:
#
#     sbatch --nodes=4 --time=00:20:00 run_network_profile.slurm a2a [test_alltoall.py options]   A2A, e = 2..16
#     (debug QOS: at most 90 node-minutes per job, so 4 nodes need --time <= 22 min; else QOSMaxNodeMinutesPerJob)
#     sbatch --nodes=2 run_network_profile.slurm p2p                              Sailor's send/recv curves
#
# a2a: test_alltoall.py on all nodes, 4 ranks per node, every EP group of the nodes at once ->
#      a2a_<N>n.json. Then fit_alltoall.py.
# p2p: the original test_sendrecv.py (via run_all.py), one srun step per configuration Sailor's coefficient
#      files are keyed by: intra-node with 2 and 4 GPUs, inter-node with 1, 2, 4 GPUs per node ->
#      p2p_intra_<g>.csv, p2p_inter_<g>.csv. Then fit_func.py on each (needs exactly 2 nodes).
#
# Output: $SCRATCH/sailor-network/<job>/.

#SBATCH --job-name=sailor-net
#SBATCH --account=infra02
#SBATCH --partition=debug
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
MODE="$1"; shift
export OUT="${SCRATCH}/sailor-network/${SLURM_JOB_ID}" EXTRA="$*"
export SAILOR_GIT="$(git rev-parse --short HEAD)$(git diff --quiet HEAD || echo -dirty)"
EDF="$(realpath ../../mcore/alps6.toml)"
mapfile -t HOSTS < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
export MASTER_ADDR="${HOSTS[0]}" MASTER_PORT=29500
mkdir -p "${OUT}"
echo "job ${SLURM_JOB_ID} mode ${MODE} on ${SLURM_JOB_NODELIST} git=${SAILOR_GIT} -> ${OUT}"

SRUN=(srun --mpi=pmix --network=disable_rdzv_get --environment="${EDF}")

case "${MODE}" in
a2a)
  "${SRUN[@]}" bash -c 'cd "${SLURM_SUBMIT_DIR}" && \
    python test_alltoall.py --out "${OUT}/a2a_${SLURM_NNODES}n.json" ${EXTRA}'
  ;;
p2p)
  [[ ${#HOSTS[@]} -eq 2 ]] || { echo "p2p needs --nodes=2"; exit 1; }
  # name nodes tasks-per-node; run_all.py writes data_<world>.csv into this directory
  for cfg in "intra_2 1 2" "intra_4 1 4" "inter_1 2 1" "inter_2 2 2" "inter_4 2 4"; do
    read -r name nodes tpn <<< "${cfg}"
    "${SRUN[@]}" -N "${nodes}" --ntasks-per-node="${tpn}" -w "$(IFS=,; echo "${HOSTS[*]:0:${nodes}}")" bash -c '
      cd "${SLURM_SUBMIT_DIR}" && python run_all.py --rank "${SLURM_PROCID}" --master_ip "${MASTER_ADDR}" \
        --world_size "${SLURM_NTASKS}" --gpus_per_node 4'
    mv "data_$((nodes * tpn)).csv" "${OUT}/p2p_${name}.csv"
  done
  ;;
*)
  echo "mode: a2a | p2p"; exit 1 ;;
esac
