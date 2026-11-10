"""A Megatron-Core training run of one plan, to validate Sailor's simulator (docs/moe_iteration_time_model.md), with
Nsight Systems traces that moe-comms' moetrace reads as its own. Started by validation_run.slurm (on the batch host,
host python3 3.6: srun starts the ranks in the container):

    sbatch --nodes=N validation_run.slurm <MODEL> --tp T --pp P --ep E --mbs B --gbs G [--layers L] [--iters I]
                                          [--no-nsys] [--gpu-metrics-ranks "0"]

Same model and settings as the profiles (../models.py: the model's training config in Planner/simulations/configs,
model_args, ENV), and the model's regime: alltoall
dispatcher, ETP = 1, no --use-distributed-optimizer (ZeRO-0), no overlap flags, sequence parallelism when TP > 1, mock
data. DP = world / (TP x PP); world = nodes x 4.

Output in <out> ($SCRATCH/sailor-validation/<job>/), as moe-comms' slurm/40_pretrain_qwen3.slurm:
  <name>.log                 the run's output (Megatron logs every iteration)
  <name>.json                manifest: arguments, nodes in rank order, Dragonfly groups, profiled steps, start/end,
                             and for validation: the plan and Megatron's per-iteration times
  <name>_rank<R>.nsys-rep / .sqlite   nsys of every rank on iterations iters-6..iters-2 (sqlite with NCCL payloads)
  <name>_rank<R>_routing.npz routing counts per MoE layer call (megatron_hooks.count_routing)
  <name>_plan_config.json    the plan and measured time in the format of Sailor's validation
                             (Planner/simulations/validation/clariden/<MODEL>/N<n>/plan_config_*.json: "real" = median
                             iteration time in s), plus "ep" (not read by the simulator yet); red_grads, max_mem: null
Then, from moe-comms: python -m moetrace extract <out> && python -m moetrace view <out>/tables -o page.html

Iteration time for validation: median of Megatron's "elapsed time per iteration" over the iterations after the first
5 and before nsys starts (untraced; with --no-nsys: all after the first 5).
"""

import argparse
import json
import os
import re
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # mcore/: models.py
from models import ENV, MODELS, NSYS, model_args, num_experts  # noqa: E402

GPUS_PER_NODE = 4
WARMUP = 5  # iterations left out of the iteration time


def plan_args(a, world):
    """Megatron arguments of the plan, after checking it."""
    experts = num_experts(a.model)
    layers = a.layers or MODELS[a.model]["num_layers"]
    tp, pp, ep = a.tp, a.pp, a.ep
    assert GPUS_PER_NODE % tp == 0, f"TP {tp} must divide the {GPUS_PER_NODE} GPUs of a node"
    assert world % (tp * pp) == 0, f"world {world} not divisible by TP x PP = {tp * pp}"
    dp = world // (tp * pp)
    assert layers % pp == 0, f"{layers} layers not divisible by PP {pp}"
    assert experts or ep == 1, f"{a.model} has no experts: EP must be 1"
    if experts:
        assert experts % ep == 0, f"EP {ep} must divide the {experts} experts"
        assert (tp * dp) % ep == 0, f"EP {ep} must divide TP x DP = {tp * dp}"
    assert a.gbs % (a.mbs * dp) == 0, f"GBS {a.gbs} not divisible by MBS x DP = {a.mbs * dp}"
    args = model_args(a.model, layers, a.iters) + [
        "--tensor-model-parallel-size", str(tp), "--pipeline-model-parallel-size", str(pp),
        "--micro-batch-size", str(a.mbs), "--global-batch-size", str(a.gbs),
    ]
    if experts:
        args[args.index("--expert-model-parallel-size") + 1] = str(ep)
    if tp > 1:
        args.append("--sequence-parallel")
    plan = {"model": a.model, "tp": tp, "pp": pp, "ep": ep, "dp": dp, "mbs": a.mbs, "gbs": a.gbs, "layers": layers,
            "micro_batches": a.gbs // (a.mbs * dp), "nodes": world // GPUS_PER_NODE, "world": world}
    return args, plan


def sh(cmd):
    return subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, universal_newlines=True).stdout


def dragonfly(nodes):
    """Node -> Dragonfly group (Slurm topology, level-0 switches), as moe-comms."""
    out = {}
    for line in sh("scontrol show topology").splitlines():
        if "Level=0" not in line:
            continue
        switch = re.search(r"SwitchName=(\S+)", line).group(1)
        members = re.search(r"Nodes=(\S+)", line).group(1)
        for n in sh("scontrol show hostnames " + members).split():
            if n in nodes:
                out[n] = switch
    return out


def sailor_plan_config(plan, real_s):
    """Sailor's validation plan: layer keys 0 embedding, 1..L transformer layers, L+1 final norm, L+2 output layer
    (as the profiles); Megatron splits the L layers evenly over PP; every DP replica of a stage has the same TP."""
    L, pp, per = plan["layers"], plan["pp"], plan["layers"] // plan["pp"]
    stages = [list(range(1 + s * per, 1 + (s + 1) * per)) for s in range(pp)]
    stages[0].insert(0, 0)
    stages[-1] += [L + 1, L + 2]
    gpu = [[["GH-96", GPUS_PER_NODE, "us-central1-a"]], plan["tp"]]  # zone name of Sailor's GH-96 coefficients
    return {"zone": "us-central1-a",
            "pipeline_list": [{"num_stages": pp, "layers_per_stage": stages,
                               "tmp_per_stage": [[gpu] * plan["dp"] for _ in range(pp)], "dp": [plan["dp"]] * pp}],
            "mbs": plan["mbs"], "gbs": plan["gbs"], "real": real_s, "red_grads": None, "max_mem": None,
            "N": plan["nodes"], "ep": plan["ep"], "model": plan["model"]}


def iteration_times(log):
    """Megatron's elapsed time per iteration (ms), {iteration: ms}."""
    times = {}
    for m in re.finditer(r"iteration\s+(\d+)/\s*\d+ \|.*?elapsed time per iteration \(ms\): ([\d.]+)", log):
        times[int(m.group(1))] = float(m.group(2))
    return times


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", choices=sorted(MODELS))
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--pp", type=int, default=1)
    p.add_argument("--ep", type=int, default=1)
    p.add_argument("--mbs", type=int, required=True)
    p.add_argument("--gbs", type=int, required=True)
    p.add_argument("--layers", type=int, help="transformer layers (default: the full model)")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--no-nsys", action="store_true", help="no traces (and no routing counts)")
    p.add_argument("--gpu-metrics-ranks", default="", help='ranks with nsys GPU metrics (10 kHz), e.g. "0"')
    p.add_argument("--out", help="default: $SCRATCH/sailor-validation/<job>")
    a = p.parse_args()

    job = os.environ["SLURM_JOB_ID"]
    nodes = sh(f"scontrol show hostnames {os.environ['SLURM_JOB_NODELIST']}").split()
    world = len(nodes) * GPUS_PER_NODE
    args, plan = plan_args(a, world)
    nsys = not a.no_nsys
    steps = [a.iters - 6, a.iters - 2]
    assert not nsys or steps[0] > WARMUP, f"--iters {a.iters}: nsys starts at iteration {steps[0]}, after {WARMUP}"
    if nsys:
        args += ["--profile", "--profile-step-start", str(steps[0]), "--profile-step-end", str(steps[1]),
                 "--profile-ranks", *map(str, range(world)), "--nvtx-ranges"]

    name = (f"{a.model.lower()}_tp{plan['tp']}_pp{plan['pp']}_ep{plan['ep']}_dp{plan['dp']}_mbs{a.mbs}"
            f"_gbs{a.gbs}_{len(nodes)}n_{job}")
    out = Path(a.out or Path(os.environ["SCRATCH"]) / "sailor-validation" / job)
    out.mkdir(parents=True, exist_ok=True)
    edf = HERE.parent / "alps6.toml"
    gm_ranks = a.gpu_metrics_ranks.split()
    print(f"=== {name} on {' '.join(nodes)}\n    plan {plan}\n    out {out}", flush=True)

    env = dict(os.environ, **ENV)
    env.update(MASTER_ADDR=nodes[0], MASTER_PORT="29500", NAME=name, OUT=str(out), MEGATRON_ARGS=" ".join(args),
               TRAIN=str(HERE / "train_mcore.py"), NSYS_ON="1" if nsys else "0", GM_RANKS=" ".join(gm_ranks),
               NSYS_CMD=" ".join(NSYS))
    if nsys:
        env["ROUTING_OUT"] = str(out)
    # one task per GPU; nsys per rank (GPU metrics on GM_RANKS), as moe-comms slurm/40_pretrain_qwen3.slurm
    rank_cmd = r'''
      export RANK=${SLURM_PROCID} WORLD_SIZE=${SLURM_NTASKS} LOCAL_RANK=${SLURM_LOCALID} LOCAL_WORLD_SIZE=${SLURM_NTASKS_PER_NODE}
      PROFILE=()
      if [[ "${NSYS_ON}" == 1 ]]; then
        PROFILE=(${NSYS_CMD} -o "${OUT}/${NAME}_rank${RANK}")
        [[ " ${GM_RANKS} " == *" ${RANK} "* ]] && PROFILE+=(--gpu-metrics-devices="${LOCAL_RANK}" --gpu-metrics-frequency=10000)
      fi
      exec "${PROFILE[@]}" python "${TRAIN}" ${MEGATRON_ARGS}
    '''
    srun = ["srun", "--mpi=pmix", "--network=disable_rdzv_get", f"--environment={edf}"]
    start = time.time()
    with open(out / f"{name}.log", "w") as log:
        log.write(" ".join(args) + "\n")
        log.flush()
        code = subprocess.run(srun + ["bash", "-c", rank_cmd], env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    if code != 0:
        print(f"!!! {name} failed (exit {code}), see {out / (name + '.log')}", flush=True)

    # own step, as moe-comms: the default export drops the NCCL NVTX payloads (communicator, bytes, peer)
    if nsys:
        export = r'''R="${OUT}/${NAME}_rank${SLURM_PROCID}"
          nsys export --type=sqlite --include-blobs=true --force-overwrite=true -o "${R}.sqlite" "${R}.nsys-rep" > /dev/null'''
        if subprocess.run(srun[:1] + [f"--environment={edf}", "bash", "-c", export], env=env).returncode != 0:
            print(f"!!! export of {name} failed", flush=True)

    times = iteration_times((out / f"{name}.log").read_text())
    last = steps[0] - 1 if nsys else a.iters
    untraced = [ms for it, ms in sorted(times.items()) if WARMUP < it <= last]
    median = statistics.median(untraced) if untraced else None
    manifest = {
        "name": name, "job": job, "variant_args": [], "args": args, "nodes": nodes, "ranks_per_node": GPUS_PER_NODE,
        "dragonfly": dragonfly(nodes), "profile_steps": steps if nsys else [], "gpu_metrics_ranks": [int(r) for r in gm_ranks],
        "start": start, "end": time.time(),
        # for validation (not read by moetrace)
        "plan": plan, "exit_code": code, "host": socket.gethostname(),
        "iteration_ms": times, "iteration_ms_untraced": [WARMUP + 1, last],
        "iteration_ms_median_untraced": median,
    }
    (out / f"{name}.json").write_text(json.dumps(manifest, indent=1))
    real = round(median / 1000, 5) if median else None
    (out / f"{name}_plan_config.json").write_text(json.dumps(sailor_plan_config(plan, real), indent=1))
    print(f"done: exit {code}, median iteration {median} ms over iterations {WARMUP + 1}..{last} -> {out}", flush=True)


if __name__ == "__main__":
    main()
