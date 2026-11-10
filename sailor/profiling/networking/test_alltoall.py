"""Network profiling of the MoE token all-to-all (A2A), for the A2A bandwidth curves of
docs/moe_iteration_time_model.md (section 2.2, F4/F6).

One process per GPU (launched by run_network_profile.slurm). For each EP size e it splits the world into
world/e groups of consecutive ranks (Megatron's EP layout, filling a node first) and runs the A2A in all
groups at once, so every node is loaded as in a training run. Per (e, V):

  * all_to_all_single on a bf16 tensor of V bytes per GPU, V split into e equal parts (balanced routing),
    with explicit split sizes, the call Megatron's alltoall dispatcher makes (mappings.py:_AllToAll);
  * warm-up, then per iteration: barrier over the world, CUDA events around the call;
  * time of an iteration = slowest rank; reported: median and min over the iterations;
  * twice: with the host launch inside the timing window (median_s, min_s), and GPU only (gpu_median_s,
    gpu_min_s: a sleep kernel keeps the GPU busy while the CPU launches the call). Which one training sees
    depends on whether the CPU runs ahead of the GPU at that call (check in the training traces).

It also times the per-expert count all-gather of the dispatcher (all_gather_into_tensor of E int64 over the
EP group, token_dispatcher.py:preprocess): latency only.

Raw results (V, time, layout) go to one JSON; fit_alltoall.py turns them into the curves.
"""

import argparse
import json
import os
import socket
import statistics
from datetime import datetime, timezone

import torch
import torch.distributed as dist


def init():
    rank = int(os.environ["SLURM_PROCID"])
    world_size = int(os.environ["SLURM_NTASKS"])
    local_rank = int(os.environ["SLURM_LOCALID"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", rank=rank, world_size=world_size, device_id=device)
    return rank, world_size, local_rank, device


SLEEP_CYCLES = 2_000_000  # ~1 ms on a GH200: longer than the host launch of one collective


def time_op(op, warmup, iters, local_rank, gpu_only):
    """Per-iteration time of op() on this rank, seconds. A barrier over the world before each iteration
    starts all groups together.
    gpu_only False: the window includes the host launch (after the barrier the GPU idles until the CPU has
    launched op). gpu_only True: a sleep kernel before the start event keeps the GPU busy while the CPU
    launches op, so the window holds only the GPU execution."""
    for _ in range(warmup):
        op()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        dist.barrier(device_ids=[local_rank])
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        if gpu_only:
            torch.cuda._sleep(SLEEP_CYCLES)
        start.record()
        op()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / 1e3)
    return times


def measure(op, args, local_rank, device):
    """{median_s, min_s} with the host launch inside the window, {gpu_median_s, gpu_min_s} without."""
    med, mn = slowest_rank(time_op(op, args.warmup, args.iters, local_rank, False), device)
    gmed, gmn = slowest_rank(time_op(op, args.warmup, args.iters, local_rank, True), device)
    return {"median_s": med, "min_s": mn, "gpu_median_s": gmed, "gpu_min_s": gmn}


def slowest_rank(times, device):
    """Per iteration the max over all ranks, then (median, min) over the iterations."""
    t = torch.tensor(times, dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    t = t.tolist()
    return statistics.median(t), min(t)


def main(args):
    rank, world_size, local_rank, device = init()
    hostnames = [None] * world_size
    dist.all_gather_object(hostnames, socket.gethostname())

    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_type": args.gpu_type,
        "world_size": world_size,
        "num_nodes": len(set(hostnames)),
        "hostnames": hostnames,
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "nccl": ".".join(map(str, torch.cuda.nccl.version())),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "git": os.environ.get("SAILOR_GIT"),
        "env_nccl": {k: v for k, v in sorted(os.environ.items()) if k.startswith(("NCCL_", "FI_"))},
        "warmup": args.warmup,
        "iters": args.iters,
        "num_experts": args.num_experts,
    }
    results = []

    for e in args.ep_sizes:
        if e > world_size or world_size % e:
            if rank == 0:
                print(f"skip e={e}: does not divide world size {world_size}")
            continue
        # every rank creates every group (collective), keeps its own
        for start in range(0, world_size, e):
            g_ = dist.new_group(list(range(start, start + e)))
            if start <= rank < start + e:
                group = g_
        members = [rank - rank % e + i for i in range(e)]
        # layout of this rank's group: GPUs on the busiest node (g) and nodes spanned; the same for every group
        per_node = [sum(hostnames[m] == h for m in members) for h in set(hostnames[m] for m in members)]
        layout = torch.tensor([max(per_node), len(per_node), -max(per_node), -len(per_node)], device=device)
        dist.all_reduce(layout, op=dist.ReduceOp.MAX)
        g, n_nodes = layout[0].item(), layout[1].item()
        assert (g, n_nodes) == (-layout[2].item(), -layout[3].item()), "EP groups with different layouts"
        row = {"e": e, "groups": world_size // e, "g": g, "n_nodes": n_nodes, "a2a": [], "count_allgather": None}

        # count all-gather: E int64 per rank
        cnt_in = torch.zeros(args.num_experts, dtype=torch.int64, device=device)
        cnt_out = torch.empty(args.num_experts * e, dtype=torch.int64, device=device)
        row["count_allgather"] = {"bytes": cnt_in.numel() * 8, **measure(
            lambda: dist.all_gather_into_tensor(cnt_out, cnt_in, group=group), args, local_rank, device)}

        for exp in range(args.min_log2, args.max_log2 + 1):
            elems = (2 ** exp // 2) // e * e  # bf16, divisible by e
            inp = torch.ones(elems, dtype=torch.bfloat16, device=device)
            out = torch.empty_like(inp)
            splits = [elems // e] * e
            m = measure(lambda: dist.all_to_all_single(out, inp, output_split_sizes=splits,
                                                       input_split_sizes=splits, group=group),
                        args, local_rank, device)
            v = elems * 2
            row["a2a"].append({"bytes": v, **m})
            if rank == 0:
                print(f"e={e:>2} g={g} nodes={n_nodes} V={v / 1e6:>10.3f} MB  median {m['median_s'] * 1e3:9.3f} ms"
                      f"  GPU only {m['gpu_median_s'] * 1e3:9.3f} ms", flush=True)
            del inp, out
            torch.cuda.empty_cache()
        results.append(row)
        dist.barrier(device_ids=[local_rank])

    if rank == 0:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"meta": meta, "results": results}, f, indent=2)
        print(f"wrote {args.out}")
    dist.barrier(device_ids=[local_rank])
    dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="output JSON (written by rank 0)")
    p.add_argument("--gpu-type", default="GH-96")
    p.add_argument("--ep-sizes", type=int, nargs="+", default=[2, 4, 8, 16])
    # V per GPU: 2^15 (32 KB, the probabilities A2A at b = 1, t = 4) .. 2^30 (1.07 GB, Qwen3 tokens at b = 8, t = 1)
    p.add_argument("--min-log2", type=int, default=15)
    p.add_argument("--max-log2", type=int, default=30)
    p.add_argument("--num-experts", type=int, default=128, help="E, for the size of the count all-gather")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=50)
    main(p.parse_args())
