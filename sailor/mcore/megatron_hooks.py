"""Hooks into Megatron for traced runs, copied from moe-comms (src/pretrain.py), so that moe-comms' moetrace reads
our traces as its own.

fast_nvtx_names: with --nvtx-ranges, Megatron names every NVTX range after its calling function through
inspect.getframeinfo / getmodule (~0.1 ms each, twice per range): CPU time that is not in an untraced run. Same names,
read directly from the caller's frame.

bind_cpus_to_gpu: pins the process to the CPUs of its GPU's NUMA node, as training launched with srun (one task per
GPU, bound) runs. Not from moe-comms.

count_routing: per MoE layer call, the router's token-expert pairs per destination GPU / node of the EP group, counted
on the GPU without host syncs, written to <out_dir>/<NAME>_rank<RANK>_routing.npz (moetrace: routing.parquet and the
alltoall bytes check).
"""

import os
import sys


def bind_cpus_to_gpu():
    """Pin this process to the CPUs of its GPU's NUMA node (GH200: GPU LOCAL_RANK <-> NUMA node LOCAL_RANK, 72 cores),
    within the CPUs it may use (Slurm's cpuset). Call before importing torch, so its threads inherit it.
    SAILOR_CPU_BIND=0 leaves the process unbound (OS placement). Returns a description for logs and profiles."""
    if os.environ.get("SAILOR_CPU_BIND", "1") == "0":
        return "unbound (SAILOR_CPU_BIND=0)"
    node = int(os.environ.get("LOCAL_RANK", "0"))
    path = f"/sys/devices/system/node/node{node}/cpulist"
    if not os.path.exists(path):
        return f"unbound (no {path})"
    cpus = set()
    for part in open(path).read().strip().split(","):
        lo, _, hi = part.partition("-")
        cpus.update(range(int(lo), int(hi or lo) + 1))
    allowed = cpus & os.sched_getaffinity(0)
    if not allowed:
        return f"unbound (NUMA node {node} outside this process' CPUs)"
    os.sched_setaffinity(0, allowed)
    return f"NUMA node {node}, CPUs {min(allowed)}-{max(allowed)} ({len(allowed)})"


def fast_nvtx_names():
    import megatron.core.utils as utils

    def caller_path():
        frame = sys._getframe(2)  # 0: here, 1: nvtx_range_push/pop, 2: its caller
        return f"{frame.f_globals['__name__']}.{frame.f_code.co_name}"

    utils._nvtx_range_get_func_path = caller_path


def count_routing(out_dir):
    import atexit

    import numpy as np
    import torch
    from megatron.core import parallel_state
    from megatron.core.num_microbatches_calculator import get_num_microbatches
    from megatron.core.transformer.moe.router import TopKRouter
    from megatron.training import get_args

    routers, rows, layers, tokens, state = [], [], [], [], {}
    path = os.path.join(out_dir, f"{os.environ['NAME']}_rank{os.environ['RANK']}_routing.npz")

    def save():
        if rows and not state.get("saved"):
            state["saved"] = True
            np.savez(path, counts=torch.stack(rows).cpu().numpy(), layer=np.array(layers), tokens=np.array(tokens),
                     ep_ranks=np.array(state["ep_ranks"]), ranks_per_node=state["per_node"],
                     calls_per_iteration=len(routers) * get_num_microbatches())

    def hook(router, inputs, outputs):
        routing_map = outputs[1].bool()  # [tokens, experts]
        if not state:
            ep_ranks = torch.distributed.get_process_group_ranks(parallel_state.get_expert_model_parallel_group())
            per_node = int(os.environ["LOCAL_WORLD_SIZE"])
            nodes = sorted({r // per_node for r in ep_ranks})
            onehot = torch.zeros(len(ep_ranks), len(nodes), device=routing_map.device)
            for i, r in enumerate(ep_ranks):
                onehot[i, nodes.index(r // per_node)] = 1
            state.update(ep_ranks=ep_ranks, per_node=per_node, onehot=onehot,
                         total=get_args().train_iters * len(routers) * get_num_microbatches())
        n, e = routing_map.shape
        per_gpu = routing_map.view(n, len(state["ep_ranks"]), -1)  # experts of one EP rank are contiguous
        in_gpu = per_gpu.any(2)
        rows.append(torch.cat([per_gpu.sum((0, 2)), in_gpu.sum(0), ((in_gpu.float() @ state["onehot"]) > 0).sum(0)]))
        layers.append(router.layer_number)
        tokens.append(n)
        if len(rows) == state["total"]:
            save()

    init = TopKRouter.__init__

    def init_with_hook(self, *args, **kwargs):
        init(self, *args, **kwargs)
        routers.append(self)
        self.register_forward_hook(hook)

    TopKRouter.__init__ = init_with_hook
    atexit.register(save)
