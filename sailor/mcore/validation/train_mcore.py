"""One rank of a Megatron-Core training run (third_party/Megatron-LM pretrain_gpt.py, unchanged), started by
validation_run.py through srun: python train_mcore.py <Megatron arguments>.

Each rank is pinned to its GPU's NUMA node (megatron_hooks.bind_cpus_to_gpu; SAILOR_CPU_BIND=0: unbound).
With --nvtx-ranges: fast NVTX names (megatron_hooks.fast_nvtx_names). With ROUTING_OUT set: routing counts
(megatron_hooks.count_routing, needs NAME). RANK, WORLD_SIZE, LOCAL_RANK, LOCAL_WORLD_SIZE, MASTER_ADDR / PORT from
the launcher.
"""

import os
import runpy
import sys
from pathlib import Path

MEGATRON = Path(__file__).resolve().parents[3] / "third_party" / "Megatron-LM"
sys.path.insert(0, str(MEGATRON))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # mcore/: megatron_hooks.py

import megatron_hooks  # noqa: E402

CPU_BIND = megatron_hooks.bind_cpus_to_gpu()  # before torch (imported by Megatron below): threads inherit it

if __name__ == "__main__":
    print(f"[sailor-validation] rank {os.environ.get('RANK', '0')} CPU binding: {CPU_BIND}", flush=True)
    if "--nvtx-ranges" in sys.argv:
        megatron_hooks.fast_nvtx_names()
    if os.environ.get("ROUTING_OUT"):
        megatron_hooks.count_routing(os.environ["ROUTING_OUT"])
    script = str(MEGATRON / "pretrain_gpt.py")
    sys.argv = [script] + sys.argv[1:]
    runpy.run_path(script, run_name="__main__")
