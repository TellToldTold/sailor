"""Sweep tensor parallelism x micro-batch size with profile_mcore.py on one node (Megatron-Core counterpart of
third_party/Megatron-DeepSpeed/profile_all.py + profile.sh). Runs that do not fit (OOM) fail without a profile.

    python profile_sweep.py --model LLAMA-3-8 --gpu-type GH-96 --results-dir <dir> [--tp 1 2 4] [--mbs 1 2 4 8]

Then: python ../gather_profs.py --model-name <model> --gpu-type <gpu> --profile-dir <dir> --sailor-parent-dir <..>
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

COMMON = [
    "--normalization", "RMSNorm", "--swiglu", "--disable-bias-linear", "--position-embedding-type", "rope",
    "--untie-embeddings-and-output-weights", "--attention-dropout", "0", "--hidden-dropout", "0",
    # as our Alps training runs (moe-comms docs/status.md)
    "--cross-entropy-loss-fusion", "--cross-entropy-fusion-impl", "native",
]
# (Megatron arguments without --num-layers, transformer layers of the full model, vocabulary size, sequence length)
MODELS = {
    # parameters per layer = 218,112,000, as in the Megatron-DeepSpeed profile (llm_info.json)
    "LLAMA-3-8": (COMMON + [
        "--hidden-size", "4096", "--ffn-hidden-size", "14336", "--num-attention-heads", "32",
        "--group-query-attention", "--num-query-groups", "8", "--norm-epsilon", "1e-5", "--rotary-base", "500000",
    ], 32, 128256, 8192),
    # as moe-comms slurm/40_pretrain_qwen3.slurm; EP = ETP = 1 (all 128 experts on each GPU)
    "QWEN3-30B-A3B": (COMMON + [
        "--hidden-size", "2048", "--ffn-hidden-size", "6144", "--num-attention-heads", "32",
        "--group-query-attention", "--num-query-groups", "4", "--kv-channels", "128", "--qk-layernorm",
        "--norm-epsilon", "1e-6", "--rotary-base", "1000000",
        "--num-experts", "128", "--moe-router-topk", "8", "--moe-ffn-hidden-size", "768", "--moe-router-dtype", "fp32",
        "--moe-router-load-balancing-type", "aux_loss", "--moe-aux-loss-coeff", "1e-3", "--moe-grouped-gemm",
        "--moe-permute-fusion", "--moe-router-fusion", "--moe-token-dispatcher-type", "alltoall",
        "--expert-model-parallel-size", "1", "--expert-tensor-parallel-size", "1",
    ], 48, 151936, 4096),
}


def megatron_args(model, tp, mbs, iters):
    arch, _, vocab, seq = MODELS[model]
    args = arch + [
        "--num-layers", "2", "--seq-length", str(seq), "--max-position-embeddings", str(seq),
        "--tensor-model-parallel-size", str(tp), "--pipeline-model-parallel-size", "1",
        # one micro-batch per iteration; DP = 1 (world size = TP)
        "--micro-batch-size", str(mbs), "--global-batch-size", str(mbs), "--train-iters", str(iters),
        "--mock-data", "--tokenizer-type", "NullTokenizer", "--vocab-size", str(vocab),
        "--make-vocab-size-divisible-by", "1",  # keep the model's vocabulary (padding is to a multiple of TP)
        "--bf16", "--use-distributed-optimizer",
        "--lr", "1e-5", "--min-lr", "1e-6", "--lr-decay-style", "constant", "--clip-grad", "1.0",
        "--eval-iters", "0", "--eval-interval", str(iters), "--log-interval", "1",
    ]
    if tp > 1:
        args.append("--sequence-parallel")
    return args


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument("--gpu-type", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--tp", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--mbs", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--skip-iters", type=int, default=5)
    parser.add_argument("--port", type=int, default=29500)
    args = parser.parse_args()

    profiler = Path(__file__).resolve().parent / "profile_mcore.py"
    logs = Path(args.results_dir) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    failed = []
    for tp in args.tp:
        for mbs in args.mbs:
            cmd = [sys.executable, "-m", "torch.distributed.run", "--nproc-per-node", str(tp),
                   "--master-port", str(args.port), str(profiler),
                   "--sailor-model-name", args.model, "--sailor-gpu-type", args.gpu_type,
                   "--sailor-results-dir", args.results_dir,
                   "--sailor-num-layers-original", str(MODELS[args.model][1]),
                   "--sailor-skip-iters", str(args.skip_iters)] + megatron_args(args.model, tp, mbs, args.iters)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(str(g) for g in range(tp)),
                       CUDA_DEVICE_MAX_CONNECTIONS="1")
            log = logs / f"{args.model}_tp{tp}_mbs{mbs}.log"
            print(f"=== TP {tp} MBS {mbs} -> {log}", flush=True)
            with open(log, "w") as f:
                f.write(" ".join(cmd) + "\n")
                f.flush()
                code = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
            if code != 0:
                print(f"!!! TP {tp} MBS {mbs} failed (exit {code}; OOM expected at large MBS)", flush=True)
                failed.append((tp, mbs))
    print(f"done; failed: {failed or 'none'}")


if __name__ == "__main__":
    main()
