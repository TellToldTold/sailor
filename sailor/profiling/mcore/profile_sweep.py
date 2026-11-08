import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

# nsys per rank, started/stopped by Megatron (--profile: cudaProfilerStart/Stop), as moe-comms
# slurm/40_pretrain_qwen3.slurm, so that moe-comms' moetrace (extract, view) reads the traces unchanged
NSYS = ["nsys", "profile", "-t", "cuda,nvtx", "--sample=none", "--cpuctxsw=none",
        "--capture-range=cudaProfilerApi", "--capture-range-end=stop", "--force-overwrite=true"]

COMMON = [
    "--normalization", "RMSNorm", "--swiglu", "--disable-bias-linear", "--position-embedding-type", "rope",
    "--untie-embeddings-and-output-weights", "--attention-dropout", "0", "--hidden-dropout", "0",
    # as Megatron's Llama-3 recipe (examples/llama/train_llama3_8b_h100_fp8.sh); default impl is native
    "--cross-entropy-loss-fusion",
]
# (Megatron arguments without --num-layers, transformer layers of the full model, vocabulary size, sequence length)
MODELS = {
    # Llama-3-8B: 218,112,000 parameters per layer, as Planner/simulations/utils.py. The Megatron-DeepSpeed profile
    # (llm_info.json) has ~268.5M: multi-head attention, FFN 4h = 16384 and biases.
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


def megatron_args(model, tp, mbs, iters, num_experts=None):
    arch, _, vocab, seq = MODELS[model]
    if num_experts:
        arch = list(arch)
        arch[arch.index("--num-experts") + 1] = str(num_experts)
    args = arch + [
        "--num-layers", "2", "--seq-length", str(seq), "--max-position-embeddings", str(seq),
        "--tensor-model-parallel-size", str(tp), "--pipeline-model-parallel-size", "1",
        # one micro-batch per iteration; DP = 1 (world size = TP)
        "--micro-batch-size", str(mbs), "--global-batch-size", str(mbs), "--train-iters", str(iters),
        "--mock-data", "--tokenizer-type", "NullTokenizer", "--vocab-size", str(vocab),
        # optimizer not sharded over DP (ZeRO-0, as original Sailor); bf16 gradient buffer, as the planner assumes
        "--bf16", "--grad-reduce-in-bf16",
        "--lr", "1e-5", "--min-lr", "1e-6", "--lr-decay-style", "constant", "--clip-grad", "1.0",
        "--eval-iters", "0", "--eval-interval", str(iters), "--log-interval", "1",
    ]
    if tp > 1:
        args.append("--sequence-parallel")
    return args


def export_traces(out, name, tp, margs, steps, start):
    """moetrace's input: <name>_rank<R>.sqlite (with the NCCL NVTX payloads: --include-blobs) and <name>.json."""
    for rank in range(tp):
        rep = out / f"{name}_rank{rank}.nsys-rep"
        if not rep.exists():
            print(f"!!! no trace {rep}", flush=True)
            continue
        subprocess.run(["nsys", "export", "--type=sqlite", "--include-blobs=true", "--force-overwrite=true",
                        "-o", str(rep.with_suffix(".sqlite")), str(rep)], stdout=subprocess.DEVNULL, check=False)
    manifest = {"name": name, "job": os.environ.get("SLURM_JOB_ID"), "args": margs, "variant_args": [],
                "nodes": [socket.gethostname().split(".")[0]], "ranks_per_node": tp, "profile_steps": steps,
                "gpu_metrics_ranks": [], "start": start, "end": time.time()}
    (out / f"{name}.json").write_text(json.dumps(manifest, indent=1))


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
    parser.add_argument("--num-experts", type=int,
                        help="MoE models: experts per layer instead of the model's E, to time the experts as at "
                             "EP = E / this (same tokens per GPU, E/EP experts with EP times the tokens each; "
                             "layer_parts['experts'] in the profile). Files and logs named <model>-E<num>")
    parser.add_argument("--nsys", action="store_true",
                        help="trace runs for moetrace: nsys on iterations iters-6..iters-2 of every rank, everything "
                             "(also the profile JSON, disturbed by nsys) in <results-dir>/traces, not read by "
                             "gather_profs.py")
    args = parser.parse_args()
    assert not args.num_experts or "--num-experts" in MODELS[args.model][0], f"{args.model} has no experts"
    model_name = f"{args.model}-E{args.num_experts}" if args.num_experts else args.model
    assert not args.nsys or args.iters >= 8, "--nsys traces iterations iters-6..iters-2: needs --iters >= 8"

    profiler = Path(__file__).resolve().parent / "profile_mcore.py"
    out = Path(args.results_dir) / "traces" if args.nsys else Path(args.results_dir)
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    failed = []
    steps = [args.iters - 6, args.iters - 2]

    for tp in args.tp:
        for mbs in args.mbs:
            margs = megatron_args(args.model, tp, mbs, args.iters, args.num_experts)
            launch = [sys.executable, "-m", "torch.distributed.run", "--nproc-per-node", str(tp),
                      "--master-port", str(args.port)]
            run = [str(profiler)]
            if args.nsys:
                name = f"{model_name}_tp{tp}_mbs{mbs}"
                margs += ["--profile", "--profile-step-start", str(steps[0]), "--profile-step-end", str(steps[1]),
                          "--profile-ranks", *map(str, range(tp)), "--nvtx-ranges"]
                # torchrun starts nsys per rank, nsys starts Python; %q{RANK}: the rank torchrun sets
                launch.append("--no-python")
                run = NSYS + ["-o", str(out / f"{name}_rank%q{{RANK}}"), sys.executable] + run
            cmd = launch + run + [
                "--sailor-model-name", model_name, "--sailor-gpu-type", args.gpu_type,
                "--sailor-results-dir", str(out),
                "--sailor-num-layers-original", str(MODELS[args.model][1]),
                "--sailor-skip-iters", str(args.skip_iters)] + margs
            start = time.time()

            env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(str(g) for g in range(tp)),
                       CUDA_DEVICE_MAX_CONNECTIONS="1")
            
            log = logs / f"{model_name}_tp{tp}_mbs{mbs}.log"
            print(f"=== TP {tp} MBS {mbs} -> {log}", flush=True)

            with open(log, "w") as f:
                f.write(" ".join(cmd) + "\n")
                f.flush()
                code = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
            
            if code != 0:
                print(f"!!! TP {tp} MBS {mbs} failed (exit {code})", flush=True)
                failed.append((tp, mbs))
            if args.nsys:
                export_traces(out, name, tp, margs, steps, start)

    print(f"done; failed: {failed or 'none'}")


if __name__ == "__main__":
    main()
