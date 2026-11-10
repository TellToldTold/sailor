import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # mcore/: models.py
from models import ENV, MODELS, NSYS, model_args, num_experts  # noqa: E402


def megatron_args(model, tp, mbs, iters, num_experts=None):
    args = model_args(model, 2, iters, num_experts) + [
        "--tensor-model-parallel-size", str(tp), "--pipeline-model-parallel-size", "1",
        "--micro-batch-size", str(mbs), "--global-batch-size", str(mbs),
    ]
    if tp > 1:
        args.append("--sequence-parallel")
    return args


def export_traces(out, name, tp, margs, steps, start):
    """<name>_rank<R>.sqlite (with the NCCL NVTX payloads: --include-blobs) and <name>.json."""
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
                        help="experts per layer instead of the model's E. Files and logs named <model>-E<num>")
    parser.add_argument("--nsys", action="store_true",
                        help="trace runs for moetrace: nsys on iterations iters-6..iters-2 of every rank, everything in <results-dir>/traces")
    args = parser.parse_args()
    
    assert not args.num_experts or num_experts(args.model), f"{args.model} has no experts"
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
                "--sailor-num-layers-original", str(MODELS[args.model]["num_layers"]),
                "--sailor-skip-iters", str(args.skip_iters)] + margs
            start = time.time()

            env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(str(g) for g in range(tp)), **ENV)
            
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
