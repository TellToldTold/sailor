"""What every Megatron-Core run of ours is: the models, the settings shared by profiles and validation runs
(model_args), the runs' environment (ENV) and the nsys command for traced runs (NSYS).
Used by profiler/profile_sweep.py and validation/validation_run.py.

The models are defined once, in Sailor's training configs (Planner/simulations/configs/training_config_<model>.json),
which the simulator and the planner read too: shapes (hidden_size, num_layers, heads, head_dim, ffn_hidden_size,
num_query_groups, vocab_size, sequence_length, ...), MoE (num_experts, moe_router_topk, moe_ffn_hidden_size,
moe_aux_loss_coeff) and the job (global_batch_size). Here only how we run them: implementation flags.
"""

import json
from pathlib import Path

CONFIGS = Path(__file__).resolve().parents[1] / "Planner" / "simulations" / "configs"
# models we run with Megatron-Core: name -> training config. Notes on the configs (JSON has no comments):
# - LLAMA-3-8: Llama-3-8B, 218,112,000 parameters per layer (as Planner/simulations/utils.py). The old
#   Megatron-DeepSpeed profile (llm_info.json) had ~268.5M: multi-head attention, FFN 4h = 16384 and biases.
# - QWEN3-30B-A3B: as moe-comms slurm/40_pretrain_qwen3.slurm (Hugging Face config: head_dim 128, qk-norm).
# - MIXTRAL-8X7B: Hugging Face config.json, 46.7B parameters, 12.9B active; sequence 4096 as Megatron's Mixtral
#   example (the model supports 32k). --num-experts profiles down to E/e = top-k = 2 (e <= 4 by emulation).
# - global_batch_size of the MoE configs (1024) is a placeholder for the job's batch size, still to choose.
CONFIG_FILES = {
    "LLAMA-3-8": "training_config_llama_3.json",
    "QWEN3-30B-A3B": "training_config_qwen3_30b_a3b.json",
    "MIXTRAL-8X7B": "training_config_mixtral_8x7b.json",
}
MODELS = {name: json.loads((CONFIGS / f).read_text()) for name, f in CONFIG_FILES.items()}

# nsys per rank, started/stopped by Megatron (--profile: cudaProfilerStart/Stop)
NSYS = ["nsys", "profile", "-t", "cuda,nvtx", "--sample=none", "--cpuctxsw=none",
        "--capture-range=cudaProfilerApi", "--capture-range-end=stop", "--force-overwrite=true"]

# Llama-style architecture of all our models, and how we run them
COMMON = [
    "--normalization", "RMSNorm", "--swiglu", "--disable-bias-linear", "--position-embedding-type", "rope",
    "--untie-embeddings-and-output-weights", "--attention-dropout", "0", "--hidden-dropout", "0",
    # as Megatron's Llama-3 recipe (examples/llama/train_llama3_8b_h100_fp8.sh); default impl is native
    "--cross-entropy-loss-fusion",
]
# MoE implementation: fp32 router, grouped GEMM, fused permute / router, NCCL all-to-all dispatcher; EP = ETP = 1
# (all experts on each GPU) unless a run sets EP
MOE = [
    "--moe-router-dtype", "fp32", "--moe-router-load-balancing-type", "aux_loss", "--moe-grouped-gemm",
    "--moe-permute-fusion", "--moe-router-fusion", "--moe-token-dispatcher-type", "alltoall",
    "--expert-model-parallel-size", "1", "--expert-tensor-parallel-size", "1",
]

# environment of every run (profiles and validation runs)
ENV = {
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",  # required by Megatron with TP + sequence parallelism
    # CUTLASS grouped GEMM: one kernel for the experts forward / dgrad instead of one cuBLAS GEMM per expert
    "NVTE_USE_CUTLASS_GROUPED_GEMM": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}


def num_experts(model):
    return MODELS[model].get("num_experts", 0)


def arch_args(model, experts=None):
    """Megatron arguments of the model's architecture, from its training config. experts: instead of num_experts
    (profile_sweep.py --num-experts)."""
    c = MODELS[model]
    args = COMMON + [
        "--hidden-size", str(c["hidden_size"]), "--ffn-hidden-size", str(c["ffn_hidden_size"]),
        "--num-attention-heads", str(c["heads"]), "--kv-channels", str(c["head_dim"]),
        "--norm-epsilon", str(c["norm_epsilon"]), "--rotary-base", str(c["rotary_base"]),
    ]
    if c.get("num_query_groups", c["heads"]) != c["heads"]:
        args += ["--group-query-attention", "--num-query-groups", str(c["num_query_groups"])]
    if c.get("qk_layernorm"):
        args.append("--qk-layernorm")
    if num_experts(model):
        args += ["--num-experts", str(experts or c["num_experts"]), "--moe-router-topk", str(c["moe_router_topk"]),
                 "--moe-ffn-hidden-size", str(c["moe_ffn_hidden_size"]),
                 "--moe-aux-loss-coeff", str(c["moe_aux_loss_coeff"])] + MOE
    return args


def model_args(model, num_layers, iters, experts=None):
    c = MODELS[model]
    seq = c["sequence_length"]
    return arch_args(model, experts) + [
        "--num-layers", str(num_layers), "--seq-length", str(seq), "--max-position-embeddings", str(seq),
        "--train-iters", str(iters),
        "--mock-data", "--tokenizer-type", "NullTokenizer", "--vocab-size", str(c["vocab_size"]),
        # bf16 gradient buffer, as the planner assumes
        "--bf16", "--grad-reduce-in-bf16",
        # no automatic Python GC (pauses on the CPU); interval 0: no collection during the run
        "--manual-gc",
        "--lr", "1e-5", "--min-lr", "1e-6", "--lr-decay-style", "constant", "--clip-grad", "1.0",
        "--eval-iters", "0", "--eval-interval", str(iters), "--log-interval", "1",
    ]
