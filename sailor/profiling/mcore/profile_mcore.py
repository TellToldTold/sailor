"""Sailor's per-layer profiler on Megatron-Core (third_party/Megatron-LM).

Runs Megatron's pretrain_gpt.py unchanged and writes <results-dir>/profile_<model>_<gpu>_<tp>_<mbs>.json in the
format of the Megatron-DeepSpeed profiler (third_party/Megatron-DeepSpeed/megatron/training.py, --sailor-profile),
which sailor/profiling/gather_profs.py turns into the planner's inputs.

    python -m torch.distributed.run --nproc-per-node <TP> profile_mcore.py \
        --sailor-model-name LLAMA-3-8 --sailor-gpu-type GH-96 --sailor-results-dir <dir> \
        --sailor-num-layers-original 32 <Megatron arguments, with --num-layers 2>

Layer keys (as in the Megatron-DeepSpeed LLAMA profiles): "0" embedding, "1".."N" transformer layers, "N+1" final
norm, "N+2" output layer + loss. "timing": [fwd, bwd, update] in seconds, where update is the optimizer step of
the whole profiled model, repeated for every key (the planner takes the max over a stage). "memory": parameters,
input/output activation sizes and the memory allocated during the layer's forward, as before. "gpu_memory"
(not read by gather_profs.py): whole-GPU bytes on rank 0 after setup and at the end (non-PyTorch overhead), and
peaks over the measured iterations, to set and check the simulator's megatron_mem.

Also not read by gather_profs.py (docs/moe_iteration_time_model.md):
- "layer_parts": {part: [fwd, bwd]} in seconds, the second transformer layer (keys "2".."N") split into parts.
  Dense: "attn" (layer input to MLP input) and "mlp" (MLP and residual add; with TE the pre-MLP norm is fused into
  the MLP's first GEMM). MoE: "attn" (with the pre-MLP norm), "router_permute", "dispatch" (A2A), "sort_in"
  (regroup by local expert), "experts" (grouped GEMMs), "sort_out", "combine" (A2A), "unpermute" (with the
  residual add). Event at each part's output in the forward, gradient hook on the same tensor in the backward.
- "optimizer": {"elements", "expert_elements"}: parameters on this GPU, all updated by its optimizer step
  (no --use-distributed-optimizer: ZeRO-0), to fit the step time as rate x elements + constant.

The model is profiled with 2 transformer layers: key "1" from the first, keys "2".."N" from the second.
Forward: CUDA events at module boundaries (forward hooks). Backward: gradient hooks on the same boundary tensors,
between the start and end of Megatron's backward_step. Averages over the iterations after --sailor-skip-iters.
Requires PP = 1 and one micro-batch per iteration (global batch = micro batch x DP).
"""

import argparse
import json
import os
import runpy
import sys
from pathlib import Path

import torch

# our copy of Megatron must come first on the path, before any megatron import
MEGATRON = Path(__file__).resolve().parents[3] / "third_party" / "Megatron-LM"
sys.path.insert(0, str(MEGATRON))

import megatron.core  # noqa: E402
import megatron.core.pipeline_parallel.schedules as schedules  # noqa: E402
import megatron.training.training as training  # noqa: E402
from megatron.core import parallel_state  # noqa: E402,F401
from megatron.core.num_microbatches_calculator import get_num_microbatches  # noqa: E402,F401
from megatron.core.utils import unwrap_model  # noqa: E402
from megatron.training import get_args  # noqa: E402


def gpu_memory_now():
    """Bytes: PyTorch allocated / reserved, device-wide used (total - free, as nvidia-smi / NVML), and the
    non-PyTorch part (used - reserved: CUDA context, NCCL buffers, library workspaces)."""
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    allocated, reserved = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
    return {"allocated": allocated, "reserved": reserved, "device_used": total - free,
            "non_torch": total - free - reserved}


def numel_bytes(tensors):
    tensors = [t for t in tensors if isinstance(t, torch.Tensor)]
    return sum(t.numel() for t in tensors), sum(t.numel() * t.element_size() for t in tensors)


def first_tensor(output):
    return output[0] if isinstance(output, (tuple, list)) else output


# parts of the second transformer layer, in forward order (see the docstring)
DENSE_PARTS = ["attn", "mlp"]
MOE_PARTS = ["attn", "router_permute", "dispatch", "sort_in", "experts", "sort_out", "combine", "unpermute"]


class LayerProfiler:
    """Segments, in forward order: emb, L1..Ln, norm, head (output layer + loss)."""

    def __init__(self, gpt_model, optimizer, skip):
        self.skip = skip
        self.gpu_mem = {"after_setup": gpu_memory_now()}  # model and optimizer built, before training
        self.layers = list(gpt_model.decoder.layers)
        self.norm = gpt_model.decoder.final_layernorm
        self.segments = ["emb"] + [f"L{i}" for i in range(1, len(self.layers) + 1)] + ["norm", "head"]

        assert len(self.layers) == 2, "profile with --num-layers 2"
        assert self.norm is not None, "final norm missing"

        modules = {"emb": gpt_model.embedding, "norm": self.norm, "head": gpt_model.output_layer}
        modules.update({f"L{i}": layer for i, layer in enumerate(self.layers, 1)})
        
        self.params = {s: numel_bytes(list(m.parameters())) for s, m in modules.items()}
        # by name: Megatron's expert tag (param.allreduce == False) is not set on a single GPU
        named = list(gpt_model.named_parameters())
        self.optimizer_elements = {"elements": sum(p.numel() for _, p in named),
                                   "expert_elements": sum(p.numel() for n, p in named if ".mlp.experts." in n)}
        self.iters = []  # per iteration: {event name: torch.cuda.Event}, contains the times of each step for each iteration
        self.mem = {}  # torch.cuda.memory_allocated() at each forward boundary for the last iteration only
        self.act = {}  # per segment: [(input numel, bytes), (output numel, bytes)], tensor sizes at each segment's input and output
        self.cur = None

        # Add forward hooks for all layers, embed and norm
        gpt_model.register_forward_pre_hook(self.on_model_start)
        gpt_model.embedding.register_forward_pre_hook(self.on_emb_input, with_kwargs=True)
        gpt_model.embedding.register_forward_hook(self.boundary("emb"))
        for i, layer in enumerate(self.layers, 1):
            layer.register_forward_hook(self.boundary(f"L{i}"))
        self.norm.register_forward_hook(self.boundary("norm"))
        gpt_model.register_forward_hook(self.on_model_end)
        self.parts = self.hook_parts(self.layers[1])

        # Add forward events for the optimizer steps
        step = optimizer.step

        def timed_step(*args, **kwargs):
            self.record("o_start")
            out = step(*args, **kwargs)
            self.record("o_end")
            return out

        optimizer.step = timed_step

    def record(self, name):
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self.cur[name] = event

    def on_model_start(self, module, args):
        if len(self.iters) == self.skip:  # first measured iteration: peaks from here on
            torch.cuda.reset_peak_memory_stats()
        self.cur = {}
        self.iters.append(self.cur)
        self.record("f_start")
        self.mem["start"] = torch.cuda.memory_allocated()

    def on_emb_input(self, module, args, kwargs):
        # GPTModel passes input_ids and position_ids as keywords
        self.act["emb"] = [numel_bytes(list(args) + list(kwargs.values())), None]

    def boundary(self, segment):
        def hook(module, args, output):
            out = first_tensor(output)
            self.record(f"f_{segment}")
            self.mem[segment] = torch.cuda.memory_allocated()
            self.act.setdefault(segment, [None, None])[1] = numel_bytes([out])
            nxt = self.segments[self.segments.index(segment) + 1]
            self.act[nxt] = [numel_bytes([out]), None]
            # add backward hook for the backward layer of this forward layer
            if out.requires_grad:
                out.register_hook(lambda grad: self.record(f"b_{segment}"))

        return hook

    def mark(self, part, out):
        """End of a part of the second layer: forward event now, backward event when its output's gradient is
        ready (the parts after it are done)."""
        self.record(f"f_{part}")
        if out.requires_grad:
            out.register_hook(lambda grad: self.record(f"b_{part}"))

    def hook_parts(self, layer):
        """Marks the ends of all parts but the last (that one ends at the layer's output, f_L2 / b_L2)."""
        mlp = layer.mlp
        mlp.register_forward_pre_hook(lambda module, args: self.mark("attn", args[0]))
        if not hasattr(mlp, "token_dispatcher"):
            return DENSE_PARTS

        def wrap(obj, method, part):
            fn = getattr(obj, method)

            def marked(*args, **kwargs):
                out = fn(*args, **kwargs)
                self.mark(part, first_tensor(out))
                return out

            setattr(obj, method, marked)

        # MoELayer.forward: route, preprocess, dispatch, routed_experts_compute (dispatcher.dispatch_postprocess,
        # experts, dispatcher.combine_preprocess), combine, postprocess
        dispatcher = mlp.token_dispatcher
        wrap(mlp, "preprocess", "router_permute")
        wrap(mlp, "dispatch", "dispatch")
        wrap(dispatcher, "dispatch_postprocess", "sort_in")
        mlp.experts.register_forward_hook(lambda module, args, out: self.mark("experts", first_tensor(out)))
        wrap(dispatcher, "combine_preprocess", "sort_out")
        wrap(mlp, "combine", "combine")
        return MOE_PARTS

    def on_model_end(self, module, args, output):
        self.record("f_end")
        self.mem["end"] = torch.cuda.memory_allocated()
        self.act["head"][1] = numel_bytes([first_tensor(output)])

    def wrap_backward_step(self, backward_step):
        def timed_backward_step(*args, **kwargs):
            self.record("b_start")
            out = backward_step(*args, **kwargs)
            self.record("b_end")
            return out

        return timed_backward_step

    def times(self, skip):
        """Mean seconds: forward and backward per segment and per part of the second layer; optimizer step;
        iteration (start to start)."""
        torch.cuda.synchronize()
        rows = [it for it in self.iters if "o_end" in it]  # completed iterations (forward, backward, step)
        used = rows[skip:]
        assert used, f"only {len(rows)} iterations, need more than --sailor-skip-iters {skip}"

        def mean_s(v):
            return round(sum(v) / len(v) / 1000, 6) if v else None

        def spans(names, f_first, f_last, b_first, b_last):
            """Consecutive spans `names`, forward between f_<name> events, backward between b_<name> events
            (the gradient of a span's output is ready when the spans after it are done)."""
            fwd_marks = [f_first] + [f"f_{s}" for s in names[:-1]] + [f_last]
            bwd_marks = [b_first] + [f"b_{s}" for s in reversed(names[:-1])] + [b_last]
            n = len(names)
            fwd = {s: mean_s([it[fwd_marks[i]].elapsed_time(it[fwd_marks[i + 1]]) for it in used])
                   for i, s in enumerate(names)}
            bwd = {s: mean_s([it[bwd_marks[n - 1 - i]].elapsed_time(it[bwd_marks[n - i]]) for it in used])
                   for i, s in enumerate(names)}
            return fwd, bwd

        fwd, bwd = spans(self.segments, "f_start", "f_end", "b_start", "b_end")
        part_fwd, part_bwd = spans(self.parts, "f_L1", "f_L2", "b_L2", "b_L1")
        parts = {p: [part_fwd[p], part_bwd[p]] for p in self.parts}
        opt = [it["o_start"].elapsed_time(it["o_end"]) for it in used]
        nxt = [rows[skip + k + 1] for k in range(len(used) - 1)]
        iteration = [it["f_start"].elapsed_time(n["f_start"]) for it, n in zip(used, nxt)]
        return fwd, bwd, parts, mean_s(opt), mean_s(iteration)

    def mem_info(self, segment):
        prev = {"emb": "start", "head": "norm"}.get(segment) or self.segments[self.segments.index(segment) - 1]
        end = "end" if segment == "head" else segment
        act_mem = self.mem[end] - self.mem[prev]
        (in_n, in_b), (out_n, out_b) = self.act[segment]
        p_n, p_b = self.params[segment]
        # "floats" are elements of the training dtype: the planner multiplies by float_size (2 for bf16)
        return {
            "params_floats": p_n, "act_output_floats": out_n, "act_input_floats": in_n, "act_mem_floats": act_mem // 2,
            "params_bytes": p_b, "act_output_bytes": out_b, "act_input_bytes": in_b, "act_mem_bytes": act_mem,
        }

    def result(self, num_layers_original, skip):
        fwd, bwd, parts, opt, iteration = self.times(skip)
        n = num_layers_original
        source = {0: "emb", 1: "L1", n + 1: "norm", n + 2: "head"}
        source.update({k: "L2" for k in range(2, n + 1)})
        timing = {str(k): [fwd[s], bwd[s], opt] for k, s in sorted(source.items())}
        memory = {str(k): self.mem_info(s) for k, s in sorted(source.items())}
        covered = sum(fwd.values()) + sum(bwd.values()) + opt
        share = f" ({100 * covered / iteration:.1f}% of the iteration)" if iteration else ""

        # whole GPU, this rank: after setup, at the end (after warm-up), and peaks over the measured iterations.
        # The peak device use is an estimate: the non-PyTorch part at the end + PyTorch's peak reserved.
        end = gpu_memory_now()
        gpu = dict(self.gpu_mem, end=end, total=torch.cuda.mem_get_info()[1],
                   peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved())
        gpu["peak_device_used_est"] = end["non_torch"] + gpu["peak_reserved"]

        def gb(b):
            return f"{b / 1e9:.2f}"

        summary = (f"[sailor-profile] fwd {fwd}\n[sailor-profile] bwd {bwd}\n[sailor-profile] update {opt} s, "
                   f"iteration {iteration} s, fwd+bwd+update {covered:.6f} s{share}\n"
                   f"[sailor-profile] GB: non-torch {gb(gpu['after_setup']['non_torch'])} after setup, "
                   f"{gb(end['non_torch'])} at end; peak allocated {gb(gpu['peak_allocated'])}, "
                   f"peak reserved {gb(gpu['peak_reserved'])}, peak device used (est.) "
                   f"{gb(gpu['peak_device_used_est'])} of {gb(gpu['total'])}\n"
                   f"[sailor-profile] layer parts [fwd, bwd] {parts}\n"
                   f"[sailor-profile] optimizer {self.optimizer_elements}")

        return {"timing": timing, "memory": memory, "gpu_memory": gpu, "layer_parts": parts,
                "optimizer": self.optimizer_elements}, summary


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--sailor-model-name", required=True)
    parser.add_argument("--sailor-gpu-type", required=True)
    parser.add_argument("--sailor-results-dir", required=True)
    parser.add_argument("--sailor-num-layers-original", type=int, required=True)
    parser.add_argument("--sailor-skip-iters", type=int, default=5)
    sailor, megatron_argv = parser.parse_known_args()

    state = {}
    setup_model_and_optimizer, train, backward_step = (training.setup_model_and_optimizer, training.train, schedules.backward_step)

    def setup_with_hooks(*args, **kwargs):
        model, optimizer, scheduler = setup_model_and_optimizer(*args, **kwargs)
        
        state["prof"] = LayerProfiler(unwrap_model(model[0]), optimizer, sailor.sailor_skip_iters)

        # Add events for full backward pass boundaries
        schedules.backward_step = state["prof"].wrap_backward_step(backward_step)
        return model, optimizer, scheduler

    def train_then_write(*args, **kwargs):
        out = train(*args, **kwargs)
        res, summary = state["prof"].result(sailor.sailor_num_layers_original, sailor.sailor_skip_iters)
        
        if torch.distributed.get_rank() == 0:
            print(summary, flush=True)

            margs = get_args()
            Path(sailor.sailor_results_dir).mkdir(parents=True, exist_ok=True)
            path = Path(sailor.sailor_results_dir) / (
                f"profile_{sailor.sailor_model_name}_{sailor.sailor_gpu_type}_"
                f"{margs.tensor_model_parallel_size}_{margs.micro_batch_size}.json")
            
            path.write_text(json.dumps(res, indent=2))
            print(f"[sailor-profile] wrote {path}", flush=True)
        
        return out

    if os.environ.get("RANK", "0") == "0":
        print(f"[sailor-profile] megatron from {Path(megatron.core.__file__).parent}", flush=True)
    
    training.setup_model_and_optimizer = setup_with_hooks
    training.train = train_then_write

    script = str(MEGATRON / "pretrain_gpt.py")
    sys.argv = [script] + megatron_argv
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
