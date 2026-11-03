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
input/output activation sizes and the memory allocated during the layer's forward, as before.

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

MEGATRON = Path(__file__).resolve().parents[3] / "third_party" / "Megatron-LM"


def numel_bytes(tensors):
    tensors = [t for t in tensors if isinstance(t, torch.Tensor)]
    return sum(t.numel() for t in tensors), sum(t.numel() * t.element_size() for t in tensors)


def first_tensor(output):
    return output[0] if isinstance(output, (tuple, list)) else output


class LayerProfiler:
    """Segments, in forward order: emb, L1..Ln, norm, head (output layer + loss)."""

    def __init__(self, gpt_model, optimizer):
        self.layers = list(gpt_model.decoder.layers)
        self.norm = gpt_model.decoder.final_layernorm
        assert len(self.layers) == 2, "profile with --num-layers 2"
        assert self.norm is not None, "final norm missing (PP > 1?)"
        n = len(self.layers)
        self.segments = ["emb"] + [f"L{i}" for i in range(1, n + 1)] + ["norm", "head"]
        modules = {"emb": gpt_model.embedding, "norm": self.norm, "head": gpt_model.output_layer}
        modules.update({f"L{i}": layer for i, layer in enumerate(self.layers, 1)})
        self.params = {s: numel_bytes(list(m.parameters())) for s, m in modules.items()}
        self.iters = []  # per iteration: {event name: torch.cuda.Event}
        self.mem = {}  # forward boundary -> torch.cuda.memory_allocated() (last iteration)
        self.act = {}  # segment -> [(input numel, bytes), (output numel, bytes)] (last iteration)
        self.cur = None

        gpt_model.register_forward_pre_hook(self.on_model_start)
        gpt_model.embedding.register_forward_pre_hook(self.on_emb_input)
        gpt_model.embedding.register_forward_hook(self.boundary("emb"))
        for i, layer in enumerate(self.layers, 1):
            layer.register_forward_hook(self.boundary(f"L{i}"))
        self.norm.register_forward_hook(self.boundary("norm"))
        gpt_model.register_forward_hook(self.on_model_end)

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
        self.cur = {}
        self.iters.append(self.cur)
        self.record("f_start")
        self.mem["start"] = torch.cuda.memory_allocated()

    def on_emb_input(self, module, args):
        self.act["emb"] = [numel_bytes(args), None]

    def boundary(self, segment):
        def hook(module, args, output):
            out = first_tensor(output)
            self.record(f"f_{segment}")
            self.mem[segment] = torch.cuda.memory_allocated()
            self.act.setdefault(segment, [None, None])[1] = numel_bytes([out])
            nxt = self.segments[self.segments.index(segment) + 1]
            self.act[nxt] = [numel_bytes([out]), None]
            if out.requires_grad:
                out.register_hook(lambda grad: self.record(f"b_{segment}"))

        return hook

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
        """Per segment: mean forward and backward seconds; mean optimizer step; mean iteration (start to start)."""
        torch.cuda.synchronize()
        segs = self.segments
        fwd_marks = ["f_start"] + [f"f_{s}" for s in segs[:-1]] + ["f_end"]
        # gradient of a segment's output is ready when the segments after it are done
        bwd_marks = ["b_start"] + [f"b_{s}" for s in reversed(segs[:-1])] + ["b_end"]
        rows = [it for it in self.iters if "o_end" in it]  # completed iterations (forward, backward, step)
        used = rows[skip:]
        assert used, f"only {len(rows)} iterations, need more than --sailor-skip-iters {skip}"

        def ms(it, a, b):
            return it[a].elapsed_time(it[b])

        fwd = {s: [ms(it, fwd_marks[i], fwd_marks[i + 1]) for it in used] for i, s in enumerate(segs)}
        bwd = {s: [ms(it, bwd_marks[len(segs) - 1 - i], bwd_marks[len(segs) - i]) for it in used]
               for i, s in enumerate(segs)}
        opt = [ms(it, "o_start", "o_end") for it in used]
        nxt = [rows[skip + k + 1] for k in range(len(used) - 1)]
        iteration = [it["f_start"].elapsed_time(n["f_start"]) for it, n in zip(used, nxt)]

        def mean_s(v):
            return round(sum(v) / len(v) / 1000, 6) if v else None

        return ({s: mean_s(v) for s, v in fwd.items()}, {s: mean_s(v) for s, v in bwd.items()}, mean_s(opt),
                mean_s(iteration))

    def mem_info(self, segment):
        prev = {"emb": "start", "head": "norm"}.get(segment) or self.segments[self.segments.index(segment) - 1]
        end = "end" if segment == "head" else segment
        act_mem = self.mem[end] - self.mem[prev]
        (in_n, in_b), (out_n, out_b) = self.act[segment]
        p_n, p_b = self.params[segment]
        return {
            "params_floats": p_n, "act_output_floats": out_n, "act_input_floats": in_n, "act_mem_floats": act_mem // 4,
            "params_bytes": p_b, "act_output_bytes": out_b, "act_input_bytes": in_b, "act_mem_bytes": act_mem,
        }

    def result(self, num_layers_original, skip):
        fwd, bwd, opt, iteration = self.times(skip)
        n = num_layers_original
        source = {0: "emb", 1: "L1", n + 1: "norm", n + 2: "head"}
        source.update({k: "L2" for k in range(2, n + 1)})
        timing = {str(k): [fwd[s], bwd[s], opt] for k, s in sorted(source.items())}
        memory = {str(k): self.mem_info(s) for k, s in sorted(source.items())}
        covered = sum(fwd.values()) + sum(bwd.values()) + opt
        share = f" ({100 * covered / iteration:.1f}% of the iteration)" if iteration else ""
        summary = (f"[sailor-profile] fwd {fwd}\n[sailor-profile] bwd {bwd}\n[sailor-profile] update {opt} s, "
                   f"iteration {iteration} s, fwd+bwd+update {covered:.6f} s{share}")
        return {"timing": timing, "memory": memory}, summary


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--sailor-model-name", required=True)
    parser.add_argument("--sailor-gpu-type", required=True)
    parser.add_argument("--sailor-results-dir", required=True)
    parser.add_argument("--sailor-num-layers-original", type=int, required=True)
    parser.add_argument("--sailor-skip-iters", type=int, default=5)
    sailor, megatron_argv = parser.parse_known_args()

    sys.path.insert(0, str(MEGATRON))
    import megatron.core
    import megatron.core.pipeline_parallel.schedules as schedules
    import megatron.training.training as training
    from megatron.core import parallel_state
    from megatron.core.num_microbatches_calculator import get_num_microbatches
    from megatron.core.utils import unwrap_model
    from megatron.training import get_args

    state = {}
    setup_model_and_optimizer, train, backward_step = (
        training.setup_model_and_optimizer, training.train, schedules.backward_step)

    def setup_with_hooks(*args, **kwargs):
        model, optimizer, scheduler = setup_model_and_optimizer(*args, **kwargs)
        assert len(model) == 1 and parallel_state.get_pipeline_model_parallel_world_size() == 1, "needs PP = 1"
        assert get_num_microbatches() == 1, "needs one micro-batch per iteration (GBS = MBS x DP)"
        assert get_args().recompute_granularity is None, "activation recomputation changes the backward"
        state["prof"] = LayerProfiler(unwrap_model(model[0]), optimizer)
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
