# Iteration time of an MoE training run: what we measure, what we calculate

Goal: one formula for the time of a training iteration of an MoE model (Qwen3-30B-A3B) on Megatron-Core. The
simulator evaluates it for a given plan. The planner searches the plans for the smallest value (or the cheapest
plan that meets a throughput target) under the memory limit. Both must use the same formula. Memory is a separate
document.

For every phase of an iteration this document gives:

- what happens, in order;
- what its duration depends on;
- whether we **measure** it (model profiler or network profiler) or **calculate** it (formula plus network model);
- why that number stays correct when the other settings of the plan change.

The last point is the important one. Sailor profiles a few small configurations on one node and predicts large
multi-node plans. That only works if every measured quantity depends only on things the profile reproduces
exactly, and everything else is calculated.

## 0. Assumptions and symbols

**Assumptions** (the v0 model):

- **No overlap.** All work runs one after the other on one CUDA stream: no communication hides behind computation
  (no `--overlap-grad-reduce`, `--overlap-param-gather`, `--tp-comm-overlap`, or A2A overlap). The iteration time is
  then the sum of the phase times on the slowest path. This is how our profiles already run.
- **ETP = 1.** Each expert sits whole on one GPU. Its matrices are never split.
- **TP within one node.** TP, EP, DP and PP can be anything else. EP may span nodes.
- **Balanced routing.** Every expert receives the same number of tokens. Real routing is imbalanced, and the
  slowest GPU sets the pace. This is a known optimistic bias; we add a correction later if measurements show it
  matters.
- **Homogeneous plan**: one GPU type (GH200) and the same configuration on every stage. Sailor itself supports
  heterogeneous plans; we can extend once the homogeneous case works.
- **Optimizer not sharded (ZeRO-0), as original Sailor.** Every GPU keeps the full optimizer state for the
  parameters it holds, and gradients are synchronised with one all-reduce per group. Megatron-Core does this when
  `--use-distributed-optimizer` is not set. Chosen for v0 because it matches original Sailor's structure: the
  update is local and independent of d, and the sync is one calculated term.
  - Cost: memory. 2 + 2 + 12 = 16 bytes per parameter on every DP copy, instead of 2 + 2 + 12/d.
  - Moving to the distributed optimizer later changes two things: the step's element count is divided by d (or EDP),
    and the all-reduce splits into a reduce-scatter before the step and an all-gather after it.
- **Sequence length s is fixed by the training job**, as in original Sailor (`training_config["sequence_length"]`):
  the planner never chooses it. Every measured number is per (t, b) *at that s*, and profiles are only valid for
  jobs with the same s. Attention grows with s², the rest with s, so a profile can't be rescaled to another s.
  Today nothing checks this; a cheap safeguard is to record s in the profile and compare it with the training
  config.
- 1F1B pipeline schedule without interleaving; bf16 with `--grad-reduce-in-bf16` (2-byte gradients); sequence
  parallelism on when TP > 1; no activation recomputation. These match our profiler settings
  (`sailor/mcore/models.py`, `profiler/profile_sweep.py`).

**Symbols**

| Symbol | Meaning | Qwen3-30B-A3B |
|---|---|---|
| b | micro-batch size (MBS) | |
| s | sequence length | 4096 |
| h | hidden size | 2048 |
| E | experts per layer | 128 |
| k | experts per token (top-k) | 8 |
| f | expert FFN hidden size | 768 |
| L | transformer layers | 48 |
| t, p, d, e | TP, PP, DP, EP | |
| EDP | expert data parallelism = t·d / e | |
| m | micro-batches per iteration = GBS / (b·d) | |
| T_loc | tokens one GPU holds between layers = b·s / t (sequence parallelism splits the sequence over TP) | |

**Two grids over the same GPUs.** Within one pipeline stage there are t·d GPUs. The dense parts (attention, norms,
router, embedding, head) are laid out as TP × DP. The experts are laid out as EP × EDP. ETP = 1 means each GPU holds
E/e whole experts, and the same experts are held by EDP = t·d/e GPUs. Constraints: e divides t·d, and e divides E.

The case we profiled with TP 4 and EP 1 is EDP = 4: every GPU holds all 128 experts.

## 1. The shape of one iteration

In order:

1. Zero the gradient buffers. Negligible, so ignored (section 2.4).
2. **Pipeline phase**: m micro-batches go forward and backward through the p stages (1F1B). Inside it, each
   transformer layer runs attention and then the MoE block. The MoE block contains the token all-to-alls, A2A for
   short.
3. **Gradient synchronisation**: all-reduce the dense gradients over DP and the expert gradients over EDP, plus a
   small TP all-reduce for replicated parameters.
4. **Optimizer step** on all the parameters this GPU holds: gradient norm and clipping, Adam, copy into the bf16
   weights.

```
T_iter = T_pipeline + T_grad_sync + T_step
```

This has the same structure as today's Sailor simulator, `T_iter = T_pp + T_sync` (`runtime_simulator.py:401-462`):

- `T_pp` is the pipeline phase, plus the profiled `update` (the max over stages).
- `T_sync` is a bucketed DP all-reduce.

What changes for MoE:
- a second all-reduce group (EDP) for the experts;
- the step scaled by the stage's size (section 4);
- the three fixes to `T_sync` in section 3.

## 2. The pipeline phase

### 2.1 How the stages combine

Let C_j be the time stage j needs for **one** micro-batch, forward plus backward. Let Q_j be the time to send that
micro-batch's activation forward and its gradient back across the boundary after stage j (point-to-point, P2P).

```
T_pipeline ≈ Σ_j (C_j + Q_j)  +  (m − 1) · max_j (C_j + Q_j)
```

This is Sailor's formula (`get_time_per_pipeline`, `runtime_simulator.py:339-399`).

Why: the first micro-batch has to go through every stage and come back. That's the fill and drain of the pipeline,
the Σ term. The other m − 1 micro-batches leave the pipeline at the rate of the slowest stage, which can't work
faster than one micro-batch per C_j + Q_j. For equal stages this gives the textbook (m + p − 1) · (C + Q).

Our stages are usually **not** equal, even in a homogeneous plan:
- the first stage also holds the embedding, and the last the final norm, output layer and loss (for LLAMA the
  output layer alone is 2–3 layers of compute);
- the layer count may not divide evenly over p;
- the last stage has no P2P send (Q = 0).

So we keep the general form. It becomes the textbook one when the planner balances the stages well (for example
fewer layers on the first and last stage). The max term is what penalises a bad split, which is why the planner
needs it to compare splits.

It doesn't change for MoE. The A2As happen *inside* a stage's layers, so they're part of C_j, not of Q_j. All
GPUs of one EP group are in the same pipeline stage, possibly in different DP replicas. All replicas run the same
schedule in lockstep, so the A2A doesn't add waiting beyond its own duration. That holds as long as the plan is
homogeneous.

### 2.2 Inside one stage: C_j

```
C_j = [ embedding, if j is the first stage ]
    + Σ over the transformer layers of stage j of (forward + backward of the layer)
    + [ final norm + output layer + loss, if j is the last stage ]
                                                        all for one micro-batch
```

**First stage, before the first layer, forward:**

| # | Phase | What happens | Depends on | How |
|---|---|---|---|---|
| E1 | Embedding | Look up the b·s token ids in this GPU's vocabulary/t slice of the table (tokens outside the slice give zeros), then reduce-scatter over TP. The sum completes every token's vector, and each GPU keeps its s/t tokens (sequence parallelism). Position information is added later by RoPE inside attention, so there's no position table. | t, b, s, h, vocabulary | **measure** (profiler key 0, per TP and MBS) |

Backward: all-gather the output gradient over TP, then scatter-add it into this GPU's rows of the table (weight
gradient). The embedding has no input gradient, because its input is token ids.

**One transformer layer, forward, in order:**

| # | Phase | What happens | Depends on | How |
|---|---|---|---|---|
| F1 | Attention block | RMSNorm, all-gather of the sequence over TP, QKV GEMM, flash attention, output projection, reduce-scatter over TP, residual add, then the pre-MLP RMSNorm (the `layer.mlp` event comes after it) | t, b, s, model shape | **measure** (profiler, per TP and MBS) |
| F2 | Router | Router GEMM (h × E) on T_loc tokens, softmax and top-k, aux-loss terms | T_loc, h, E | **measure** (profiler) |
| F3 | Permute | Copy each token k times and sort the copies by expert: T_loc·k rows of h | T_loc, k, h | **measure** (profiler) |
| F4 | Dispatch A2A | Send each copy to the GPU that holds its expert, over the EP group | bytes, e, nodes spanned | **calculate** (A2A bandwidth curves) |
| S1 | Sort in | Regroup the received rows from sender order to local-expert order, so each expert's rows are contiguous for the grouped GEMM (`dispatch_postprocess`, permutation 2). A copy of T_loc·k rows of h. At EP = 1 it also holds the GPU-to-CPU synchronisation of the per-expert counts. | T_loc, k, h | **measure** (profiler `sort_in`, base profile) |
| F5 | Expert MLPs | Grouped GEMM over the E/e local experts, with T_loc·k·e/E tokens each (SwiGLU: two GEMMs, activation, one GEMM) | E/e and tokens per expert | **measure** (profiler with `--num-experts E/e`) |
| S2 | Sort out | The inverse of S1: put the expert outputs back in sender order for the combine A2A (`combine_preprocess`) | T_loc, k, h | **measure** (profiler `sort_out`, base profile) |
| F6 | Combine A2A | Send the outputs back to their tokens' GPUs (the reverse of F4) | same as F4 | **calculate** |
| F7 | Unpermute | Undo the sort, weight by the router probabilities, sum the k copies, residual add | T_loc, k, h | **measure** (with F3) |

**Last stage, after the last layer, forward:**

| # | Phase | What happens | Depends on | How |
|---|---|---|---|---|
| H1 | Final norm | RMSNorm on this GPU's s/t tokens (local) | t, b, s, h | **measure** (profiler key N+1) |
| H2 | Output layer | All-gather the sequence over TP, then a GEMM against this GPU's vocabulary/t columns: logits for all b·s tokens over V/t words | t, b, s, h, vocabulary | **measure** (profiler key N+2, with H3) |
| H3 | Loss | Cross-entropy over the split vocabulary (fused): all-reduces over TP of the per-token maximum, of the sum of exponentials, and of the target word's logit (b·s values each, so small), then the mean | t, b, s, vocabulary | **measure** (with H2) |

Backward, from the loss: the gradient of the logits (the fused loss computes it during the forward), the output
layer's input and weight gradients (two GEMMs), and a reduce-scatter of the input gradient over TP back to s/t
tokens, then the final norm's backward.

The output layer is the largest single piece of a stage. For LLAMA at TP 1, MBS 1 it takes 21 / 34 ms forward /
backward against 7 / 17 ms for a transformer layer, and holds 525M parameters, about 2.4 layers' worth. This is why
the planner usually gives the last stage fewer layers.

The loss is averaged over DP once per iteration, for logging only: a scalar all-reduce, latency only.

**Backward** is the same list in reverse:
- F7 → F1, each with its own backward kernels.
- The two A2As run again, on the gradients: the backward of combine is a dispatch-shaped A2A, and the backward
  of dispatch is a combine-shaped one.
- The expert GEMMs do about twice the forward work: input gradient plus weight gradient.
- The attention block's TP all-gather and reduce-scatter swap roles.

So one layer has **4 A2As per micro-batch** (2 forward, 2 backward).

With ETP = 1 the MoE block has **no TP communication**. Each GPU runs its own T_loc tokens through whole experts,
and the output already has the full hidden size h for those tokens. That's what keeps F2–F7 independent of t
except through T_loc.

Confirmed in Megatron's code: the dispatcher's "TP" group is the **expert** TP group
(`megatron/core/transformer/moe/token_dispatcher.py:89`, `self.tp_group = pg_collection.expt_tp`). With ETP = 1 it
has size 1, so the TP all-gather in `dispatch_postprocess` (`:743`) never runs.

Below, each row is explained.

#### F1 Attention block, E1 embedding, H1–H3 final norm, output layer and loss: measured

E1 and H1–H3 are measured and carry over for the same reasons as F1: only this GPU and its TP partners take part,
on shapes fixed by (t, b, s) and the model. The profiler already records them as separate keys (0, N+1, N+2).

- **What we measure:** with the profiler, the time from the layer's input to the MoE block's input, forward and
  backward, for each (t, b): the `attn` part of the profile's `layer_parts` (event at the input of `layer.mlp`).
- **Why the number carries over:**
  - Everything in F1 involves only this GPU and its t − 1 TP partners: the same kernels on the same tensor shapes
    (b, s, h/t per GPU), and the same TP collectives over the same links (NVLink, because TP is within a node).
  - Nothing in F1 depends on d, p or e.
  - A later micro-batch runs exactly the same kernels as the first. Weight gradients accumulate into the gradient
    buffer in place, with no extra pass.
  - So the profile at (t, b) gives the time of F1 in any plan with the same (t, b).
- This is exactly the assumption Sailor already makes for dense models.

#### F2, F3, S1, S2, F7 Router, permute, sorts, unpermute: measured

- **What we measure:** with the profiler at EP = 1, the `layer_parts` `router_permute` (F2 + F3), `sort_in`,
  `sort_out` and `unpermute` (F7), forward and backward.
- **The local work is the same at EP = 1 as at EP > 1.** From Megatron's A2A dispatcher
  (`megatron/core/transformer/moe/token_dispatcher.py`):

  | Step | Work | Changes with e? |
  |---|---|---|
  | Router | GEMM T_loc × h × E, softmax, top-k over all E experts | no |
  | Permute 1 (`dispatch_preprocess`) | copy and sort T_loc·k rows by expert; dropless, so the size is fixed at T_loc·k (`:556`) | no |
  | Permute 2 (`sort_chunks_by_idxs`, `:785` after dispatch, `:820` before combine): our `sort_in`, `sort_out` | regroup the received rows by local expert. Runs whenever there's more than one local expert, so at EP = 1 too. Rows T_loc·k; chunks e × E/e = E, constant. | no |
  | Unpermute | inverse of permute 1, weighted sum of the k copies | no |
  | Count exchange (`preprocess`, `:560-599`) | all-gather of the per-expert token counts over the EP group, then a GPU-to-CPU synchronisation for the A2A's split sizes | **only at EP > 1**; at EP = 1 there's still one GPU-to-CPU synchronisation (`:605`), for the grouped GEMM's counts |

  So the only extra local cost at EP > 1 is the count all-gather: a few bytes, latency only. It goes into the
  A2A term (F4/F6), not into the measured parts.
- **Why the number carries over:** these are local operations on this GPU's T_loc tokens, sized T_loc × h,
  T_loc × E and T_loc·k × h. Under balanced routing none of them depends on e; only where the rows go changes, not
  the cost of sorting and copying them. Measured at (t, b), they hold for any e, d, p.
- **Caveat:** the GPU-to-CPU synchronisation stops the CPU from queueing kernels ahead, so GPU idle time can follow
  it. That's the same at every e, so it's inside the measurement, but it's a reason our no-overlap model may need
  a constant per layer.

#### F4, F6 Dispatch and combine A2A: calculated

**Bytes.** Each GPU sends T_loc·k rows of h bf16 values per token A2A:

```
V = T_loc · k · h · 2 B  =  b · s · k · h · 2 / t
```

For Qwen3: 134 MB at b = 1, t = 1; from 33.5 MB (b = 1, t = 4) to 1.07 GB (b = 8, t = 1).

**Where the bytes go.** Under balanced routing each GPU sends V/e to each of the e GPUs of its EP group. With
Megatron's default rank order the EP group takes consecutive ranks, so it fills a node before it uses the next one
(to confirm in Megatron's process-group code). With g = min(e, 4) GPUs of the group on each node and
n_nodes = e / g nodes, each GPU sends:

- 1 part to itself: a local copy at memory speed, negligible;
- g − 1 parts over **NVLink**: `V_nvlink = V · (g − 1) / e`;
- e − g parts over the **network** (one NIC per GH200): `V_net = V · (e − g) / e`.

**Time.** Both links work at the same time and, with balanced routing and no overlap, every GPU waits for the
slowest:

```
T_a2a = max( V_nvlink / BW_nvlink(V_nvlink; g) ,  V_net / BW_net(V_net; n_nodes) )
```

The bandwidths aren't constants:
- **They depend on message size.** Small messages don't reach peak (latency, NCCL's protocol choice, chunking), and
  in an A2A each per-peer message is only V/e. At b = 1, t = 4, e = 16 that's about 2 MB.
- **BW_nvlink depends on g.** Sending to 1 peer (e = 2) uses only the links to that GPU; sending to 3 peers (e ≥ 4)
  uses the links to all three at once.
- **BW_net depends on the number of nodes.** With more nodes, more of V leaves the node, and each NIC carries more
  and smaller flows. Receivers get traffic from more senders at once (incast), the job is more likely to span
  Slingshot switch groups, and NCCL's all-to-all becomes less efficient with more peers. The same bytes per GPU
  take longer over 4 nodes than over 2.

**Per layer and micro-batch:**

```
T_ep,layer = 4 · T_a2a(V)           tokens: dispatch and combine, forward and backward
           + 2 · T_a2a(V_probs)     router probabilities: a second A2A in dispatch, forward and backward;
                                    V_probs = T_loc · k · 4 B (131 KB for Qwen3 at b = 1, t = 1), so latency only
           + T_allgather(E · 8 B, e)  per-expert token counts, forward only; latency only
```

At e = 1 all three are zero.

**What we measure.** `torch.distributed.all_to_all_single` on a bf16 tensor over the EP group, the call Megatron
makes (`token_dispatcher.py:708`, `:867`):
- each GPU sends V split into e equal parts (balanced routing);
- V is swept from about 1 MB to 1 GB; it's an input we choose, not something we measure;
- e = 2 and 4 (1 node) give BW_nvlink for g = 2 and g = 4; e = 8 and 16 (2 and 4 nodes) give BW_net for
  n_nodes = 2 and 4. There the network part dominates (about 6× slower than NVLink), so BW_net = V_net / T. Any
  slowdown from the NVLink traffic running alongside ends up in that fitted bandwidth, which is fine: it's measured
  in the layout plans use;
- the time per call is from CUDA events after warm-up: the median over iterations, of the slowest rank;
- **whole nodes, all EP groups at once.** In a real job every EP group of a stage does its A2A at the same moment.
  For e ≥ 4 each node belongs to one group, so the network profiling has to load the nodes the same way, or it misses NIC
  contention.

Effective bandwidth = bytes / time, the "algbw" of nccl-tests. Each curve is fitted as **time = α + bytes / β**
(latency plus bytes over bandwidth; `sailor/profiling/networking/fit_alltoall.py`), by least squares on the relative
time error. Decided 2026-11-10 instead of original Sailor's form (bandwidth quadratic in log2(message size),
`fit_func.py`): on the measured A2A (job 3901891) the quadratic misses the small router-probabilities A2A by
53–106% (median), while α + bytes / β stays within 10% from 32 KB to 1 GB. The two numbers are stored per
configuration, the same lookup pattern as Sailor's `get_zone_coeffs`, with our configuration as the key:

```
coefficients[nvlink][g]        g = 2, 4          {alpha_s, beta_GBps}
coefficients[net][n_nodes]     n_nodes = 2, 4
```

α is measured two ways (`test_alltoall.py`): with the host launch inside the timing window and GPU only. Which one
training sees depends on whether the CPU runs ahead of the GPU at that call; the training traces decide.

That's four curves. Before using them, check that they make physical sense: NVLink much faster than the network,
bandwidth rising with size.

**Compared with original Sailor.** Sailor uses the same building block (size-dependent bandwidth curves per link
and per number of GPUs per node taking part), but it measures only point-to-point sends and *derives* collectives
from them: its all-reduce is the ring model `2·(n − 1) · t_p2p(size/n)` (`profile_utils.py:estimate_ar_time`). We
measure the A2A itself. A point-to-point curve measures one pair alone, while an A2A has many simultaneous flows per
GPU and incast at every receiver, which is exactly what makes it expensive.

**Limit: only measured EP sizes.** The planner may only choose e for which data exist (A2A curves and an expert
profile), and must reject other e rather than extrapolate. Nothing in the code caps e; the limits come from the
model and from what we chose to measure:
- e must divide E (for Qwen3: 1, 2, 4, 8, 16, 32, …), so layouts like e = 12 on 3 nodes don't occur;
- **our emulation, not the model:** the real model runs at any e that divides E, down to one expert per GPU
  (Megatron only requires top-k ≤ the *total* E). The `--num-experts E/e` profiles run at EP = 1 with E/e *total*
  experts, so they need E/e ≥ k (Qwen3: e ≤ 16; Mixtral: e ≤ 4). For smaller E/e, profile at the real EP = e
  (multi-node if needed): `layer_parts` still separates `experts` from `dispatch` / `combine`;
- **measurement:** the A2A curves exist only for the node counts we ran. So far we stay within 4 nodes (e ≤ 16) to
  save compute; larger e needs runs on more nodes, not code changes.

**Biases:** imbalanced routing makes the busiest GPU send and receive more than V. EP rank layouts other than
consecutive would change g and n_nodes.

#### F5 Expert MLPs: measured, with fewer experts

- **What runs:** the grouped GEMM over E/e experts, each with n = T_loc·k·e/E tokens under balanced routing.
- **Key observation:** the total work is `T_loc · k · 3 · 2 · h · f` FLOPs forward, the same for every e. Only the
  GEMM shapes change: fewer experts, more tokens each, as e grows. Small GEMMs run far below peak.
  - We already see this: at MBS 1 the Qwen3 layer time didn't fall with TP (13.0 → 12.4 → 13.3 ms) although T_loc
    did.
  - So time isn't proportional to FLOPs and must be measured per shape.
- **What we measure:** the profiler's `experts` part (`layer_parts` in the profile), for the same model run at EP = 1
  with `--num-experts E/e` (`profile_sweep.py --num-experts`).
  - Each GPU then holds E/e experts and routes its T_loc·k token copies over them: T_loc·k·e/E tokens each. That's
    exactly the grouped GEMM of the real model at EP = e.
  - The TP × MBS sweep varies T_loc. One sweep per E/e covers the tokens-per-expert range: for Qwen3, E/e = 128, 64,
    32, 16 and 8. This emulation can't go below E/e = k (Qwen3: 8, so e ≤ 16); the real model can. See the limit
    below.
  - Only the `experts` part of these runs is used. The rest of the model is the same as in the normal profile.
- **Why the number carries over:**
  - The grouped GEMM's inputs are completely determined by (E/e, n, h, f). Where the tokens came from (which GPU,
    which micro-batch, which DP replica) doesn't affect the kernels.
  - So one set of sweeps per GPU type covers every plan.
- **Why not profile at EP = e instead:** a profile at EP = e also times the A2A, and its EP group is limited to one
  node. With fewer experts at EP = 1 there's no A2A in the measurement, and the GEMM sees the same shapes, using
  Megatron's real expert module with no separate benchmark code.

### 2.3 P2P between stages: Q_j, calculated

- **What:** after stage j's forward, each GPU sends its part of the activation (b·s/t × h with sequence
  parallelism) to the matching GPU of stage j + 1. In the backward pass the gradient of the same size goes back.
- **What crosses the boundary:** inside a stage, between layers, each of the t TP GPUs holds a different s/t slice of
  the tokens (full h each), so the activation is complete only across the t GPUs together. Megatron pairs TP rank i
  of stage j with TP rank i of stage j + 1, and each pair sends its own slice. The receiver needs exactly that slice
  (its first layer starts with RMSNorm on s/t tokens, then the all-gather before attention), so nothing is gathered
  at the boundary. In total s·h crosses the boundary, exactly once. Without sequence parallelism every TP GPU would
  hold the same full copy, which is what original Sailor assumed.
- **Formula:** the size is fixed by the shapes, so it's calculated, not profiled:

  ```
  bytes_j = b · s · h / t · 2 B                       per GPU, per micro-batch (bf16)
  Q_j     = T_send(bytes_j, link j → j+1)           forward: the activation
          + T_send(bytes_j, link j+1 → j)           backward: its gradient, same size
  Q_last  = 0
  ```

  The t GPUs of the stage send in parallel; on Alps each GH200 has its own NIC. `T_send` is Sailor's point-to-point
  model (`estimate_send_time`, with the coefficients of the link). Example, LLAMA at b = 1, t = 4:
  8192 · 4096 / 4 = 8.4M elements, 16.8 MB per GPU per direction.
- **Why it carries over:** the size depends only on (b, s, h, t), and a point-to-point send only on its size and the
  link. It's unchanged for MoE: at the boundary the tokens are back on their own GPUs (combine has returned them),
  so no expert-routed data crosses it.
- **Fix in Sailor:** `runtime_simulator.py:161-175` takes `act_output_floats` from the **TP 1** entry when one exists
  (the full s·h), whatever the plan's t. With sequence parallelism that overestimates the bytes by t. Use the
  formula above (or the plan's own t entry). The profiled `act_output_floats` stays useful as a cross-check: it
  equals s·h/t, and a mismatch would reveal a misunderstood layout.

### 2.4 Zeroing the gradient buffers: T_zero

- A memset of this GPU's gradient buffer at the start of the iteration. It's proportional to the bytes of
  gradients held (dense / t, experts / e, × 2 bytes) and bandwidth-bound.
- Small: about 2 bytes per element held, against roughly 20 for the optimizer step. A Qwen3 stage of 12 layers at
  EP 8 holds about 1.1B parameters, 2.3 GB, so about 0.6 ms at roughly 4 TB/s, under 0.1% of an iteration of
  seconds.
- **Ignored** in the formula. If it ever matters, it's proportional to n like the step, so it can be absorbed into
  r (section 4). It falls outside the profiler's events; it's part of the few % the profile doesn't cover.

## 3. Gradient synchronisation: T_grad_sync, calculated

After the last micro-batch's backward, each parameter's gradient has to be summed over every GPU that holds a copy
of that parameter. With ZeRO-0, an all-reduce does this, and every GPU ends up with the full summed gradient of
everything it holds.

| Group | Parameters | Bytes per GPU | Group size | Group layout |
|---|---|---|---|---|
| DP | dense: attention, plus the embedding (first stage) or output layer (last stage), /t; norms and router at full size (replicated over TP) | (P_dense,j / t + P_repl,j) × 2 B | d | spread by t; usually spans nodes |
| EDP | experts of this stage held by this GPU | P_exp,j / e × 2 B | EDP = t·d/e | spread by e |
| TP | parameters replicated over TP and fed by sequence-split inputs: norms, router, qk-norm. Summed over TP first, then also over DP in the DP row. | P_repl,j × 2 B, tiny | t | within a node |

- The **TP** row exists because, with sequence parallelism, each TP partner computes the gradient of a replicated
  weight from only its part of the sequence, so the partial gradients have to be summed. Megatron does this when
  it finalizes gradients. It's small, but it falls outside our profiler events, like the expert all-reduce.
- `2 B` because `--grad-reduce-in-bf16`.
- Small computation inside this phase:
  - The additions of the all-reduce run inside NCCL's kernels. A collective benchmark in bf16 includes them.
  - Before the all-reduce, Megatron scales the gradients by 1/d (1/EDP for experts) to get a mean. That's one
    elementwise pass over the gradient buffer, under 1 ms, so it's ignored like T_zero.

```
T_grad_sync = max over stages j of [ AR(dense bytes_j, d, layout) + AR(expert bytes_j, EDP, layout) + AR_small(t) ]
```

- Within a stage the terms are summed: no overlap.
- Across stages it's the max: each stage's groups are disjoint, so the stages synchronise in parallel.

**Why calculate:**
- Group sizes and layouts depend on d and e, which a one-node profile at DP = 1 can't reproduce.
- An all-reduce's duration depends only on its bytes, group size and links.
- Original Sailor already models DP this way (`estimate_sync_time`, `runtime_simulator.py:464`): a bucketed
  all-reduce of the stage's parameters / TP over the stage's DP, using the network coefficients of the GPUs
  involved, max over stages.

**What changes compared to Sailor today:**

1. **Two groups instead of one.** Today the whole stage's parameters are all-reduced over DP. For Qwen3, 97% of a
   layer's parameters are experts, which sync over EDP. That's a different group size and layout.
2. **bf16 gradients** (2 B).
3. **Three fixes to the existing `estimate_sync_time`:**
   - Use the **intra-node** network coefficients when the group fits in one node. Today it always uses the
     inter-node coefficients with `gpu_count = 1`.
   - Remove the cap of 4 on `tp_min`.
   - Use **Megatron's gradient bucketing**, not DeepSpeed's `MEMORY_BUCKET_DEEPSPEED_SIZE`. Without
     `--overlap-grad-reduce` Megatron doesn't bucket at all (`distributed_data_parallel.py:135-137` sets
     `bucket_size = None`): one all-reduce per gradient buffer, dense and experts.

## 4. Optimizer step: T_step, measured as a rate, then scaled

**What runs** on each GPU, on every parameter it holds (nothing is sharded with ZeRO-0):

1. The gradient norm, then scalar all-reduces to combine it over all GPUs (latency only). Megatron keeps dense and
   expert parameters in two optimizers, whose norms are reduced over different groups, and also checks for
   NaN or Inf values, so it's a few scalar all-reduces.
2. Clipping: scale the gradients.
3. Convert the bf16 gradients to fp32; Adam on the fp32 master weights, m and v.
4. Copy the updated master weights into the bf16 weights.

**Elements this GPU updates** in stage j:

```
n_j = P_dense,j / t + P_repl,j + P_exp,j / e
```

That's the dense part split by TP, the TP-replicated part in full, and this GPU's E/e experts. It doesn't depend
on d, as in original Sailor.

**Formula:**

```
T_step,j = r · n_j + c
```

- r is the time per element, for one GPU type.
- c is the fixed cost: kernel launches, CPU work, and the scalar all-reduce latencies.

**How:**
- Fit r and c with a straight line through the profiled `update` times against the profiled n. The profiles at
  TP 1/2/4 and the two models give different n.
- The profiler must record n: the elements it actually updated.

**Why it carries over:** every operation is an elementwise pass over the held parameters. Time = bytes read and
written / memory bandwidth + fixed launch cost, so it's linear in n, wherever the parameters came from.

**Cautions:**
- First ZeRO-0 tests (jobs 3895459, 3895460, 3895513, 3896090; MBS 1): 19–22 ps per element across Qwen3 TP 1 and
  2, Qwen3 with 16 experts and LLAMA TP 2, so the step is linear in n. With ZeRO-0 the odd TP 1 → TP 2 result is
  gone (38.9 → 33.3 ms, in line with 1.87 → 1.54B elements).
- The Qwen3 profiles before that (job 3884919) ran **with** the distributed optimizer. Their `update` is from a sharded
  step, with the experts sharded over EDP = t. That probably includes the parameter all-gather, and explains the
  odd TP 1 → 2 update times (46 → 44 ms). They have to be re-profiled with ZeRO-0 before fitting r and c. The same
  applies to the LLAMA profiles (job 3884918); there the sharding has no effect at DP = 1, but the code path
  differs.
- Today's planner uses `stage.size() × update` and the simulator uses the max over the stage, where `update` is the
  step time of the whole profiled model (2 layers + embedding + head) copied onto every layer key. The simulator
  therefore underestimates the step of a stage with many layers, and the planner overestimates it. Both are
  replaced by `r · n_j + c`.

## 5. Full formula and who uses it

```
T_iter = Σ_j (C_j + Q_j) + (m − 1) · max_j (C_j + Q_j)
       + max_j [ AR_DP,j + AR_EDP,j + AR_TP,j ]
       + max_j [ r · n_j + c ]

C_j = Σ over the layers of stage j of
        [ F1(t,b) + F2,3,S1,S2,7(t,b) + F5(E/e, T_loc·k·e/E) + T_ep,layer(V, e) ]                  (forward + backward)
      + E1 on the first stage, H1–H3 on the last stage                                         (forward + backward)
```

- **Simulator:** given a plan (p, t, d, e, b, layers per stage, which nodes), evaluate this. It's a direct update of
  `runtime_simulator.py`, whose `T_pp + T_sync` structure it keeps.
- **Planner:** search over plans for the minimum T_iter (or Sailor's cost objective) subject to the memory limit
  (16 bytes per parameter held, plus activations and the fixed overhead). It uses the same formula in C++.
  - New search dimension: **e**, with the constraints e | t·d, e | E, and e among the EP sizes with measured A2A
    curves and expert profiles (Qwen3 today: up to 16).
  - e changes four things: F5 (shapes), the A2A (bytes stay the same, but group size and layout change), the EDP
    all-reduce, and the step's expert elements (P_exp / e). It doesn't change F1–F3, F7 or P2P.
  - That makes the search cheap: for each (t, b), the dense part is a single lookup, and only the expert terms are
    re-evaluated per e.

## 6. Where each number comes from: summary

| Term | Measured / calculated | Source | Depends only on |
|---|---|---|---|
| F1 attention, E1 embedding, H1–H3 final norm / output layer / loss | measured | profiler, 1 node, per (t, b) | t, b, model |
| F2/F3/S1/S2/F7 router, permute, sorts, unpermute | measured | profiler at EP = 1, hooks on router/dispatcher | t, b, model |
| F5 expert MLPs | measured | profiler `experts` part with `--num-experts E/e`, at EP = 1 | E/e, n, h, f |
| F4/F6 A2A ×4, probabilities ×2, count all-gather | calculated | bandwidth curves: NVLink per g, network per n_nodes | bytes, g, n_nodes |
| Q P2P between stages | calculated | b·s·h/t·2 B with Sailor's point-to-point model | b, s, h, t, link |
| T_zero, gradient scaling | ignored | negligible | gradient bytes |
| Gradient all-reduce | calculated | collective model | bytes, d / EDP, layout |
| Optimizer step | measured r, c × elements | line fit through profiled `update` against n | n |

## 7. Profiler: what it does now, what it needs

**Now** (`sailor/mcore/`: `models.py` shared, `profiler/`, `validation/`; models in `Planner/simulations/configs/`):
- `profile_mcore.py` writes per layer forward / backward / `update` times and the memory fields Sailor reads, for
  one (t, b) per run.
- It also writes a `gpu_memory` block: non-PyTorch overhead after setup and at the end, and peak allocated /
  reserved memory. We keep collecting this to set `megatron_mem` later. `gather_profs.py` doesn't read it.
- `profile_sweep.py` runs TP 1/2/4 × MBS 1/2/4/8 on one node with the settings above (ZeRO-0 since this decision,
  bf16 gradients, sequence parallelism, cross-entropy fusion).
- `profile_sweep.py --nsys` records Nsight Systems traces per rank in moe-comms' `moetrace` format, in
  `<job>/traces/`. This is the tool for the trace checks below.
- **Layer parts.** The profile's `layer_parts` splits the second transformer layer into parts, forward and backward:
  - MoE: `attn` (F1), `router_permute` (F2 + F3), `dispatch` (F4), `sort_in`, `experts` (F5), `sort_out`,
    `combine` (F6), `unpermute` (F7). `sort_in` and `sort_out` are the dispatcher's regrouping by local expert,
    local work like F3 and F7.
  - Dense: `attn` and `mlp`.
  - At EP = 1, `dispatch` and `combine` should be about zero, which is a sanity check.
- **Optimizer elements.** The profile's `optimizer` block gives n (`elements`, with `expert_elements`), to fit
  `T_step = r · n + c`.
- **Expert shapes.** `profile_sweep.py --num-experts K` runs the same model with K experts (F5 above).

**Needed for this model**, in order:

1. **Re-profile with ZeRO-0** (1 node, about 10 minutes per sweep): both models, plus one Qwen3 sweep for each
   `--num-experts` of 64, 32, 16 and 8.
2. **A2A bandwidth curves (F4/F6)**: the network profiling in section 2.2 (F4/F6), `sailor/profiling/networking/test_alltoall.py`, at e = 2, 4 (1 node), 8 and 16 (2 and 4
   nodes, whole nodes). Check first whether moe-comms' NCCL A2A measurements already cover these points with
   full-node loading.
3. **Simulator**: implement the formula, the three `estimate_sync_time` fixes and the P2P fix (section 2.3);
   validate against measured
   end-to-end iteration times of real runs (1–4 nodes, several (t, d, e)).
4. **Planner**: the same formula in C++, plus the search over e and the memory model.
5. **Things to confirm**:
   - the EP rank layout (consecutive ranks, filling a node first), in Megatron's process-group code;
   - in an nsys trace, the small backward times of some parts (`unpermute` 0.05 ms against 0.49 ms forward,
     `combine` 2 µs at EP = 1), before relying on them.

**Known biases of this model**, to keep in mind when validating:
- balanced routing (optimistic);
- no overlap (pessimistic, compared with a tuned run that overlaps);
- CPU launch overhead not modelled (we assume the GPU is the bottleneck);
- stragglers.
