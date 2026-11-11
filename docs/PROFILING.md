Sailor is based on profiling information for the model and cluster.
* We profile a model's memory footprint and forward/backward pass under different scenarios. For each GPU type, we need 1 machine with >=1 GPUs to profile multiple tensor parallelism degrees. For LLMs, we dont profile all transformer layers since they are identical. This enables us to profile large models on a single machine.
* We profile a cluster's network bandwidth across machines using Torch NCCL-based send/recv for varying message sizes. We then fit a function to get a relation between network bandwidth and message size.

## 1. Network profiling

When adding a new GPU or cluster, the first step is to profile the network bandwidth across machines.
Sailor, as well as the rest baselines need this information. We will profile network bandwidth across GPUs on the same machine, and across machines, and update the proper files. For the remaining of this subsection, we will assume you are profiling GPU *X*, and the machine has 4 GPUs of type *X*.

### 1. Profile bandwidth + update profiling files

1. Run profiling script

Go to 'sailor/profiling/networking' directory. The file run.sh is called with arguments <world_size>, <start_rank>, <IP>, <port>. Assume a free port <p>

For a pair of 2 GPUs:

```bash
bash run.sh 2 0 127.0.0.1 <p>
bash run.sh 2 1 127.0.0.1 <p>
```
This will generate a data2.csv file.

For a pair of 4 GPUs:

```bash
bash run.sh 4 0 127.0.0.1 <p>
bash run.sh 4 2 127.0.0.1 <p>
```

This will generate a data4.csv file.

2. Fit function to get network coefficients (used by Sailor), and max network bandwidth (used by baselines)

For data2.csv, do:

```bash
python fit_func.py data_2.csv
```
This will output a list of coefficients, and the max network bandwidth found.
Do the same for the other generated files as well.

3. Add generated info to the appropriate files

Put the coeffecients and max network bandwidth found in the [sailor/providers/intra_node_bandwidths.json](/sailor/providers/intra_node_bandwidths.json) and [sailor/providers/network_coeffs_all_homo.json](/sailor/providers/network_coeffs_all_homo.json) (under the "intra" keyword) for all gpu counts (look at the examples already provided).


### 2. Profile bandwidth across machines + update profiling files

1. Run profiling script

Do the same as before, but:
* Change '127.0.0.1' to an IP address/hostname discoverable across machines. It should be the IP address/hostname of the node with <start_rank>==0
* Also run for world_size=8 (assuming again 4 GPUs per node)

2. Fit function to get network coefficients (used by Sailor), and max network bandwidth (used by baselines)

Same as before

3. Add generated info to the appropriate files

Same as before, but now update:
1. the [sailor/providers/network_coeffs_all_homo.json](/sailor/providers/network_coeffs_all_homo.json) under "inter"
2. the [sailor/providers/multizone_bandwidths.json](/sailor/providers/multizone_bandwidths.json)

## 2. Model Profiling

### 1. Profile the model

The script *profile.sh* profiles a given model under different combinations of tensor parallelism and microbatch size. We use a big enough *largest* microbatch size, and OOM failures are expected and do not affect profiling. Of course, you can adapt this accordingly. Make sure to provide the *model_name* and *prof_dir*.


```bash
cd /root/sailor/third_party/Megatron-DeepSpeed
bash profile.sh <model_name> <prof_dir>
```

e.g. `bash profile.sh OPT-350 tests_profs`
A file will be generated in the form of *profile_model_gpu_tmp_mbs.json*, where *tmp* is the Tensor Model parallel degree, and *mbs* is the microbatch size.

### 2. Generate results for Sailor

After constructing the files for the different combinations of *tmp* and *mbs*, combine them by running:

```bash

cd /root/sailor/sailor/profiling
python gather_profs.py --model-name <model> --gpu-type <gpu_type> --profile-dir <prof_dir> --sailor-parent-dir <path_to_sailor_dir>

```
e.g. `python gather_profs.py --model-name OPT-350 --gpu-type A100 --profile-dir ~/sailor/third_party/Megatron-DeepSpeed/test_profs --sailor-parent-dir /root`
This will:
1. Generate a file of the form *model/gpu/profile.json* under *sailor/Planner/sailor_planner/profiles/*, containing timing information.
2. Append memory-specific info for the model in the file *sailor/Planner/llm_info.json*.
3. Append timing-specific info for this combination of model and gpu in the file *sailor/Planner/simulations/profiles_tmp.json*, which is the file used for simulations

### 3. Generate results for baselines

If you are adding a new GPU type, make sure that network profiling is done, as mentioned above.
Also, make sure to update the GPU_MEMORY_GB dictionary in [sailor/Planner/simulations/constants.py](/sailor/Planner/simulations/constants.py).
All baselines require some profiling information that we can extract from the data we already collected. Some baselines require some extra information which we collect:

### For Varuna

1. Collect the extra info:

```bash
cd /root/sailor/third_party/Megatron-DeepSpeed
bash profile_varuna.sh <model_name> <prof_dir>
```
This should generate a json file under <prof_dir>.

2. Generate profiles:

```bash

cd /root/sailor/sailor/profiling/baselines
bash generate_varuna.sh <path_to_the_generated_file_from_prev_step>
```

Adapt the GPU type, and number of GPUs per VM accordingly in the file

### For Galvatron

1. Collect the extra info:

```bash
cd /root/sailor/third_party/Megatron-DeepSpeed
bash profile_galvatron.sh <model_name> <prof_dir>
Copy the <prof_dir>/memory_profiles.json under a directory sailor/Planner/baselines/Galvatron/profiles/<model_name>/<gpu_type>
```

2. Get the communication-computation overlap coefficient, for 2, 4 GPUs etc using the [sailor/sailor/profiling/baselines/profile_overlap_galvatron.sh](/sailor/sailor/profiling/baselines/profile_overlap_galvatron.sh) file. Note that you have to do it for every tensor-parallel degree you profiled in step 1. Put the coefficients under [sailor/sailor/Planner/baselines/Galvatron/overlap_coe_dicts.json](/sailor/sailor/Planner/baselines/Galvatron/overlap_coe_dicts.json)

### For the rest baselines (incl Galvatron)

Generate profiles:

```bash

cd /root/sailor/sailor/profiling/baselines
bash generate.sh
```

Adapt the name of the baseline, GPU type, and number of GPUs per machine accordingly in the file.

## 3. Model profiling with Megatron-Core (Alps)

`sailor/mcore/profiler/` profiles with the Megatron-Core copy in `third_party/Megatron-LM` (upstream `e429478`)
instead of Megatron-DeepSpeed, and writes the same `profile_<model>_<gpu>_<tmp>_<mbs>.json` files, so step 2
(`gather_profs.py`) and the planner are unchanged. Models: `LLAMA-3-8`, `QWEN3-30B-A3B`, `MIXTRAL-8X7B` (EP = ETP = 1), defined once in Sailor's training
configs (`sailor/Planner/simulations/configs/training_config_<model>.json`, also read by the simulator and planner);
`sailor/mcore/models.py` turns them into Megatron arguments and adds the settings and environment every run shares. Sequence parallelism is on when TP > 1. The optimizer is not sharded over DP (ZeRO-0, as
original Sailor), with bf16 gradients (`--grad-reduce-in-bf16`); see `docs/moe_iteration_time_model.md`. On
Clariden, from `sailor/mcore/profiler`:

```bash
sbatch profile_sweep.slurm LLAMA-3-8        # TP 1/2/4 x MBS 1/2/4/8 on one node -> $SCRATCH/sailor-profiles/LLAMA-3-8/<job>/
```

Method and output keys: docstring of `profile_mcore.py`. Besides the planner's inputs, each profile has
`layer_parts` (one transformer layer split into attention, router, dispatch, experts, combine, ...),
`optimizer` (parameters updated per GPU) and `gpu_memory`, for the MoE cost model in
`docs/moe_iteration_time_model.md`. `--num-experts K` profiles an MoE model with K experts per layer (files named
`<model>-E<K>`), to time the experts as at EP = E/K:

```bash
sbatch profile_sweep.slurm QWEN3-30B-A3B --num-experts 16
```

Nsight Systems traces for moe-comms' `moetrace` timelines: `--nsys` runs nsys on every rank for iterations
`iters-6..iters-2` (Megatron's `--profile`, `--nvtx-ranges`) and writes `<name>_rank<R>.sqlite` and a `<name>.json`
manifest to `<job>/traces/`. Trace runs are separate: their profiles (disturbed by nsys) stay in `traces/`.

```bash
sbatch profile_sweep.slurm LLAMA-3-8 --tp 2 --mbs 4 --iters 20 --nsys
# from moe-comms:
python -m moetrace extract $SCRATCH/sailor-profiles/LLAMA-3-8/<job>/traces
python -m moetrace view $SCRATCH/sailor-profiles/LLAMA-3-8/<job>/traces/tables -o llama_tp2_mbs4.html
```

Layout of `sailor/mcore/` (Megatron-Core runs on Alps): `models.py` (the models' training configs as Megatron
arguments, shared settings, environment, nsys command), `megatron_hooks.py` (fast NVTX names, routing counts, CPU binding: every rank of a profile or validation run is pinned to its GPU's NUMA node, as srun-launched training; `SAILOR_CPU_BIND=0` leaves it unbound), `alps6.toml` (container), `profiler/` (per-layer
profiles: `profile_mcore.py`, `profile_sweep.py`, `profile_sweep.slurm`), `validation/` (training runs of one plan:
`validation_run.py`, `validation_run.slurm`, per-rank `train_mcore.py`).

### Validation runs (Megatron-Core training of one plan)

`sailor/mcore/validation/validation_run.slurm` runs one plan of a model in `models.py` (same model, settings
and environment as the profiles, through `model_args` and `ENV`) on N nodes, with nsys traces in the layout of moe-comms'
`slurm/40_pretrain_qwen3.slurm`, so moe-comms' `moetrace` reads them unchanged. Regime of the MoE cost model
(`docs/moe_iteration_time_model.md`): `alltoall` dispatcher, ETP 1, ZeRO-0, no overlap flags, mock data. World = N × 4,
DP = world / (TP × PP); the plan is checked before the run (TP within a node, EP divides E and TP × DP, GBS divisible
by MBS × DP, layers by PP).

```bash
cd sailor/mcore/validation
sbatch --nodes=1 validation_run.slurm QWEN3-30B-A3B --tp 1 --ep 4 --mbs 1 --gbs 16
sbatch --nodes=2 validation_run.slurm MIXTRAL-8X7B --tp 2 --pp 2 --ep 2 --mbs 1 --gbs 16 --layers 8
# options: --layers L (default: full model), --iters I (20), --no-nsys, --gpu-metrics-ranks "0"
# from moe-comms:
python -m moetrace extract $SCRATCH/sailor-validation/<job> && python -m moetrace view $SCRATCH/sailor-validation/<job>/tables -o page.html
```

Output in `$SCRATCH/sailor-validation/<job>/`: `<name>.log` (Megatron's output), `<name>.json` (moetrace manifest,
plus the plan and Megatron's per-iteration times; the validation time is the median over the untraced iterations
6..iters-7), traces `<name>_rank<R>.{nsys-rep,sqlite}` (iterations iters-6..iters-2, NCCL payloads kept), routing
counts `<name>_rank<R>_routing.npz`, and `<name>_plan_config.json` in the format of Sailor's validation
(`Planner/simulations/validation/clariden/<MODEL>/N<n>/`, `"real"` in seconds, plus `"ep"`). Traced runs use moe-comms'
fast NVTX names (`mcore/megatron_hooks.py`; Megatron's own costs ~0.1 ms per range); the profiler does the same with `--nsys`.
