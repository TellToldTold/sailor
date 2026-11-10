"""Fit the A2A curves from test_alltoall.py output (docs/moe_iteration_time_model.md, F4/F6).

    python fit_alltoall.py a2a_4n.json [more.json ...] --out alltoall_coeffs.json [--timing host|gpu]

Per GPU and A2A of V bytes over an EP group of e GPUs, g of them on each node, n_nodes nodes:
  * one node (e = g): the NVLink curve of g, bytes over NVLink x = V (g - 1) / e;
  * several nodes: the network curve of n_nodes, bytes over the network x = V (e - g) / e.
Each curve is fitted as time = alpha + x / beta (latency plus bytes over bandwidth), by least squares on the
relative time error over all measured sizes. Sailor's quadratic in log2(size) (fit_func.py) misses small messages
by up to 250% here; alpha + x / beta stays within ~10% from 32 KB to 1 GB (moe_todo.md section 3).

--timing host: the time with the host launch inside the window (median_s); gpu: GPU only (gpu_median_s).

Output: {gpu_type: {"timing": ..., "nvlink": {g: {"alpha_s", "beta_GBps", "max_bw_GBps"}},
                    "net": {n_nodes: {...}}, "count_allgather": {e: seconds}}}
time(x) = alpha_s + x / (beta_GBps * 1e9). Points of the same curve from several files are pooled.
"""

import argparse
import json
from collections import defaultdict

import numpy as np

FIELD = {"host": "median_s", "gpu": "gpu_median_s"}


def curve_of(row):
    """(link, key, fraction of V over that link) for a result row."""
    e, g, n = row["e"], row["g"], row["n_nodes"]
    if n == 1:
        assert e == g
        return "nvlink", g, (g - 1) / e
    return "net", n, (e - g) / e


def load(paths, field):
    points = defaultdict(list)  # (gpu_type, link, key) -> [(bytes over the link, seconds)]
    allgather = defaultdict(list)  # (gpu_type, e) -> [seconds]
    for path in paths:
        with open(path) as f:
            data = json.load(f)
        gpu = data["meta"]["gpu_type"]
        for row in data["results"]:
            link, key, frac = curve_of(row)
            for p in row["a2a"]:
                points[(gpu, link, key)].append((p["bytes"] * frac, p[field]))
            allgather[(gpu, row["e"])].append(row["count_allgather"][field])
    return points, allgather


def fit_alpha_beta(x, t):
    """alpha [s], beta [B/s] minimising sum ((alpha + x / beta) / t - 1)^2."""
    a, inv_b = np.linalg.lstsq(np.vstack([np.ones_like(x), x]).T / t[:, None], np.ones_like(t), rcond=None)[0]
    return a, 1 / inv_b


def main(args):
    field = FIELD[args.timing]
    points, allgather = load(args.inputs, field)
    out = {}
    for (gpu, link, key), pts in sorted(points.items()):
        x = np.array([p[0] for p in pts])
        t = np.array([p[1] for p in pts])
        alpha, beta = fit_alpha_beta(x, t)
        rel = np.abs(alpha + x / beta - t) / t
        out.setdefault(gpu, {"timing": args.timing}).setdefault(link, {})[str(key)] = {
            "alpha_s": float(alpha), "beta_GBps": float(beta / 1e9), "max_bw_GBps": float((x / t).max() / 1e9)}
        print(f"{gpu} {link:6} {key}: {len(x)} points, {x.min() / 1e6:.3g}-{x.max() / 1e6:.3g} MB, "
              f"alpha {alpha * 1e6:.0f} us, beta {beta / 1e9:.1f} GB/s, "
              f"time error median {np.median(rel):.0%} max {rel.max():.0%}")
    for (gpu, e), ts in sorted(allgather.items()):
        out.setdefault(gpu, {}).setdefault("count_allgather", {})[str(e)] = float(np.median(ts))
        print(f"{gpu} count all-gather e={e}: {np.median(ts) * 1e6:.1f} us")

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("inputs", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--timing", choices=sorted(FIELD), default="host")
    main(p.parse_args())
