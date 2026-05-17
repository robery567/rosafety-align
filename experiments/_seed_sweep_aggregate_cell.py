"""
Seed-sweep aggregation cell.

Combines the 12 new seed-sweep safety.json files (3 anchors x 2
data scales x 2 new seeds) plus the existing seed=17 baselines
with Paper 2 base anchor refusal labels to produce a single
delta-vs-base JSON that exposes:

  - per-(anchor, data_scale, seed) delta-pp on the four refusal
    dimensions (tox, jb, or, xl)
  - per-(anchor, data_scale) seed-aggregated mean +/- SE across
    the three seeds {17, 1729, 65537}
  - cross-anchor mean per data_scale per dim (the load-bearing
    dissociation summary statistic)

Output:
  results/multi_anchor_delta_vs_base__seed-sweep.json
"""

import json
import math
from pathlib import Path

from prompts import ANCHORS

# Reuses RESULTS_DIR + PAPER2_ROOT from earlier cells in nb04.
PEER_REFUSAL_DIR = PAPER2_ROOT / "experiments" / "judged"

ALL_SEEDS = [17, 1729, 65537]  # registered seeds; 17 is the existing baseline
DATA_SCALES = ("e6", "x4")
DIMS = ["toxicity", "jailbreak", "overrefusal", "crosslingual"]

# Per-anchor + per-scale condition tag: must match the seed=17 file naming.
PER_ANCHOR_E6 = {
    "qwen2.5-3b":   "rd-dpo-k4-bal-e6-lr2e5",
    "llama-3.2-3b": "rd-dpo-k4-bal-e6-lr2e5",
    "gemma-3-4b":   "rd-dpo-k4-bal-e6",
}
PER_ANCHOR_X4 = {
    "qwen2.5-3b":   "rd-dpo-k4-bal-e6-x4",
    "llama-3.2-3b": "rd-dpo-k4-bal-e6-x4",
    "gemma-3-4b":   "rd-dpo-k4-bal-e6-x4",
}
COND_FOR = {"e6": PER_ANCHOR_E6, "x4": PER_ANCHOR_X4}

# Map Paper 3 short-name to Paper 2 short-name
P2_NAME = {
    "qwen2.5-3b":   "qwen2.5-3b",
    "llama-3.2-3b": "llama3.2-3b",
    "gemma-3-4b":   "gemma3-4b",
}


def _load_p2_judgments_for_anchor(anchor_p2):
    path = PEER_REFUSAL_DIR / f"refusal_{anchor_p2}.json"
    if not path.exists():
        raise FileNotFoundError(f"Paper 2 file not found: {path}")
    d = json.loads(path.read_text())
    by_dim = {}
    for j in d["judgments"]:
        dim = j.get("dim")
        if dim not in DIMS: continue
        by_dim.setdefault(dim, {})[j["id"]] = j.get("judge_label", "")
    return by_dim


def _load_p3_safety(anchor_p3, data_scale, seed):
    cond_tag = COND_FOR[data_scale][anchor_p3]
    run_tag = f"{anchor_p3}__{cond_tag}__seed{seed}"
    path = RESULTS_DIR / f"{run_tag}__safety.json"
    if not path.exists():
        raise FileNotFoundError(f"safety file not found: {path}")
    return json.loads(path.read_text())


def _delta_for(p3_safety, p2_by_dim, dim):
    p3_judged = p3_safety["judged"][dim]
    holdout_ids = [j["id"] for j in p3_judged]
    if not holdout_ids:
        return None
    p2_dim = p2_by_dim.get(dim, {})
    inter = [i for i in holdout_ids if i in p2_dim]
    if not inter:
        return None
    base_n_ref = sum(1 for i in inter if p2_dim[i] == "refusal")
    base_rate  = base_n_ref / len(inter)
    p3_by_id   = {j["id"]: j["judge_label"] for j in p3_judged}
    rdpo_n_ref = sum(1 for i in inter if p3_by_id[i] == "refusal")
    rdpo_rate  = rdpo_n_ref / len(inter)
    return {
        "n":             len(inter),
        "base_refusal":  round(base_rate, 4),
        "rdpo_refusal":  round(rdpo_rate, 4),
        "delta_pp":      round((rdpo_rate - base_rate) * 100, 1),
    }


def _mean_se(values):
    """Population mean + standard error of the mean across seeds."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    n = len(vals)
    m = sum(vals) / n
    if n < 2:
        return {"mean": round(m, 2), "se": None, "n_seeds": n,
                "values": [round(v, 2) for v in vals]}
    var = sum((v - m) ** 2 for v in vals) / (n - 1)
    se = math.sqrt(var / n)
    return {"mean": round(m, 2), "se": round(se, 2), "n_seeds": n,
            "values": [round(v, 2) for v in vals]}


def main():
    out = {
        "comparison_method": "intersect Paper 3 holdout IDs with Paper 2 per-id refusal judgments",
        "judge": "openai/gpt-5-mini",
        "split": "split_v1.holdout",
        "experiment": "Seed sweep over {17, 1729, 65537} on the e6 and x4 conditions",
        "seeds": ALL_SEEDS,
        "data_scales": list(DATA_SCALES),
        "shared_hyperparams": {
            "beta": 0.1, "epochs": 6, "warmup_steps": 2,
            "per_device_train_batch_size": 4, "gradient_accumulation_steps": 8,
            "lora_r": 16, "lora_alpha": 32,
            "rebalance": "down-sample help-side (overref) to match refuse-side (core_s2 + xl)",
            "probe": "top-of-net selected_blocks.json",
        },
        "per_anchor_condition_tags": {"e6": PER_ANCHOR_E6, "x4": PER_ANCHOR_X4},
        "per_seed": {},
        "seed_aggregated": {},
        "cross_anchor_mean_per_scale": {},
    }

    # Per (anchor, scale, seed) deltas
    for anchor in ANCHORS:
        anchor_p3 = {
            "Qwen/Qwen2.5-3B-Instruct":          "qwen2.5-3b",
            "meta-llama/Llama-3.2-3B-Instruct":  "llama-3.2-3b",
            "google/gemma-3-4b-it":              "gemma-3-4b",
        }[anchor]
        anchor_p2 = P2_NAME[anchor_p3]
        p2_by_dim = _load_p2_judgments_for_anchor(anchor_p2)

        anchor_block = {}
        for data_scale in DATA_SCALES:
            scale_block = {}
            for seed in ALL_SEEDS:
                try:
                    p3_safety = _load_p3_safety(anchor_p3, data_scale, seed)
                except FileNotFoundError as e:
                    print(f"[{anchor_p3} {data_scale} seed{seed}] {e}")
                    continue
                cell = {dim: _delta_for(p3_safety, p2_by_dim, dim) for dim in DIMS}
                scale_block[f"seed_{seed}"] = cell
            anchor_block[data_scale] = scale_block
        out["per_seed"][anchor_p3] = anchor_block

    # Per (anchor, scale) seed-aggregated mean +/- SE
    for anchor_p3, anchor_block in out["per_seed"].items():
        seed_agg = {}
        for data_scale, scale_block in anchor_block.items():
            agg = {}
            for dim in DIMS:
                vals = []
                for seed_key, cell in scale_block.items():
                    v = cell.get(dim)
                    if v is not None:
                        vals.append(v["delta_pp"])
                agg[dim] = _mean_se(vals)
            seed_agg[data_scale] = agg
        out["seed_aggregated"][anchor_p3] = seed_agg

    # Cross-anchor mean per scale per dim (the dissociation summary stat)
    for data_scale in DATA_SCALES:
        per_dim = {}
        for dim in DIMS:
            anchor_means = []
            for anchor_p3 in out["seed_aggregated"]:
                cell = out["seed_aggregated"][anchor_p3][data_scale].get(dim)
                if cell and cell.get("mean") is not None:
                    anchor_means.append(cell["mean"])
            per_dim[dim] = _mean_se(anchor_means)
        out["cross_anchor_mean_per_scale"][data_scale] = per_dim

    out_path = RESULTS_DIR / "multi_anchor_delta_vs_base__seed-sweep.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")

    # Human-readable summary
    print("\n=== seed-sweep delta-vs-base (mean +/- SE across seeds) ===")
    for data_scale in DATA_SCALES:
        print(f"\n  data scale = {data_scale}")
        hdr = f"  {'anchor':<14} | {'tox':>14} | {'jb':>14} | {'or':>14} | {'xl':>14}"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for anchor_p3, sa in out["seed_aggregated"].items():
            row_cells = []
            for dim in DIMS:
                v = sa[data_scale].get(dim)
                if v and v.get("se") is not None:
                    row_cells.append(f"{v['mean']:+6.1f} +/- {v['se']:>4.1f}")
                elif v:
                    row_cells.append(f"{v['mean']:+6.1f}    n=1   ")
                else:
                    row_cells.append("    n/a       ")
            print(f"  {anchor_p3:<14} | " + " | ".join(row_cells))
        # cross-anchor mean line
        cm = out["cross_anchor_mean_per_scale"][data_scale]
        cm_row = []
        for dim in DIMS:
            v = cm.get(dim)
            if v and v.get("se") is not None:
                cm_row.append(f"{v['mean']:+6.1f} +/- {v['se']:>4.1f}")
            elif v:
                cm_row.append(f"{v['mean']:+6.1f}    n=1   ")
            else:
                cm_row.append("    n/a       ")
        print(f"  {'cross-anchor':<14} | " + " | ".join(cm_row))

    # Cross-lingual focus block
    print("\n=== cross-lingual delta-vs-base by seed (the W1 question) ===")
    for anchor_p3, anchor_block in out["per_seed"].items():
        for data_scale in DATA_SCALES:
            cells = []
            for seed in ALL_SEEDS:
                cell = anchor_block.get(data_scale, {}).get(f"seed_{seed}")
                if cell and cell.get("crosslingual"):
                    cells.append(f"seed{seed}: {cell['crosslingual']['delta_pp']:+5.1f}")
                else:
                    cells.append(f"seed{seed}: n/a")
            print(f"  {anchor_p3:<14} {data_scale}: " + "  |  ".join(cells))

    # Sign-test summary on cross-lingual at x4 (the load-bearing claim)
    print("\n=== cross-lingual sign test on x4 (3 anchors x 3 seeds = 9 cells) ===")
    pos = neg = zero = 0
    cells = []
    for anchor_p3, anchor_block in out["per_seed"].items():
        for seed in ALL_SEEDS:
            cell = anchor_block.get("x4", {}).get(f"seed_{seed}")
            if cell and cell.get("crosslingual"):
                d = cell["crosslingual"]["delta_pp"]
                cells.append(d)
                if d > 0: pos += 1
                elif d < 0: neg += 1
                else: zero += 1
    print(f"  {len(cells)}/9 cells available; pos={pos}, neg={neg}, zero={zero}")
    if cells:
        m = sum(cells) / len(cells)
        var = sum((v - m) ** 2 for v in cells) / max(len(cells) - 1, 1)
        se = math.sqrt(var / len(cells)) if len(cells) > 1 else float("nan")
        print(f"  pooled mean = {m:+.2f} pp,  SE = {se:.2f} pp,  N = {len(cells)}")


main()
