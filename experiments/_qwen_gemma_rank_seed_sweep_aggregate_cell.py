"""
Qwen + Gemma rank x seed sweep aggregation cell.

Builds a three-seed rank-sweep delta JSON for Qwen and Gemma, so the Qwen
and Gemma rows of Table 7 (rank-sweep) can be reported as mean +/- SE across
{17, 1729, 65537} at every rank -- removing the single-seed asymmetry that
reviewer hHpd (W2) and Xoi2 (W2) flagged.

Seed availability:
  - r=16 : the x4 base condition `rd-dpo-k4-bal-e6-x4__seed{17,1729,65537}`
           (already three-seed from the seed sweep -- reused, not retrained).
  - r=64 : seed=17 (v1.1 rank sweep) + seed={1729,65537} (this sweep).
  - r=128: seed=17 (v1.1 rank sweep) + seed={1729,65537} (this sweep).

Output:
  results/multi_anchor_delta_vs_base__rd-dpo-k4-bal-e6-x4-rank-sweep-qwen-gemma-multi-seed.json

Kept in a separate file (parallel to the Llama multi-seed file) so the v1.1
single-seed rank-sweep JSON is not clobbered. Mirrors the structure of
experiments/_llama_rank_seed_sweep_aggregate_cell.py.
"""

import json
import math
from pathlib import Path

# Reuses RESULTS_DIR + PAPER2_ROOT from earlier cells in nb04.
PEER_REFUSAL_DIR = PAPER2_ROOT / "experiments" / "judged"

LRS_BASE_COND = "rd-dpo-k4-bal-e6-x4"
RANKS_ALL     = [16, 64, 128]
SEEDS         = [17, 1729, 65537]
DIMS          = ["toxicity", "jailbreak", "overrefusal", "crosslingual"]
ANCHORS_P3    = ["qwen2.5-3b", "gemma-3-4b"]

P2_NAME = {
    "qwen2.5-3b": "qwen2.5-3b",
    "gemma-3-4b": "gemma3-4b",
}


def _load_p2(anchor_p2):
    path = PEER_REFUSAL_DIR / f"refusal_{anchor_p2}.json"
    if not path.exists():
        raise FileNotFoundError(f"Paper 2 file not found: {path}")
    d = json.loads(path.read_text())
    by = {}
    for j in d["judgments"]:
        dim = j.get("dim")
        if dim not in DIMS: continue
        by.setdefault(dim, {})[j["id"]] = j.get("judge_label", "")
    return by


def _safety_path(anchor_p3, lora_r, seed):
    cond_tag = LRS_BASE_COND if lora_r == 16 else f"{LRS_BASE_COND}-r{lora_r}"
    return RESULTS_DIR / f"{anchor_p3}__{cond_tag}__seed{seed}__safety.json"


def _delta_for(p3_safety, p2_by, dim):
    p3_judged = p3_safety["judged"][dim]
    holdout_ids = [j["id"] for j in p3_judged]
    if not holdout_ids: return None
    p2_dim = p2_by.get(dim, {})
    inter = [i for i in holdout_ids if i in p2_dim]
    if not inter: return None
    n = len(inter)
    base_n = sum(1 for i in inter if p2_dim[i] == "refusal")
    p3_by  = {j["id"]: j["judge_label"] for j in p3_judged}
    rdpo_n = sum(1 for i in inter if p3_by[i] == "refusal")
    return {
        "n":            n,
        "base_refusal": round(base_n / n, 4),
        "rdpo_refusal": round(rdpo_n / n, 4),
        "delta_pp":     round((rdpo_n - base_n) / n * 100, 1),
    }


def _mean_se(values):
    vals = [v for v in values if v is not None]
    if not vals: return None
    n = len(vals); m = sum(vals) / n
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
        "experiment": "LoRA-rank ablation at fixed data scale (x4); Qwen + Gemma rows multi-seed at r in {16, 64, 128}",
        "ranks_evaluated": RANKS_ALL,
        "seeds": SEEDS,
        "shared_hyperparams": {
            "beta": 0.1, "epochs": 6, "warmup_steps": 2,
            "per_device_train_batch_size": 4, "gradient_accumulation_steps": 8,
            "rebalance": "down-sample help-side (overref) to match refuse-side (core_s2 + xl)",
            "probe": "top-of-net selected_blocks.json",
            "data": "x4 expansion (~434 pairs/anchor after rebalance)",
            "lora_alpha_rule": "alpha = 2 * rank",
            "best_lr_per_anchor": {"qwen2.5-3b": 2e-5, "gemma-3-4b": 5e-6},
        },
        "deltas_pp": {},         # summary: mean per (anchor, rank, dim)
        "per_seed": {},          # per (anchor, rank, seed, dim)
        "seed_aggregated": {},   # mean +/- SE per (anchor, rank, dim)
    }

    for anchor_p3 in ANCHORS_P3:
        p2 = _load_p2(P2_NAME[anchor_p3])
        per_seed_block = {}
        agg_block = {}
        summary_block = {}

        for r in RANKS_ALL:
            per_seed = {}
            for seed in SEEDS:
                path = _safety_path(anchor_p3, r, seed)
                if not path.exists():
                    print(f"[{anchor_p3} r={r} seed{seed}] {path.name} missing")
                    continue
                p3 = json.loads(path.read_text())
                per_seed[f"seed_{seed}"] = {dim: _delta_for(p3, p2, dim) for dim in DIMS}
            per_seed_block[f"r_{r}"] = per_seed

            agg = {}
            summary = {}
            for dim in DIMS:
                vals = [cell[dim]["delta_pp"] for cell in per_seed.values()
                        if cell.get(dim) is not None]
                cell_agg = _mean_se(vals)
                agg[dim] = cell_agg
                if cell_agg is None:
                    summary[dim] = None
                else:
                    summary[dim] = {"delta_pp": cell_agg["mean"],
                                    "delta_pp_se": cell_agg["se"],
                                    "n_seeds": cell_agg["n_seeds"]}
            agg_block[f"r_{r}"] = agg
            summary_block[f"r_{r}"] = summary

        out["per_seed"][anchor_p3] = per_seed_block
        out["seed_aggregated"][anchor_p3] = agg_block
        out["deltas_pp"][anchor_p3] = summary_block

    out_path = RESULTS_DIR / "multi_anchor_delta_vs_base__rd-dpo-k4-bal-e6-x4-rank-sweep-qwen-gemma-multi-seed.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")

    # Human-readable summary: cross-lingual delta by (anchor, rank)
    for anchor_p3 in ANCHORS_P3:
        print(f"\n=== {anchor_p3} rank sweep: cross-lingual delta-vs-base (3 seeds) ===")
        hdr = f"  {'rank':<6} | {'seed=17':>9} | {'seed=1729':>11} | {'seed=65537':>12} | {'mean +/- SE':>15}"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        per_seed_block = out["per_seed"][anchor_p3]
        agg_block = out["seed_aggregated"][anchor_p3]
        for r in RANKS_ALL:
            per_seed = per_seed_block.get(f"r_{r}", {})
            cells = []
            for seed in SEEDS:
                cell = per_seed.get(f"seed_{seed}", {}).get("crosslingual")
                cells.append(f"{cell['delta_pp']:+5.1f}" if cell else " n/a")
            agg_xl = agg_block.get(f"r_{r}", {}).get("crosslingual")
            if agg_xl is not None and agg_xl.get("se") is not None:
                agg_str = f"{agg_xl['mean']:+5.1f} +/- {agg_xl['se']:>4.1f}"
            elif agg_xl is not None:
                agg_str = f"{agg_xl['mean']:+5.1f}    n=1   "
            else:
                agg_str = "    n/a       "
            print(f"  {'r=' + str(r):<6} | {cells[0]:>9} | {cells[1]:>11} | {cells[2]:>12} | {agg_str:>15}")

        # All-dim summary at r=128 (the cells most under reviewer scrutiny)
        print(f"  --- {anchor_p3} r=128 all four dims (mean +/- SE) ---")
        agg_128 = agg_block.get("r_128", {})
        for dim in DIMS:
            v = agg_128.get(dim)
            if v is None: continue
            if v.get("se") is not None:
                print(f"    {dim:<14}  mean={v['mean']:+5.1f} +/- {v['se']:>4.1f}  values={v['values']}")
            else:
                print(f"    {dim:<14}  mean={v['mean']:+5.1f}  n={v.get('n_seeds')}  values={v['values']}")


main()
