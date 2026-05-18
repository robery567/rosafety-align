"""
Llama rank x seed sweep aggregation cell.

Combines the existing seed=17 rank-sweep safety files (Qwen, Llama, Gemma
at r in {16, 64, 128}) with the 4 new Llama rank-seed-sweep files
(r in {64, 128} x seed in {1729, 65537}) to produce an updated rank-sweep
delta JSON that:

  - keeps the original seed=17 cells for Qwen and Gemma at all 3 ranks
    (the cells that drove the v1.1 rank-ablation table);
  - adds a per-seed Llama row for r=64 and r=128 with mean +/- SE across
    {seed=17, seed=1729, seed=65537};
  - keeps Llama r=16 single-seed (we did not multi-seed the baseline rank
    in this sweep; the manuscript v1.2 e6/x4 multi-seed cells already
    report Llama at r=16 as 3-seed via the seed-sweep aggregator).

Output:
  results/multi_anchor_delta_vs_base__rd-dpo-k4-bal-e6-x4-rank-sweep-llama-multi-seed.json

The separate file lets the manuscript update Table 7 (rank-sweep) to a
mean +/- SE format on the Llama row without retroactively rewriting the
Qwen / Gemma cells (which stay single-seed seed=17, as in v1.1).
"""

import json
import math
from pathlib import Path

# Reuses RESULTS_DIR + PAPER2_ROOT from earlier cells in nb04.
PEER_REFUSAL_DIR = PAPER2_ROOT / "experiments" / "judged"

LRS_BASE_COND = "rd-dpo-k4-bal-e6-x4"
RANKS_ALL     = [16, 64, 128]
LLAMA_SEEDS   = [17, 1729, 65537]
DIMS          = ["toxicity", "jailbreak", "overrefusal", "crosslingual"]

# Anchors that stay seed=17 in the rank-sweep table (v1.1 cells unchanged).
QWEN_GEMMA_SEED = 17

P2_NAME = {
    "qwen2.5-3b":   "qwen2.5-3b",
    "llama-3.2-3b": "llama3.2-3b",
    "gemma-3-4b":   "gemma3-4b",
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
    if lora_r == 16:
        cond_tag = LRS_BASE_COND
    else:
        cond_tag = f"{LRS_BASE_COND}-r{lora_r}"
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
        "experiment": "LoRA-rank ablation at fixed data scale (x4); Llama row multi-seed at r in {64, 128}",
        "ranks_evaluated": RANKS_ALL,
        "shared_hyperparams": {
            "beta": 0.1, "epochs": 6, "warmup_steps": 2,
            "per_device_train_batch_size": 4, "gradient_accumulation_steps": 8,
            "rebalance": "down-sample help-side (overref) to match refuse-side (core_s2 + xl)",
            "probe": "top-of-net selected_blocks.json",
            "data": "x4 expansion (~434 pairs/anchor after rebalance)",
            "lora_alpha_rule": "alpha = 2 * rank",
        },
        "deltas_pp": {},
        "llama_per_seed": {},   # per-rank, per-seed cells for Llama
        "llama_seed_aggregated": {},  # mean +/- SE per rank for Llama
    }

    # Qwen + Gemma: load seed=17 cells (unchanged from v1.1 rank sweep)
    for anchor_p3 in ("qwen2.5-3b", "gemma-3-4b"):
        anchor_block = {}
        p2 = _load_p2(P2_NAME[anchor_p3])
        for r in RANKS_ALL:
            path = _safety_path(anchor_p3, r, QWEN_GEMMA_SEED)
            if not path.exists():
                print(f"[{anchor_p3} r={r}] {path} missing")
                continue
            p3 = json.loads(path.read_text())
            cells = {dim: _delta_for(p3, p2, dim) for dim in DIMS}
            anchor_block[f"r_{r}"] = cells
        out["deltas_pp"][anchor_p3] = anchor_block

    # Llama: r=16 stays seed=17; r in {64, 128} aggregate across {17, 1729, 65537}.
    p2_llama = _load_p2(P2_NAME["llama-3.2-3b"])
    llama_block = {}
    llama_per_seed_block = {}
    llama_agg_block = {}

    # r=16 (seed 17 only; matches existing rank-sweep)
    path = _safety_path("llama-3.2-3b", 16, QWEN_GEMMA_SEED)
    if path.exists():
        p3 = json.loads(path.read_text())
        llama_block["r_16"] = {dim: _delta_for(p3, p2_llama, dim) for dim in DIMS}

    # r=64 and r=128 multi-seed
    for r in (64, 128):
        per_seed = {}
        for seed in LLAMA_SEEDS:
            path = _safety_path("llama-3.2-3b", r, seed)
            if not path.exists():
                print(f"[Llama r={r} seed{seed}] {path} missing")
                continue
            p3 = json.loads(path.read_text())
            per_seed[f"seed_{seed}"] = {dim: _delta_for(p3, p2_llama, dim) for dim in DIMS}
        llama_per_seed_block[f"r_{r}"] = per_seed

        # Aggregate per-dim across the seeds we have
        agg = {}
        for dim in DIMS:
            vals = []
            for seed_key, cell in per_seed.items():
                v = cell.get(dim)
                if v is not None:
                    vals.append(v["delta_pp"])
            agg[dim] = _mean_se(vals)
        llama_agg_block[f"r_{r}"] = agg

        # Also write a "summary" entry into deltas_pp.llama-3.2-3b for the
        # main per-rank table -- use mean as the headline number.
        summary = {}
        for dim in DIMS:
            cell = agg[dim]
            if cell is None:
                summary[dim] = None
                continue
            summary[dim] = {
                "delta_pp": cell["mean"],
                "delta_pp_se": cell["se"],
                "n_seeds": cell["n_seeds"],
            }
        llama_block[f"r_{r}"] = summary

    out["deltas_pp"]["llama-3.2-3b"] = llama_block
    out["llama_per_seed"]["llama-3.2-3b"] = llama_per_seed_block
    out["llama_seed_aggregated"]["llama-3.2-3b"] = llama_agg_block

    out_path = RESULTS_DIR / "multi_anchor_delta_vs_base__rd-dpo-k4-bal-e6-x4-rank-sweep-llama-multi-seed.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")

    # Human-readable summary
    print("\n=== Llama rank-sweep multi-seed: cross-lingual delta-vs-base ===")
    hdr = f"  {'rank':<6} | {'seed=17':>9} | {'seed=1729':>11} | {'seed=65537':>12} | {'mean +/- SE':>15}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r_key in ["r_16", "r_64", "r_128"]:
        per_seed = llama_per_seed_block.get(r_key, {})
        agg = llama_agg_block.get(r_key, {}).get("crosslingual")
        cells = []
        for seed in LLAMA_SEEDS:
            cell = per_seed.get(f"seed_{seed}", {}).get("crosslingual") if r_key in llama_per_seed_block else None
            if r_key == "r_16" and seed == 17 and "r_16" in llama_block:
                # r=16 single-seed; keep
                cell16 = llama_block["r_16"].get("crosslingual")
                cells.append(f"{cell16['delta_pp']:+5.1f}" if cell16 else "  n/a")
            elif r_key == "r_16":
                cells.append(" n/a")
            elif cell is not None:
                cells.append(f"{cell['delta_pp']:+5.1f}")
            else:
                cells.append(" n/a")
        if agg is not None and agg.get("se") is not None:
            agg_str = f"{agg['mean']:+5.1f} +/- {agg['se']:>4.1f}"
        elif agg is not None:
            agg_str = f"{agg['mean']:+5.1f}    n=1   "
        else:
            agg_str = "    n/a       "
        rank_label = r_key.replace("r_", "r=")
        print(f"  {rank_label:<6} | {cells[0]:>9} | {cells[1]:>11} | {cells[2]:>12} | {agg_str:>15}")

    # Also print all four dims at r=128 (the cell most under reviewer scrutiny)
    print("\n=== Llama r=128 multi-seed: all four refusal dims ===")
    if "r_128" in llama_agg_block:
        agg_128 = llama_agg_block["r_128"]
        for dim in DIMS:
            v = agg_128.get(dim)
            if v is None: continue
            if v.get("se") is not None:
                print(f"  {dim:<14}  mean={v['mean']:+5.1f} +/- SE {v['se']:>4.1f}  (values: {v['values']})")
            else:
                print(f"  {dim:<14}  mean={v['mean']:+5.1f}  n={v.get('n_seeds')}  (values: {v['values']})")


main()
