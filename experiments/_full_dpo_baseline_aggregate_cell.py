"""
Full-DPO / all-layers-LoRA baseline aggregate cell (Reviewer Xoi2 sugg. 3).

Computes delta-vs-base for the baseline runs and seed-aggregates the
all-layers-LoRA control across {17, 1729, 65537}. Writes
results/multi_anchor_delta_vs_base__dpo-baselines.json and prints a
cross-lingual comparison against RD-DPO-k4 (x4, 3-seed) and full-model DPO.

The question this answers: is the flat cross-lingual result an artifact of
RD-DPO restricting training to 4 probe-selected layers? If all-layers LoRA
and full DPO also stay <= 0 on cross-lingual, the answer is no.
"""

import json
import math

PEER_REFUSAL_DIR = PAPER2_ROOT / "experiments" / "judged"
DATA_COND = "bal-e6-x4"
DIMS      = ["toxicity", "jailbreak", "overrefusal", "crosslingual"]
ANCHORS   = ["qwen2.5-3b", "llama-3.2-3b", "gemma-3-4b"]
LORA_ALL_SEEDS = [17, 1729, 65537]
FULL_SEEDS     = {"qwen2.5-3b": 17, "llama-3.2-3b": 17}  # gemma full skipped
P2_NAME   = {"qwen2.5-3b": "qwen2.5-3b", "llama-3.2-3b": "llama3.2-3b",
             "gemma-3-4b": "gemma3-4b"}
SEED_SWEEP = RESULTS_DIR / "multi_anchor_delta_vs_base__seed-sweep.json"


def _load_p2(anchor_p2):
    d = json.loads((PEER_REFUSAL_DIR / f"refusal_{anchor_p2}.json").read_text())
    by = {}
    for j in d["judgments"]:
        dim = j.get("dim")
        if dim in DIMS:
            by.setdefault(dim, {})[j["id"]] = j.get("judge_label", "")
    return by


def _delta_for(safety_path, p2_by, dim):
    if not safety_path.exists():
        return None
    p3 = json.loads(safety_path.read_text())
    p3j = p3["judged"].get(dim, [])
    ids = [j["id"] for j in p3j]
    p2d = p2_by.get(dim, {})
    inter = [i for i in ids if i in p2d]
    if not inter:
        return None
    base_n = sum(1 for i in inter if p2d[i] == "refusal")
    p3by = {j["id"]: j["judge_label"] for j in p3j}
    rd_n = sum(1 for i in inter if p3by[i] == "refusal")
    return round((rd_n - base_n) / len(inter) * 100, 1)


def _mean_se(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    n = len(vals); m = sum(vals) / n
    if n < 2:
        return {"mean": round(m, 2), "se": None, "n_seeds": n, "values": vals}
    se = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1) / n)
    return {"mean": round(m, 2), "se": round(se, 2), "n_seeds": n, "values": vals}


def _safety(short_a, cond_tag, seed):
    return RESULTS_DIR / f"{short_a}__{cond_tag}__seed{seed}__safety.json"


# --- RD-DPO-k4 x4 3-seed reference (from the seed sweep) ---
rdpo_ref = {}
if SEED_SWEEP.exists():
    ss = json.loads(SEED_SWEEP.read_text()).get("seed_aggregated", {})
    for a in ANCHORS:
        cell = ss.get(a, {}).get("x4", {}).get("crosslingual")
        if cell:
            rdpo_ref[a] = {"mean": cell["mean"], "se": cell.get("se")}

out = {
    "comparison_method": "intersect Paper 3 holdout IDs with Paper 2 per-id refusal judgments",
    "judge": "openai/gpt-5-mini", "split": "split_v1.holdout",
    "experiment": "Upper-bound baselines vs RD-DPO-k4: all-layers LoRA (3 seeds) and full-model DPO (seed 17)",
    "rdpo_k4_x4_reference_crosslingual": rdpo_ref,
    "dpo_lora_all": {"per_seed": {}, "seed_aggregated": {}},
    "dpo_full": {},
}

lora_cond = f"dpo-lora-all-{DATA_COND}"
full_cond = f"dpo-full-{DATA_COND}"

for a in ANCHORS:
    p2 = _load_p2(P2_NAME[a])
    # all-layers LoRA, three seeds
    per_seed, agg = {}, {}
    for s in LORA_ALL_SEEDS:
        cell = {dim: _delta_for(_safety(a, lora_cond, s), p2, dim) for dim in DIMS}
        per_seed[f"seed_{s}"] = cell
    for dim in DIMS:
        agg[dim] = _mean_se([per_seed[f"seed_{s}"][dim] for s in LORA_ALL_SEEDS])
    out["dpo_lora_all"]["per_seed"][a] = per_seed
    out["dpo_lora_all"]["seed_aggregated"][a] = agg
    # full-model DPO, seed 17 (Qwen, Llama)
    if a in FULL_SEEDS:
        s = FULL_SEEDS[a]
        out["dpo_full"][a] = {dim: _delta_for(_safety(a, full_cond, s), p2, dim) for dim in DIMS}

out_path = RESULTS_DIR / "multi_anchor_delta_vs_base__dpo-baselines.json"
out_path.write_text(json.dumps(out, indent=2))
print(f"wrote {out_path}\n")

# --- readable cross-lingual comparison ---
def _fmt(cell):
    if cell is None:
        return "n/a"
    if isinstance(cell, dict):
        if cell.get("se") is not None:
            return f"{cell['mean']:+.1f}+/-{cell['se']:.1f}"
        return f"{cell['mean']:+.1f} (1s)"
    return f"{cell:+.1f}"

print("=== Cross-lingual delta-vs-base: is it the probe-layer restriction? ===")
hdr = f"  {'anchor':<14} {'RD-DPO k4 x4 (3s)':>18} {'all-layers LoRA (3s)':>22} {'full-DPO (s17)':>16}"
print(hdr); print("  " + "-" * (len(hdr) - 2))
for a in ANCHORS:
    rd = rdpo_ref.get(a)
    rd_s = f"{rd['mean']:+.1f}+/-{rd['se']:.1f}" if rd and rd.get("se") is not None else (f"{rd['mean']:+.1f}" if rd else "n/a")
    la = out["dpo_lora_all"]["seed_aggregated"][a]["crosslingual"]
    fl = out["dpo_full"].get(a, {}).get("crosslingual") if a in out["dpo_full"] else None
    print(f"  {a:<14} {rd_s:>18} {_fmt(la):>22} {_fmt(fl):>16}")

print("\n  All four dims, all-layers LoRA (3-seed mean +/- SE):")
for a in ANCHORS:
    cells = out["dpo_lora_all"]["seed_aggregated"][a]
    line = "  ".join(f"{dim[:3]} {_fmt(cells[dim])}" for dim in DIMS)
    print(f"    {a:<14} {line}")

print("\n  Read: if all-layers LoRA and full DPO cross-lingual stay <= 0 like")
print("  RD-DPO, the gap is not caused by the probe-layer restriction.")
