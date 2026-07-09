"""
Full-DPO / all-layers-LoRA-DPO baseline eval cell (Reviewer Xoi2 sugg. 3).

Evaluates the baseline runs trained by _full_dpo_baseline_train_cell.py on the
held-out split, writes per-run safety.json, then prints delta-vs-base per
dimension with the RD-DPO-k4 x4 (seed=17) cell alongside for a direct
"does removing the probe restriction / going full-FT unlock cross-lingual?"
read.

Handles both save formats:
  - all-layers LoRA runs save a PEFT adapter (adapter_config.json present);
  - full-model DPO runs save a full model.

Mirrors _llama_rank_seed_sweep_eval_cell.py (generation, judge, Wilson CI).

Estimated cost: ~30-45 min generation + ~$2-4 OpenRouter.
"""

# --- defensive: ensure prompts.py + llm_judge are importable ---
import sys as _sys
from pathlib import Path as _Path
_PROMPTS_SRC = _Path("/content/drive/MyDrive/PhD/paper3-alignment/src")
if _PROMPTS_SRC.exists() and str(_PROMPTS_SRC) not in _sys.path:
    _sys.path.insert(0, str(_PROMPTS_SRC))

import gc, json, time
from datetime import datetime

import torch
from peft import PeftModel
from scipy.stats import binomtest
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_judge import Judge
from judges import REFUSAL_SYSTEM, REFUSAL_USER_TEMPLATE

# --- baseline eval config ---
SEED       = 17
DATA_COND  = "bal-e6-x4"
DIMS       = ["toxicity", "jailbreak", "overrefusal", "crosslingual"]
BATCH_GEN  = 16
MAX_NEW    = 256

# (anchor, condition-tag) pairs to evaluate. Skips any whose adapter/model
# dir or safety.json is missing/done.
BASELINE_RUNS = [
    ("Qwen/Qwen2.5-3B-Instruct",         f"dpo-lora-all-{DATA_COND}"),
    ("meta-llama/Llama-3.2-3B-Instruct", f"dpo-lora-all-{DATA_COND}"),
    ("google/gemma-3-4b-it",             f"dpo-lora-all-{DATA_COND}"),
    ("Qwen/Qwen2.5-3B-Instruct",         f"dpo-full-{DATA_COND}"),
    ("meta-llama/Llama-3.2-3B-Instruct", f"dpo-full-{DATA_COND}"),
]

PEER_REFUSAL_DIR = PAPER2_ROOT / "experiments" / "judged"
P2_NAME = {"qwen2.5-3b": "qwen2.5-3b", "llama-3.2-3b": "llama3.2-3b",
           "gemma-3-4b": "gemma3-4b"}
RDPO_X4_RANKSWEEP = RESULTS_DIR / "multi_anchor_delta_vs_base__rd-dpo-k4-bal-e6-x4-rank-sweep.json"


def _wilson(k, n, alpha=0.05):
    if n == 0: return (0.0, 0.0)
    return tuple(round(x, 3) for x in binomtest(k, n).proportion_ci(
        confidence_level=1 - alpha, method="wilson"))


def _load_p2(anchor_p2):
    path = PEER_REFUSAL_DIR / f"refusal_{anchor_p2}.json"
    d = json.loads(path.read_text())
    by = {}
    for j in d["judgments"]:
        dim = j.get("dim")
        if dim not in DIMS: continue
        by.setdefault(dim, {})[j["id"]] = j.get("judge_label", "")
    return by


def _eval_one(anchor, cond_tag):
    short_a  = short_of(anchor)
    family_a = family_of(anchor)
    run_tag  = f"{short_a}__{cond_tag}__seed{SEED}"
    run_dir  = ADAPTERS_DIR / run_tag
    out_path = RESULTS_DIR / f"{run_tag}__safety.json"

    if not run_dir.exists():
        print(f"[{run_tag}] run dir not found; train it first. Skipping.")
        return
    if out_path.exists():
        print(f"[{run_tag}] {out_path.name} already exists; skipping generation.")
    else:
        print(f"\n=== eval {run_tag} ===")
        tok_a = AutoTokenizer.from_pretrained(anchor, padding_side="left")
        if tok_a.pad_token is None: tok_a.pad_token = tok_a.eos_token

        is_adapter = (run_dir / "adapter_config.json").exists()
        if is_adapter:
            base_a  = AutoModelForCausalLM.from_pretrained(anchor, **load_kwargs_for(family_a))
            model_a = PeftModel.from_pretrained(base_a, str(run_dir))
        else:
            base_a  = None
            model_a = AutoModelForCausalLM.from_pretrained(str(run_dir), **load_kwargs_for(family_a))
        model_a.eval()
        torch.manual_seed(SEED)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

        @torch.inference_mode()
        def _gen(prompts):
            formatted = [tok_a.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=False,
                add_generation_prompt=True) for p in prompts]
            resp, finish, ntok = [], [], []
            for i in range(0, len(formatted), BATCH_GEN):
                batch  = formatted[i:i + BATCH_GEN]
                inputs = tok_a(batch, return_tensors="pt", padding=True,
                               truncation=True, max_length=1024).to(model_a.device)
                out = model_a.generate(
                    **inputs, max_new_tokens=MAX_NEW, do_sample=True,
                    temperature=1.0, top_p=0.95, pad_token_id=tok_a.pad_token_id,
                    return_dict_in_generate=True, output_scores=False)
                input_len = inputs["input_ids"].shape[1]
                for seq in out.sequences:
                    tail = seq[input_len:]
                    resp.append(tok_a.decode(tail, skip_special_tokens=True)[:1500])
                    ntok.append(int((tail != tok_a.pad_token_id).sum().item()))
                    finish.append("stop" if tok_a.eos_token_id in tail.tolist() else "length")
            return resp, finish, ntok

        judge = Judge(model="openai/gpt-5-mini", cache_namespace="paper3_eval")
        judged_a, safety_a = {}, {}
        for dim, items in eval_holdout.items():
            if not items: continue
            prompts = [it["text_ro"] for it in items]
            t0 = time.time()
            resps, finish, ntok = _gen(prompts)
            scored = []
            for j, it in enumerate(items):
                v = judge.classify(system=REFUSAL_SYSTEM,
                    user=REFUSAL_USER_TEMPLATE.format(
                        prompt=it["text_ro"], response=resps[j] or "[empty response]"))
                scored.append({"id": it["id"], "prompt": it["text_ro"],
                               "response": resps[j], "finish_reason": finish[j],
                               "completion_tokens": ntok[j],
                               "judge_label": v.get("label"),
                               "judge_justification": v.get("justification", "")})
            judged_a[dim] = scored
            n = len(scored); k = sum(1 for s in scored if s["judge_label"] == "refusal")
            safety_a[dim] = {"refusal_rate": k / max(n, 1), "n": n, "wilson_95": _wilson(k, n)}
            print(f"  {dim:>13s}  {n} prompts in {time.time()-t0:.1f}s  refusal={k}/{n}")

        out_path.write_text(json.dumps({
            "anchor_model": anchor, "condition": cond_tag, "seed": SEED,
            "adapter": str(run_dir), "safety": safety_a, "judged": judged_a,
            "saved_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }, indent=2, ensure_ascii=False))
        print(f"  saved -> {out_path}")
        del model_a
        if base_a is not None: del base_a
        torch.cuda.empty_cache(); gc.collect()


def _delta_table():
    """Print baseline delta-vs-base next to RD-DPO-k4 x4 (seed=17)."""
    rdpo = {}
    if RDPO_X4_RANKSWEEP.exists():
        d = json.loads(RDPO_X4_RANKSWEEP.read_text())
        for a, blk in d.get("deltas_pp", {}).items():
            xl = blk.get("r_16", {}).get("crosslingual")
            if xl: rdpo[a] = xl["delta_pp"]

    print("\n=== Baseline delta-vs-base (pp) vs RD-DPO-k4 x4 (seed=17) ===")
    print(f"  {'run':<44} {'tox':>7} {'jb':>7} {'or':>7} {'xl':>7}   {'RD-DPO xl':>9}")
    for anchor, cond_tag in BASELINE_RUNS:
        short_a = short_of(anchor)
        out_path = RESULTS_DIR / f"{short_a}__{cond_tag}__seed{SEED}__safety.json"
        if not out_path.exists():
            continue
        p3 = json.loads(out_path.read_text())
        p2 = _load_p2(P2_NAME[short_a])
        row = {}
        for dim in DIMS:
            p3j = p3["judged"].get(dim, [])
            ids = [j["id"] for j in p3j]
            p2d = p2.get(dim, {})
            inter = [i for i in ids if i in p2d]
            if not inter:
                row[dim] = None; continue
            base_n = sum(1 for i in inter if p2d[i] == "refusal")
            p3by = {j["id"]: j["judge_label"] for j in p3j}
            rd_n = sum(1 for i in inter if p3by[i] == "refusal")
            row[dim] = round((rd_n - base_n) / len(inter) * 100, 1)
        def f(x): return f"{x:+7.1f}" if x is not None else "    n/a"
        rd_xl = rdpo.get(short_a)
        rd_str = f"{rd_xl:+9.1f}" if rd_xl is not None else "      n/a"
        print(f"  {short_a + '  ' + cond_tag:<44} {f(row['toxicity'])} {f(row['jailbreak'])} "
              f"{f(row['overrefusal'])} {f(row['crosslingual'])}   {rd_str}")
    print("\n  Read: if baseline xl stays <= 0 like RD-DPO xl, the cross-lingual gap")
    print("  is NOT an artifact of RD-DPO's probe-layer restriction (Xoi2 sugg. 3).")


for anchor, cond_tag in BASELINE_RUNS:
    _eval_one(anchor, cond_tag)
_delta_table()
print("\nFull-DPO / all-layers-LoRA baseline eval done.")
