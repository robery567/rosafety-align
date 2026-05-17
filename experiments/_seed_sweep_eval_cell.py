"""
Seed-sweep eval cell.

Evaluates the 12 seed-sweep adapters trained by the seed-sweep
training cell (3 anchors x 2 data scales x 2 new seeds) on the
held-out split. Saves per-anchor safety.json files.

Estimated cost:
  - Generation: ~60 min total on A100 (12 adapters x ~5 min each).
  - Judging:    ~$4-8 OpenRouter (gpt-5-mini judge).
"""

# --- defensive: ensure prompts.py + llm_judge are importable ---
import sys as _sys
from pathlib import Path as _Path
_PROMPTS_SRC = _Path("/content/drive/MyDrive/PhD/paper3-alignment/src")
if _PROMPTS_SRC.exists() and str(_PROMPTS_SRC) not in _sys.path:
    _sys.path.insert(0, str(_PROMPTS_SRC))

import gc, json, time
from datetime import datetime
from collections import Counter

import torch
from peft import PeftModel
from scipy.stats import binomtest
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompts import ANCHORS
from llm_judge import Judge
from judges import REFUSAL_SYSTEM, REFUSAL_USER_TEMPLATE

# --- seed sweep eval config ---
NEW_SEEDS    = [1729, 65537]
SS_BATCH_GEN = 16
SS_MAX_NEW   = 256

# Mirror the training cell's per-anchor condition tag map
PER_ANCHOR_E6 = {
    "Qwen/Qwen2.5-3B-Instruct":            "rd-dpo-k4-bal-e6-lr2e5",
    "meta-llama/Llama-3.2-3B-Instruct":    "rd-dpo-k4-bal-e6-lr2e5",
    "google/gemma-3-4b-it":                "rd-dpo-k4-bal-e6",
}
PER_ANCHOR_X4 = {
    "Qwen/Qwen2.5-3B-Instruct":            "rd-dpo-k4-bal-e6-x4",
    "meta-llama/Llama-3.2-3B-Instruct":    "rd-dpo-k4-bal-e6-x4",
    "google/gemma-3-4b-it":                "rd-dpo-k4-bal-e6-x4",
}


def _wilson(k, n, alpha=0.05):
    if n == 0: return (0.0, 0.0)
    return tuple(round(x, 3) for x in binomtest(k, n).proportion_ci(
        confidence_level=1 - alpha, method="wilson"))


def _eval_one(anchor, data_scale, seed):
    short_a   = short_of(anchor)
    family_a  = family_of(anchor)
    cond_map  = PER_ANCHOR_E6 if data_scale == "e6" else PER_ANCHOR_X4
    cond_tag  = cond_map[anchor]
    run_tag   = f"{short_a}__{cond_tag}__seed{seed}"
    adapter_a = ADAPTERS_DIR / run_tag
    out_path  = RESULTS_DIR / f"{run_tag}__safety.json"

    if out_path.exists():
        print(f"[{anchor} {data_scale} seed{seed}] {out_path.name} already exists; skipping.")
        return
    if not adapter_a.exists():
        print(f"[{anchor} {data_scale} seed{seed}] adapter {adapter_a} not found; "
              f"run the seed-sweep training cell in nb03 first.")
        return

    print(f"\n=== seed-sweep eval {anchor} @ {data_scale} seed={seed} -> {out_path.name} ===")

    tok_a = AutoTokenizer.from_pretrained(anchor, padding_side="left")
    if tok_a.pad_token is None: tok_a.pad_token = tok_a.eos_token
    base_a  = AutoModelForCausalLM.from_pretrained(anchor, **load_kwargs_for(family_a))
    model_a = PeftModel.from_pretrained(base_a, str(adapter_a))
    model_a.eval()

    # Decoding RNG: re-seed torch with the training seed so different
    # training seeds produce reproducibly-different generation samples.
    # This is what gives the multi-seed cells genuine variance to characterise
    # rather than identical-decoded outputs from differently-trained models.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    @torch.inference_mode()
    def _generate_batch(prompts):
        formatted = [
            tok_a.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False, add_generation_prompt=True,
            ) for p in prompts
        ]
        responses, finish, ntok = [], [], []
        for i in range(0, len(formatted), SS_BATCH_GEN):
            batch  = formatted[i:i + SS_BATCH_GEN]
            inputs = tok_a(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=1024).to(model_a.device)
            out = model_a.generate(
                **inputs, max_new_tokens=SS_MAX_NEW,
                do_sample=True, temperature=1.0, top_p=0.95,
                pad_token_id=tok_a.pad_token_id,
                return_dict_in_generate=True, output_scores=False,
            )
            input_len = inputs["input_ids"].shape[1]
            for seq in out.sequences:
                tail = seq[input_len:]
                resp = tok_a.decode(tail, skip_special_tokens=True)
                responses.append(resp[:1500])
                ntok.append(int((tail != tok_a.pad_token_id).sum().item()))
                finish.append("stop" if tok_a.eos_token_id in tail.tolist() else "length")
        return responses, finish, ntok

    all_gen_a = {}
    for dim, items in eval_holdout.items():
        if not items: continue
        prompts = [it["text_ro"] for it in items]
        t0 = time.time()
        resps, finish, ntok = _generate_batch(prompts)
        all_gen_a[dim] = [
            {"id": it["id"], "prompt": it["text_ro"],
             "response": resps[j], "finish_reason": finish[j],
             "completion_tokens": ntok[j]}
            for j, it in enumerate(items)
        ]
        print(f"  {dim:>13s}  {len(items)} prompts in {time.time()-t0:.1f}s")

    judge = Judge(model="openai/gpt-5-mini", cache_namespace="paper3_eval")
    judged_a = {}
    for dim, items in all_gen_a.items():
        scored = []
        for it in items:
            v = judge.classify(
                system=REFUSAL_SYSTEM,
                user=REFUSAL_USER_TEMPLATE.format(
                    prompt=it["prompt"], response=it["response"] or "[empty response]",
                ),
            )
            scored.append({**it, "judge_label": v.get("label"),
                           "judge_justification": v.get("justification", "")})
        judged_a[dim] = scored

    safety_a = {}
    for dim, items in judged_a.items():
        n = len(items)
        k = sum(1 for it in items if it["judge_label"] == "refusal")
        safety_a[dim] = {"refusal_rate": k / max(n, 1), "n": n,
                         "wilson_95": _wilson(k, n)}

    out_path.write_text(json.dumps({
        "anchor_model": anchor, "condition": cond_tag, "seed": seed,
        "adapter": str(adapter_a),
        "safety": safety_a, "judged": judged_a,
        "saved_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }, indent=2, ensure_ascii=False))
    print(f"  saved -> {out_path}")

    del model_a, base_a
    torch.cuda.empty_cache(); gc.collect()


# Loop in same order as training cell
for seed in NEW_SEEDS:
    for data_scale in ("e6", "x4"):
        for anchor in ANCHORS:
            _eval_one(anchor, data_scale, seed)
print("\nseed-sweep batch eval done.")
