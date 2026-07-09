"""
Qwen + Gemma rank x seed sweep eval cell.

Evaluates the 8 adapters trained by the corresponding training cell
({Qwen, Gemma} x {r=64, r=128} x {seed=1729, seed=65537}) on the held-out
split. Saves per-(anchor, rank, seed) safety.json files.

Mirrors experiments/_llama_rank_seed_sweep_eval_cell.py, extended to loop
over the two anchors.

Estimated cost:
  - Generation: ~45 min total on A100 (8 adapters x ~5-6 min each).
  - Judging:    ~$2-4 OpenRouter (gpt-5-mini; cache hits are high because
                the prompts are identical to previous evals).
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

from llm_judge import Judge
from judges import REFUSAL_SYSTEM, REFUSAL_USER_TEMPLATE

# --- Qwen + Gemma rank x seed sweep eval config ---
ANCHORS       = ["Qwen/Qwen2.5-3B-Instruct", "google/gemma-3-4b-it"]
RANKS         = [64, 128]
NEW_SEEDS     = [1729, 65537]
LRS_BASE_COND = "rd-dpo-k4-bal-e6-x4"
LRS_BATCH_GEN = 16
LRS_MAX_NEW   = 256


def _wilson(k, n, alpha=0.05):
    if n == 0: return (0.0, 0.0)
    return tuple(round(x, 3) for x in binomtest(k, n).proportion_ci(
        confidence_level=1 - alpha, method="wilson"))


def _eval_one(anchor, lora_r, seed):
    short_a   = short_of(anchor)
    family_a  = family_of(anchor)
    cond_tag  = f"{LRS_BASE_COND}-r{lora_r}"
    run_tag   = f"{short_a}__{cond_tag}__seed{seed}"
    adapter_a = ADAPTERS_DIR / run_tag
    out_path  = RESULTS_DIR / f"{run_tag}__safety.json"

    if out_path.exists():
        print(f"[{short_a} r={lora_r} seed{seed}] {out_path.name} already exists; skipping.")
        return
    if not adapter_a.exists():
        print(f"[{short_a} r={lora_r} seed{seed}] adapter {adapter_a} not found; "
              f"run the Qwen/Gemma rank-seed-sweep training cell in nb03 first.")
        return

    print(f"\n=== {short_a} rank-seed-sweep eval r={lora_r} seed={seed} -> {out_path.name} ===")

    tok_a = AutoTokenizer.from_pretrained(anchor, padding_side="left")
    if tok_a.pad_token is None: tok_a.pad_token = tok_a.eos_token
    base_a  = AutoModelForCausalLM.from_pretrained(anchor, **load_kwargs_for(family_a))
    model_a = PeftModel.from_pretrained(base_a, str(adapter_a))
    model_a.eval()

    # Decoding RNG: re-seed so different training seeds produce reproducibly
    # different generation samples (same convention as seed-sweep eval cell).
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
        for i in range(0, len(formatted), LRS_BATCH_GEN):
            batch  = formatted[i:i + LRS_BATCH_GEN]
            inputs = tok_a(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=1024).to(model_a.device)
            out = model_a.generate(
                **inputs, max_new_tokens=LRS_MAX_NEW,
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


for seed in NEW_SEEDS:
    for anchor in ANCHORS:
        for lora_r in RANKS:
            _eval_one(anchor, lora_r, seed)
print("\nQwen + Gemma rank-seed-sweep batch eval done.")
