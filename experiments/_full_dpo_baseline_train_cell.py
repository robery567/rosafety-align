"""
Full-DPO / all-layers-LoRA-DPO baseline training cell (Reviewer Xoi2 sugg. 3).

Xoi2 asked for "full DPO over all layers ... as an upper bound ... to validate
whether the conclusions from the probe-guided setting are robust." We add TWO
upper-bound baselines at the x4 data scale so the comparison is clean:

  1. dpo-lora-all : LoRA (r=16, alpha=32) on ALL layers -- identical to RD-DPO
     except the probe-layer restriction is removed. This is the *matched*
     control: the ONLY difference from RD-DPO-k4 is layer coverage, so it
     isolates "does restricting to 4 probe-selected layers cause the failure?"
     Same per-anchor LR as RD-DPO (Qwen/Llama 2e-5, Gemma 5e-6).

  2. dpo-full : full-model DPO (no LoRA) -- the literal upper bound Xoi2 asked
     for. Full fine-tuning uses a much smaller LR (5e-7, Zephyr-style) and more
     memory; run on Qwen + Llama (Gemma full-FT is OOM-risk on A100-40G).

Interpretation for the rebuttal:
  - If BOTH baselines also leave held-out cross-lingual flat/negative, the gap
    is NOT an artifact of RD-DPO's restricted subspace -> strengthens the paper.
  - If either lifts cross-lingual materially, it bounds RD-DPO's limitation and
    tells us capacity/coverage matters -> still sharpens the paper.

Mirrors experiments/_llama_rank_seed_sweep_train_cell.py (paths, naming,
run_meta, idempotency, rebalance).

Adapter/run naming:
  {short}__dpo-lora-all-bal-e6-x4__seed17
  {short}__dpo-full-bal-e6-x4__seed17

Estimated cost: all-layers LoRA ~30-45 min/anchor (~19-22 GB); full-model DPO
~1.5-2.5 h/anchor (~34-40 GB, A100-80G recommended). Seed 17 only for the
baseline (add seeds later only if a baseline turns out to move cross-lingual).
"""

# --- defensive: ensure prompts.py is importable when run standalone ---
import sys as _sys
from pathlib import Path as _Path
_PROMPTS_SRC = _Path("/content/drive/MyDrive/PhD/paper3-alignment/src")
if _PROMPTS_SRC.exists() and str(_PROMPTS_SRC) not in _sys.path:
    _sys.path.insert(0, str(_PROMPTS_SRC))

import gc, json, random, time
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import DPOConfig, DPOTrainer

# --- baseline config ---
SEED       = 17
DATA_COND  = "bal-e6-x4"                 # rebalanced, 6 epochs, x4 data
RDPO_LR    = {                            # matched to RD-DPO per-anchor best-LR
    "Qwen/Qwen2.5-3B-Instruct": 2e-5,
    "meta-llama/Llama-3.2-3B-Instruct": 2e-5,
    "google/gemma-3-4b-it": 5e-6,
}
FULL_FT_LR = 5e-7                         # full-model DPO LR (Zephyr-style)

# Which anchors get which baseline.
LORA_ALL_ANCHORS = ["Qwen/Qwen2.5-3B-Instruct",
                    "meta-llama/Llama-3.2-3B-Instruct",
                    "google/gemma-3-4b-it"]
FULL_ANCHORS     = ["Qwen/Qwen2.5-3B-Instruct",
                    "meta-llama/Llama-3.2-3B-Instruct"]  # Gemma full-FT: OOM-risk on 40G

BETA          = 0.1
EPOCHS        = 6
WARMUP        = 2
BATCH         = 4
GA            = 8
FULL_BATCH    = 1                          # full-FT: smaller batch for memory
FULL_GA       = 32                         # keep effective batch 32
MAX_SEQ       = 1024
LORA_DROPOUT  = 0.05
LORA_R        = 16
LORA_ALPHA    = 32
LORA_TARGETS  = ["q_proj", "k_proj", "v_proj", "o_proj",
                 "gate_proj", "up_proj", "down_proj"]

REFUSE_SOURCES = {"core_s2", "core_s2_ext", "xl"}
HELP_SOURCES   = {"overref", "overref_ext"}


def _rebalance(pairs, seed):
    refuse = [r for r in pairs if r["meta"]["source"] in REFUSE_SOURCES]
    helpp  = [r for r in pairs if r["meta"]["source"] in HELP_SOURCES]
    other  = [r for r in pairs if r["meta"]["source"] not in REFUSE_SOURCES | HELP_SOURCES]
    if other:
        raise ValueError(f"unexpected sources: {sorted({r['meta']['source'] for r in other})}")
    n_target = min(len(refuse), len(helpp))
    rng = random.Random(seed)
    refuse_kept = rng.sample(refuse, n_target) if len(refuse) > n_target else refuse
    helpp_kept  = rng.sample(helpp,  n_target) if len(helpp)  > n_target else helpp
    out = refuse_kept + helpp_kept
    rng.shuffle(out)
    return out, n_target


def _train_one(anchor, mode):
    """mode in {'lora_all', 'full'}."""
    short_a  = short_of(anchor)
    family_a = family_of(anchor)
    if mode == "lora_all":
        cond_tag = f"dpo-lora-all-{DATA_COND}"
        lr = RDPO_LR[anchor]
        batch, ga = BATCH, GA
    elif mode == "full":
        cond_tag = f"dpo-full-{DATA_COND}"
        lr = FULL_FT_LR
        batch, ga = FULL_BATCH, FULL_GA
    else:
        raise ValueError(mode)

    run_tag = f"{short_a}__{cond_tag}__seed{SEED}"
    run_dir = ADAPTERS_DIR / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "run_meta.json").exists():
        print(f"[{short_a} {mode}] run_meta.json already exists; skipping.")
        return

    print(f"\n=== {short_a} baseline mode={mode} lr={lr:g} -> {run_tag}")

    pairs_path = PREFS_DIR / short_a / "preferences_x4.jsonl"
    if not pairs_path.exists():
        raise FileNotFoundError(f"{pairs_path} missing; run nb02b x4 first")
    pairs = [json.loads(l) for l in open(pairs_path, encoding="utf-8")]
    n_loaded = len(pairs)
    pairs, n_target = _rebalance(pairs, SEED)
    src_counts = Counter(r["meta"]["source"] for r in pairs)
    print(f"  loaded {n_loaded} -> rebalanced {len(pairs)} pairs ({n_target}+{n_target})")

    tok_a = AutoTokenizer.from_pretrained(anchor, padding_side="left")
    if tok_a.pad_token is None: tok_a.pad_token = tok_a.eos_token

    def _format(r):
        prompt_chat = tok_a.apply_chat_template(
            [{"role": "user", "content": r["prompt"]}],
            tokenize=False, add_generation_prompt=True,
        )
        return {"prompt": prompt_chat, "chosen": r["chosen"], "rejected": r["rejected"]}

    ds_full  = Dataset.from_list([_format(r) for r in pairs])
    ds_split = ds_full.train_test_split(test_size=0.05, seed=SEED)
    ds_train_a, ds_eval_a = ds_split["train"], ds_split["test"]
    print(f"  train={len(ds_train_a)}  eval={len(ds_eval_a)}")

    # LoRA over ALL layers (no probe-layer regex) for lora_all; None for full.
    if mode == "lora_all":
        lora_cfg = LoraConfig(
            r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
            target_modules=LORA_TARGETS,   # list of names -> matches ALL layers
            bias="none", task_type="CAUSAL_LM",
        )
    else:
        lora_cfg = None

    set_seed(SEED)
    model_a = AutoModelForCausalLM.from_pretrained(anchor, **load_kwargs_for(family_a))
    model_a.config.use_cache = False
    if hasattr(model_a, "gradient_checkpointing_enable"):
        model_a.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )

    dpo_cfg = DPOConfig(
        output_dir=str(run_dir), num_train_epochs=EPOCHS,
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=ga,
        learning_rate=lr, lr_scheduler_type="cosine",
        warmup_steps=WARMUP,
        bf16=True, gradient_checkpointing=True,
        logging_steps=2, save_steps=250, eval_steps=100,
        seed=SEED, report_to=["none"],
        beta=BETA, loss_type="sigmoid",
        max_length=MAX_SEQ,
        precompute_ref_log_probs=True,   # frees the reference model -> full-FT fits
        max_steps=-1,
    )
    trainer_a = DPOTrainer(
        model=model_a, args=dpo_cfg,
        peft_config=lora_cfg,            # None => full-model DPO
        train_dataset=ds_train_a, eval_dataset=ds_eval_a, processing_class=tok_a,
    )

    print(f"  starting training (mode={mode})...")
    t0 = time.time()
    trainer_a.train()
    elapsed = time.time() - t0
    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"  done in {elapsed/60:.1f} min  peak={peak_mem:.2f} GB")

    trainer_a.save_model(str(run_dir))
    (run_dir / "run_meta.json").write_text(json.dumps({
        "run_tag": run_tag, "anchor": anchor, "condition": cond_tag,
        "baseline_mode": mode, "seed": SEED,
        "preference_dataset": str(pairs_path),
        "n_pairs": len(pairs), "pair_source_counts": dict(src_counts),
        "training_compute": {
            "wallclock_seconds": round(elapsed, 1),
            "peak_gpu_memory_gb": round(peak_mem, 2),
            "device": DEVICE_NAME,
        },
        "hyperparams": {
            "beta": BETA, "lr": lr, "epochs": EPOCHS, "warmup_steps": WARMUP,
            "lora": (None if mode == "full"
                     else {"r": LORA_R, "alpha": LORA_ALPHA,
                           "target_blocks": "all", "target_modules": LORA_TARGETS}),
            "max_seq_len": MAX_SEQ,
            "per_device_train_batch_size": batch,
            "gradient_accumulation_steps": ga,
        },
        "saved_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }, indent=2))
    print(f"  saved -> {run_dir}")

    del model_a, trainer_a
    torch.cuda.empty_cache(); gc.collect()


# All-layers LoRA-DPO first (cheap, matched control), then full-model DPO.
for anchor in LORA_ALL_ANCHORS:
    _train_one(anchor, "lora_all")
for anchor in FULL_ANCHORS:
    _train_one(anchor, "full")
print("\nFull-DPO / all-layers-LoRA baseline training done.")
