"""
Seed-sweep training cell.

Trains 12 new adapters: 3 anchors x 2 data conditions (e6, x4) x
2 new seeds (1729, 65537), at the best-LR per anchor and otherwise
identical hyperparameters to the seed=17 baselines that appear in
the multi-anchor delta and x4-comparison manuscript tables.

The point: convert the load-bearing dissociation cells from
single-seed to three-seed (17 + 1729 + 65537), so the cross-anchor
cross-lingual non-lift becomes a 9-cell sign test instead of a
3-cell one.

Reused symbols (defined in earlier cells of nb03):
  - ANCHORS (from prompts)
  - short_of, family_of, load_kwargs_for, DEVICE_NAME
  - ADAPTERS_DIR, PREFS_DIR, PROBE_DIR

Adapter naming:
  e6 conditions (mirror existing best-LR-seed17 tags):
    qwen2.5-3b__rd-dpo-k4-bal-e6-lr2e5__seed{1729,65537}
    llama-3.2-3b__rd-dpo-k4-bal-e6-lr2e5__seed{1729,65537}
    gemma-3-4b__rd-dpo-k4-bal-e6__seed{1729,65537}
  x4 conditions:
    {short}__rd-dpo-k4-bal-e6-x4__seed{1729,65537}

Estimated cost: ~7-9 A100-hours total.
"""

# --- defensive: ensure prompts.py is importable when run standalone ---
import sys as _sys
from pathlib import Path as _Path
_PROMPTS_SRC = _Path("/content/drive/MyDrive/PhD/paper3-alignment/src")
if _PROMPTS_SRC.exists() and str(_PROMPTS_SRC) not in _sys.path:
    _sys.path.insert(0, str(_PROMPTS_SRC))

import gc, json, random, re, time
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import DPOConfig, DPOTrainer

from prompts import ANCHORS

# --- seed sweep config ---
# Seeds {17, 1729, 65537} are pre-registered in configs/models.yaml.
# Seed 17 already exists; we add the other two.
NEW_SEEDS = [1729, 65537]

# Per-anchor best-LR + e6 condition tag (mirrors the seed=17 baseline
# adapters cited in tables/multi-anchor-delta and tables/x4-comparison).
PER_ANCHOR_E6 = {
    # anchor                              : (condition_tag,                   lr)
    "Qwen/Qwen2.5-3B-Instruct":            ("rd-dpo-k4-bal-e6-lr2e5",       2e-5),
    "meta-llama/Llama-3.2-3B-Instruct":    ("rd-dpo-k4-bal-e6-lr2e5",       2e-5),
    "google/gemma-3-4b-it":                ("rd-dpo-k4-bal-e6",             5e-6),
}
# x4 condition: shared tag across anchors; LR per-anchor matches PER_ANCHOR_E6.
PER_ANCHOR_X4 = {
    "Qwen/Qwen2.5-3B-Instruct":            ("rd-dpo-k4-bal-e6-x4",          2e-5),
    "meta-llama/Llama-3.2-3B-Instruct":    ("rd-dpo-k4-bal-e6-x4",          2e-5),
    "google/gemma-3-4b-it":                ("rd-dpo-k4-bal-e6-x4",          5e-6),
}

# preferences file per data scale
PREFS_FILE = {
    "e6": "preferences_v2.jsonl",
    "x4": "preferences_x4.jsonl",
}

# Hyperparameters: identical to the seed=17 baselines for these condition tags.
SS_BETA            = 0.1
SS_EPOCHS          = 6
SS_WARMUP          = 2
SS_BATCH           = 4
SS_GA              = 8
SS_MAX_SEQ         = 1024
SS_LORA_R          = 16
SS_LORA_A          = 32
SS_LORA_DROPOUT    = 0.05
SS_LORA_TARGETS    = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"]

REFUSE_SOURCES = {"core_s2", "core_s2_ext", "xl"}
HELP_SOURCES   = {"overref", "overref_ext"}


def _rebalance(pairs, seed):
    """Same rebalance routine as the seed=17 baselines: down-sample
    the larger of refuse-side / help-side to match. Seed-keyed RNG
    so different seeds see slightly different rebalance subsamples,
    which is part of the variance we are trying to characterise.
    """
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


def _train_one(anchor, data_scale, seed):
    """Train one (anchor, data_scale, seed) triple. Idempotent: skips
    runs whose run_meta.json already exists."""
    short_a   = short_of(anchor)
    family_a  = family_of(anchor)
    cond_map  = PER_ANCHOR_E6 if data_scale == "e6" else PER_ANCHOR_X4
    cond_tag, lr = cond_map[anchor]
    run_tag   = f"{short_a}__{cond_tag}__seed{seed}"
    run_dir   = ADAPTERS_DIR / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "run_meta.json").exists():
        print(f"[{anchor} {data_scale} seed{seed}] run_meta.json already exists; skipping.")
        return

    print(f"\n=== seed-sweep training {anchor} @ {data_scale} seed={seed} lr={lr:.0e}")
    print(f"    -> {run_tag}")

    # Probe layers (same selection as seed=17 baseline: top-of-net)
    sel_path = PROBE_DIR / short_a / "selected_blocks.json"
    if not sel_path.exists():
        raise FileNotFoundError(f"{sel_path} missing; run nb01 probe first")
    selected_blocks = json.loads(sel_path.read_text())["4"]

    # Load preferences for the data scale
    pairs_path = PREFS_DIR / short_a / PREFS_FILE[data_scale]
    if not pairs_path.exists():
        raise FileNotFoundError(
            f"{pairs_path} missing; run nb02 (and nb02b for x4) first"
        )
    pairs = [json.loads(l) for l in open(pairs_path, encoding="utf-8")]
    n_loaded = len(pairs)
    pairs, n_target = _rebalance(pairs, seed)
    src_counts = Counter(r["meta"]["source"] for r in pairs)
    print(f"  loaded {n_loaded} -> rebalanced {len(pairs)} pairs ({n_target}+{n_target})")
    print(f"  source distribution: {dict(src_counts)}")

    tok_a = AutoTokenizer.from_pretrained(anchor, padding_side="left")
    if tok_a.pad_token is None: tok_a.pad_token = tok_a.eos_token

    def _format(r):
        prompt_chat = tok_a.apply_chat_template(
            [{"role": "user", "content": r["prompt"]}],
            tokenize=False, add_generation_prompt=True,
        )
        return {"prompt": prompt_chat, "chosen": r["chosen"], "rejected": r["rejected"]}

    ds_full   = Dataset.from_list([_format(r) for r in pairs])
    ds_split  = ds_full.train_test_split(test_size=0.05, seed=seed)
    ds_train_a, ds_eval_a = ds_split["train"], ds_split["test"]
    print(f"  train={len(ds_train_a)}  eval={len(ds_eval_a)}")

    block_pat  = "|".join(str(b) for b in selected_blocks)
    modules_re = "|".join(SS_LORA_TARGETS)
    target_re  = rf"^.*\.layers\.({block_pat})\.(?:.*\.)?(?:{modules_re})$"
    lora_cfg = LoraConfig(
        r=SS_LORA_R, lora_alpha=SS_LORA_A, lora_dropout=SS_LORA_DROPOUT,
        target_modules=target_re, bias="none", task_type="CAUSAL_LM",
    )

    set_seed(seed)
    model_a = AutoModelForCausalLM.from_pretrained(anchor, **load_kwargs_for(family_a))
    model_a.config.use_cache = False
    if hasattr(model_a, "gradient_checkpointing_enable"):
        model_a.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )

    dpo_cfg = DPOConfig(
        output_dir=str(run_dir), num_train_epochs=SS_EPOCHS,
        per_device_train_batch_size=SS_BATCH,
        gradient_accumulation_steps=SS_GA,
        learning_rate=lr, lr_scheduler_type="cosine",
        warmup_steps=SS_WARMUP,
        bf16=True, gradient_checkpointing=True,
        logging_steps=2, save_steps=250, eval_steps=100,
        seed=seed, report_to=["none"],
        beta=SS_BETA, loss_type="sigmoid",
        max_length=SS_MAX_SEQ,
        precompute_ref_log_probs=True,
        max_steps=-1,
    )
    trainer_a = DPOTrainer(
        model=model_a, args=dpo_cfg, peft_config=lora_cfg,
        train_dataset=ds_train_a, eval_dataset=ds_eval_a, processing_class=tok_a,
    )

    print("  starting training...")
    t0 = time.time()
    trainer_a.train()
    elapsed = time.time() - t0
    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"  done in {elapsed/60:.1f} min  peak={peak_mem:.2f} GB")

    trainer_a.save_model(str(run_dir))
    (run_dir / "run_meta.json").write_text(json.dumps({
        "run_tag": run_tag, "anchor": anchor, "condition": cond_tag,
        "seed": seed, "selected_blocks": selected_blocks,
        "preference_dataset": str(pairs_path),
        "n_pairs": len(pairs),
        "pair_source_counts": dict(src_counts),
        "training_compute": {
            "wallclock_seconds": round(elapsed, 1),
            "peak_gpu_memory_gb": round(peak_mem, 2),
            "device": DEVICE_NAME,
        },
        "hyperparams": {
            "beta": SS_BETA, "lr": lr, "epochs": SS_EPOCHS,
            "warmup_steps": SS_WARMUP,
            "lora_r": SS_LORA_R, "lora_alpha": SS_LORA_A,
            "max_seq_len": SS_MAX_SEQ,
            "per_device_train_batch_size": SS_BATCH,
            "gradient_accumulation_steps": SS_GA,
        },
        "saved_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }, indent=2))
    print(f"  saved -> {run_dir}")

    del model_a, trainer_a
    torch.cuda.empty_cache(); gc.collect()


# Loop order: e6 first across all anchors (cheaper, smoke-tests pipeline),
# then x4 across all anchors. Seeds outermost so all anchors get seed=1729
# before any seed=65537 — lets you stop after one extra seed if compute
# budget tightens (still gives a 6-cell two-seed sign test).
for seed in NEW_SEEDS:
    for data_scale in ("e6", "x4"):
        for anchor in ANCHORS:
            _train_one(anchor, data_scale, seed)
print("\nseed-sweep batch training done.")
