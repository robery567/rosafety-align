"""
Llama rank x seed sweep training cell.

Trains 4 new adapters: Llama-3.2-3B x {r=64, r=128} x {seed=1729, seed=65537}.
With the existing seed=17 rank-sweep adapters
(rd-dpo-k4-bal-e6-x4-r{64,128}__seed17), this gives a 3-seed average per
rank for the Llama row of Table 7 (rank-sweep), addressing the reviewer
concern that Llama's r=128 cell is the only single-seed cell in the
load-bearing rank ablation.

Mirrors the convention of the rank-sweep + seed-sweep cells:
  - same hyperparameters as the seed=17 baselines
  - per-anchor best-LR (Llama: 2e-5)
  - LoRA alpha = 2 * rank (matches rank-sweep convention)
  - rebalance with seed-keyed RNG (different seeds see different subsamples)
  - idempotent: skips runs whose run_meta.json already exists

Adapter naming: llama-3.2-3b__rd-dpo-k4-bal-e6-x4-r{rank}__seed{1729,65537}

Estimated cost: ~3-4 A100-hours total.
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

# --- Llama rank x seed sweep config ---
ANCHOR    = "meta-llama/Llama-3.2-3B-Instruct"
RANKS     = [64, 128]
NEW_SEEDS = [1729, 65537]
LR        = 2e-5  # best-LR for Llama (matches Table 7 seed=17 cells)

LRS_BASE_COND = "rd-dpo-k4-bal-e6-x4"

# Hyperparameters: identical to the rank-sweep seed=17 baselines.
LRS_BETA            = 0.1
LRS_EPOCHS          = 6
LRS_WARMUP          = 2
LRS_BATCH           = 4
LRS_GA              = 8
LRS_MAX_SEQ         = 1024
LRS_LORA_DROPOUT    = 0.05
LRS_LORA_TARGETS    = ["q_proj", "k_proj", "v_proj", "o_proj",
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


def _train_one(lora_r, seed):
    short_a   = short_of(ANCHOR)
    family_a  = family_of(ANCHOR)
    cond_tag  = f"{LRS_BASE_COND}-r{lora_r}"
    run_tag   = f"{short_a}__{cond_tag}__seed{seed}"
    run_dir   = ADAPTERS_DIR / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "run_meta.json").exists():
        print(f"[Llama r={lora_r} seed{seed}] run_meta.json already exists; skipping.")
        return

    print(f"\n=== Llama rank-seed-sweep training r={lora_r} seed={seed}")
    print(f"    -> {run_tag}")

    # Probe layers (top-of-net selection; same as all other Llama runs)
    sel_path = PROBE_DIR / short_a / "selected_blocks.json"
    if not sel_path.exists():
        raise FileNotFoundError(f"{sel_path} missing; run nb01 probe first")
    selected_blocks = json.loads(sel_path.read_text())["4"]

    # Load x4 preferences
    pairs_path = PREFS_DIR / short_a / "preferences_x4.jsonl"
    if not pairs_path.exists():
        raise FileNotFoundError(f"{pairs_path} missing; run nb02b x4 first")
    pairs = [json.loads(l) for l in open(pairs_path, encoding="utf-8")]
    n_loaded = len(pairs)
    pairs, n_target = _rebalance(pairs, seed)
    src_counts = Counter(r["meta"]["source"] for r in pairs)
    print(f"  loaded {n_loaded} -> rebalanced {len(pairs)} pairs ({n_target}+{n_target})")
    print(f"  source distribution: {dict(src_counts)}")

    tok_a = AutoTokenizer.from_pretrained(ANCHOR, padding_side="left")
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
    modules_re = "|".join(LRS_LORA_TARGETS)
    target_re  = rf"^.*\.layers\.({block_pat})\.(?:.*\.)?(?:{modules_re})$"
    # alpha = 2 * rank to keep effective magnitude comparable with the
    # r=16 / alpha=32 baselines (matches the rank-sweep convention).
    lora_alpha = 2 * lora_r
    lora_cfg = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=LRS_LORA_DROPOUT,
        target_modules=target_re, bias="none", task_type="CAUSAL_LM",
    )

    set_seed(seed)
    model_a = AutoModelForCausalLM.from_pretrained(ANCHOR, **load_kwargs_for(family_a))
    model_a.config.use_cache = False
    if hasattr(model_a, "gradient_checkpointing_enable"):
        model_a.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )

    dpo_cfg = DPOConfig(
        output_dir=str(run_dir), num_train_epochs=LRS_EPOCHS,
        per_device_train_batch_size=LRS_BATCH,
        gradient_accumulation_steps=LRS_GA,
        learning_rate=LR, lr_scheduler_type="cosine",
        warmup_steps=LRS_WARMUP,
        bf16=True, gradient_checkpointing=True,
        logging_steps=2, save_steps=250, eval_steps=100,
        seed=seed, report_to=["none"],
        beta=LRS_BETA, loss_type="sigmoid",
        max_length=LRS_MAX_SEQ,
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
        "run_tag": run_tag, "anchor": ANCHOR, "condition": cond_tag,
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
            "beta": LRS_BETA, "lr": LR, "epochs": LRS_EPOCHS,
            "warmup_steps": LRS_WARMUP,
            "lora_r": lora_r, "lora_alpha": lora_alpha,
            "max_seq_len": LRS_MAX_SEQ,
            "per_device_train_batch_size": LRS_BATCH,
            "gradient_accumulation_steps": LRS_GA,
        },
        "saved_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }, indent=2))
    print(f"  saved -> {run_dir}")

    del model_a, trainer_a
    torch.cuda.empty_cache(); gc.collect()


# Loop: ranks outermost so all ranks finish at seed=1729 before any seed=65537,
# letting you stop after seed=1729 (2 runs) for a 2-seed mean if budget tightens.
for seed in NEW_SEEDS:
    for lora_r in RANKS:
        _train_one(lora_r, seed)
print("\nLlama rank-seed-sweep training done.")
