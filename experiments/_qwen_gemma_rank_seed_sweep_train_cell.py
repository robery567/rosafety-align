"""
Qwen + Gemma rank x seed sweep training cell.

Trains 8 new adapters: {Qwen-2.5-3B, Gemma-3-4B} x {r=64, r=128} x
{seed=1729, seed=65537}. With the existing seed=17 rank-sweep adapters
(rd-dpo-k4-bal-e6-x4-r{64,128}__seed17) this gives a 3-seed average per
rank for the Qwen and Gemma rows of Table 7 (rank-sweep).

Motivation (reviewer hHpd W2 / Xoi2 W2): in the v1.1 rank sweep the only
positive cross-lingual cells -- Qwen r=128 (+9.3 pp) and Gemma r=128
(+1.2 pp) -- are single-seed (seed=17), while the contrary Llama result is
already three-seed. This cell removes that asymmetry so the capacity claim
rests on three-seed mean +/- SE for every anchor. (Note: Gemma's +1.2 pp is
a single-prompt move on n=86 and is expected to be within noise; running the
seeds lets us say so with evidence rather than assertion.)

Mirrors experiments/_llama_rank_seed_sweep_train_cell.py exactly, except:
  - two anchors (Qwen, Gemma) with per-anchor best-LR (Qwen 2e-5, Gemma 5e-6);
  - r=16 is NOT retrained here -- the x4 r=16 cell is the seed-sweep's
    `rd-dpo-k4-bal-e6-x4` condition, already three-seed for both anchors.

Adapter naming: {short}__rd-dpo-k4-bal-e6-x4-r{rank}__seed{1729,65537}

Estimated cost: ~5-6 A100-hours total (Qwen r64 ~30m / r128 ~45m;
Gemma r64 ~35m / r128 ~55m; x 2 seeds).
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
from datasets import Dataset, disable_caching
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import DPOConfig, DPOTrainer

# Keep the tiny in-memory preference datasets in RAM. On Colab the default
# on-disk datasets cache lands in a volatile /tmp dir that can be cleaned
# between write and read during DPOTrainer's precompute_ref_log_probs step,
# raising FileNotFoundError on cache-*.arrow. Disabling caching avoids that.
disable_caching()

# --- Qwen + Gemma rank x seed sweep config ---
# Per-anchor best-LR (matches Table 7 / x4-comparison seed=17 cells).
ANCHOR_LR = {
    "Qwen/Qwen2.5-3B-Instruct": 2e-5,
    "google/gemma-3-4b-it":     5e-6,
}
RANKS     = [64, 128]
NEW_SEEDS = [1729, 65537]

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


def _train_one(anchor, lora_r, seed):
    lr        = ANCHOR_LR[anchor]
    short_a   = short_of(anchor)
    family_a  = family_of(anchor)
    cond_tag  = f"{LRS_BASE_COND}-r{lora_r}"
    run_tag   = f"{short_a}__{cond_tag}__seed{seed}"
    run_dir   = ADAPTERS_DIR / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "run_meta.json").exists():
        print(f"[{short_a} r={lora_r} seed{seed}] run_meta.json already exists; skipping.")
        return

    print(f"\n=== {short_a} rank-seed-sweep training r={lora_r} seed={seed} lr={lr:g}")
    print(f"    -> {run_tag}")

    # Probe layers (top-of-net selection; same as all other runs for this anchor)
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
    model_a = AutoModelForCausalLM.from_pretrained(anchor, **load_kwargs_for(family_a))
    model_a.config.use_cache = False
    if hasattr(model_a, "gradient_checkpointing_enable"):
        model_a.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )

    dpo_cfg = DPOConfig(
        output_dir=str(run_dir), num_train_epochs=LRS_EPOCHS,
        per_device_train_batch_size=LRS_BATCH,
        gradient_accumulation_steps=LRS_GA,
        learning_rate=lr, lr_scheduler_type="cosine",
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
            "beta": LRS_BETA, "lr": lr, "epochs": LRS_EPOCHS,
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


# Loop: seeds outermost so both anchors x both ranks finish at seed=1729
# before any seed=65537, letting you stop after seed=1729 (4 runs) for a
# 2-seed mean if compute tightens.
for seed in NEW_SEEDS:
    for anchor in ANCHOR_LR:
        for lora_r in RANKS:
            _train_one(anchor, lora_r, seed)
print("\nQwen + Gemma rank-seed-sweep training done.")
