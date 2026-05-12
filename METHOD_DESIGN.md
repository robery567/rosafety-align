# RD-DPO — Method Design Spec

> Companion to `PAPER3_PLAN.md`. The plan defines the contribution; this file
> defines what we actually implement. Keep it precise — anything ambiguous
> here will surface as a reviewer question or a re-run.

---

## 1. Notation

- `M` — base model (e.g., Qwen-2.5-3B-Instruct). Frozen reference.
- `M_θ` — same architecture, with trainable LoRA adapters parameterised by `θ`.
- `B` — number of transformer blocks in `M`.
- `h_ℓ(x)` — residual-stream activation at the **last position of the assistant prefix** (i.e., immediately before the first generated token), at the output of block `ℓ ∈ {0, …, B−1}`, given prompt `x`.
- `D_probe` — held-out 200-prompt set of Romanian (toxic, benign) pairs, disjoint from train/dev/holdout. Same set is used across all models for cross-model comparability.
- `D_train`, `D_dev`, `D_holdout` — 60/20/20 split of the union of RoSafetyBench prompts that admit a meaningful preference pair (i.e., where the base model produces at least one harmful compliance under sampling). See §3.
- `T` — frontier teacher model (default: `anthropic/claude-opus-4.7` via OpenRouter).

## 2. Refusal-direction probe (selection step)

### 2.1 Construction

Let `D_probe = {(x_i, y_i)}` with `y_i ∈ {tox, ben}`, `|D_probe| = 200`, `100` per class.

For each block `ℓ`:

```
μ_tox(ℓ) = mean_{i: y_i = tox} h_ℓ(x_i)
μ_ben(ℓ) = mean_{i: y_i = ben} h_ℓ(x_i)
d_ℓ      = (μ_tox(ℓ) − μ_ben(ℓ)) / ‖μ_tox(ℓ) − μ_ben(ℓ)‖₂
```

`d_ℓ` is the **block-`ℓ` refusal direction**. It is computed once per base model and cached.

### 2.2 Layer ranking

Each block gets a scalar score `s_ℓ`:

```
s_ℓ = ‖μ_tox(ℓ) − μ_ben(ℓ)‖₂ × |cos(d_ℓ, v_ℓ)|
```

where `v_ℓ` is the first principal component of the centred toxic-class activations at block `ℓ`. The first factor measures *separation* (how far apart the two class means are at this block); the second factor measures *unidirectionality* (how much the toxic-class variance aligns with the difference-in-means direction). We want both: a block that separates the classes *and* does so along a single dominant direction.

Blocks are ranked by `s_ℓ` descending; the top-k are selected. **Default k = 4.**

### 2.3 Sanity checks (must all pass before §3 runs)

1. `s_ℓ` distribution must have a clear peak — top-k should be ≥ 2× the bottom-k mean. If flat, the probe is uninformative for this model and we either re-run with a larger `D_probe` or fall back to all-layer LoRA for that anchor (and call it out as a methodological limitation).
2. The selected blocks must be in the **mid-to-late** part of the network (typically blocks `B/2`-ish through `B−2`). Refusal directions in early blocks (< `B/4`) are usually input-feature artefacts and should be excluded post-hoc; if the top-k contains an early block we replace it with the next-ranked late block and log the swap. Symmetrically, refusal directions in the **final 2 blocks** (`≥ B−2`) are the unembedding-adjacent region rather than an internal circuit; we drop those too. The two-sided cutoff was added after the May 12 three-anchor probe sweep where Qwen-2.5-3B and Llama-3.2-3B both peaked in the last 10% of layers; with the cutoff in place all three anchors land in the 0.71-0.94 depth band.
3. Cross-model consistency: the selected layer indices for Qwen-2.5-3B at temperature 0 vs temperature 1 sampling of `D_probe` activations must match in ≥ k−1 of k positions. If not, the probe is too noisy and `D_probe` is doubled.

### 2.4 Caching

Probe artefacts are written to `data/probes/<model_short>/`:
- `block_directions.pt` — `(B, d_model)` tensor of `d_ℓ`.
- `block_scores.json` — per-block `s_ℓ`, separation, unidirectionality.
- `selected_blocks.json` — `{k: [block_indices]}` for `k ∈ {1, 2, 4, 8}`.
- `probe_set.jsonl` — the exact 200-prompt `D_probe` (frozen).
- `meta.json` — model id, commit hash, generation params, `D_probe` SHA-256.

These artefacts ship with the public release.

## 3. Synthetic preference data

### 3.1 Source

RoSafetyBench prompts (Paper 2). Splits:

| Split    | Fraction | Use                                     |
|----------|----------|-----------------------------------------|
| Train    | 60%      | Preference-pair generation only.        |
| Dev      | 20%      | Hyperparameter selection, judge calibration, early stopping. |
| Holdout  | 20%      | Reported numbers come from this split only. |

Cross-lingual subset (86 prompts) is **eval-only** across all splits. Held-out subcategory split (Roma + Hungarian bias probes — ~30 prompts each) is excluded from train and dev, eval-only.

### 3.2 Pair construction (per anchor model)

For each train prompt `x`:

1. **Sample base completion.** Generate `n = 8` completions from the base model `M` at temperature 1, top-p 0.95, max 256 tokens. Score each with the same `gpt-5-mini` refusal judge used in Paper 2.
2. **Pick rejected.** First completion judged `compliance` is `r`. If none, drop the prompt — there is no "rejected" signal to align against.
3. **Generate chosen.** Two cases:
   - Harmful prompt → query `anthropic/claude-opus-4.7` (frontier teacher) under the locked Romanian-aware refusal-style system prompt (§3.4 harmful-side). Sample at temperature 0 for determinism. Verify with the judge that the response is `refusal`. If `compliance`, drop the pair (rare; teacher pipeline failure).
   - Benign prompt (over-refusal counter-pair) → query `meta-llama/Llama-3.3-70B-Instruct` via Together under the benign-side system prompt (§3.4 benign-side). Sample at temperature 0. Verify with the judge that the response is `compliance` (the helpful answer should *not* be a refusal).
4. **Emit pair.** `(x, chosen, rejected)` with full metadata: model versions, judge labels, sampling params, finish_reason, completion + reasoning token counts, timestamp.

Yield rate (estimated): ~70% of harmful train prompts yield a pair; ~95% of benign counter-prompts yield a pair.

### 3.3 Composition target (~800 pairs per anchor)

| Component | Pairs | Source intents | Teacher |
|---|---|---|---|
| Core harmful | ~400 | Toxicity / jailbreak / bias train splits | Claude Opus 4.7 |
| Cross-lingual harmful | ~200 | RO harmful prompts paired with EN translations; chosen = teacher RO refusal, rejected = base evaluatee EN compliance translated to RO via mechanical MT | Claude Opus 4.7 |
| Over-refusal counter | ~200 | Benign RO prompts (cooking, language-learning, first-aid, civic information, regional history); chosen = helpful answer, rejected = apologetic refusal | Llama-3.3-70B-Instruct |

**Mixed-teacher justification.** Harmful-side `chosen` carries the refusal-style alignment signal — frontier-tier needed. Benign-side `chosen` carries a capability-preservation signal — any decent open-weight teacher is sufficient. This asymmetry is named explicitly in the manuscript and in PAPER3_PLAN §5.1.

This sits in the same operating point as Safe LoRA (Hsu 2024, ~800-2,000), SaLoRA (Li 2024, ~1,000), and NSPO (Lin 2024, ~600). LoRA on 4 selected blocks bounds parameter capacity, which bounds useful dataset size; we defend this choice with a 4× ablation point on Qwen-2.5-3B (PAPER3_PLAN §5.4).

### 3.4 Teacher style prompts (locked at week 1)

Harmful-side teacher (Claude Opus 4.7):

```
You are a helpful Romanian-speaking assistant. When the user asks for
something harmful, illegal, or unsafe:

  - Refuse politely and concisely (1-3 sentences).
  - Do not moralise or lecture.
  - When a benign alternative exists, redirect to it (one sentence).
  - Always respond in fluent Romanian, matching the user's register.
  - Never invent legal advice, medical advice, or facts about Romanian
    persons or institutions.

When the user asks for something benign, answer normally and helpfully
in Romanian.
```

Benign-side teacher (Llama-3.3-70B-Instruct):

```
You are a helpful Romanian-speaking assistant. Answer the user's question
helpfully, factually, and concisely in fluent Romanian. Do not refuse or
hedge unless the question is genuinely harmful.
```

Both wordings are locked as module-level constants (PAPER3_PLAN §15.2), version-controlled, and quoted in the manuscript appendix. Changes after data generation begins require regenerating the affected pairs and a cache-namespace bump.

### 3.5 Translated-EN-preference control

Take 10K pairs from Anthropic HH-RLHF; pass `prompt`, `chosen`, `rejected` through Google Translate (mechanical, not LLM-MT — Paper 2 R8 lesson: LLM translators refuse on harmful content). Train identical recipe. **One-time generation, shared across all three anchors.**

This is the central "are RO-native preferences worth the effort" comparison. If RD-DPO with translated preferences ≈ RD-DPO with native preferences, the data-pipeline contribution shrinks and the paper has to lean entirely on the layer-selection contribution. If native >> translated, both contributions stand.

### 3.6 Datasheet

Following Gebru et al. 2021. Required sections: motivation, composition, collection, preprocessing, uses, distribution, maintenance, ethical considerations. Drafted in week 1, finalised before any training run, shipped with v1 release.

## 4. Training recipe

### 4.1 LoRA configuration

For each block `ℓ` in the selected set (size k):

- LoRA on `q_proj`, `k_proj`, `v_proj`, `o_proj` (attention).
- LoRA on `gate_proj`, `up_proj`, `down_proj` (MLP).
- Default rank: 16. Alpha: 32. Dropout: 0.05.
- All other parameters frozen (including LayerNorm).

`peft.LoraConfig` with `target_modules` resolved per architecture (`q_proj` etc. on Llama/Qwen; the Gemma-3 attention naming is slightly different — verify in week 1).

### 4.2 DPO objective

Standard DPO (Rafailov et al. 2023). Reference model = base `M` (frozen, full precision in inference mode). Effective objective:

```
L_DPO(θ) = −E_{(x, y_w, y_l) ~ D} [
  log σ(β · (log π_θ(y_w|x) − log π_M(y_w|x)
            − log π_θ(y_l|x) + log π_M(y_l|x)))
]
```

Hyperparameters:
- β = 0.1 (default; ablate {0.05, 0.1, 0.2, 0.5}).
- LR = 5e-6 with cosine decay, 100-step warmup.
- 1 epoch over the preference set.
- Max seq len 1024.
- Effective batch 32 via `per_device_train_batch_size = 4` × `gradient_accumulation_steps = 8`.
- bf16 mixed precision; gradient checkpointing on.
- 3 random seeds per condition; report mean ± bootstrap-95% CI.

### 4.3 Hyperparameter selection

- Layer count `k`: ablate on Qwen-2.5-3B over `{1, 2, 4, 8, all-layers}`. Default carried forward = best-on-dev.
- Other hyperparameters: select on Qwen-2.5-3B dev set, freeze, apply to all anchors. **No per-anchor tuning** of headline hyperparameters — that would be a data-leakage path. Per-anchor tuning is allowed only on `lr` and is documented if used.

### 4.4 Implementation

- TRL `DPOTrainer` (latest as of week 1) + `peft` LoRA.
- Custom callback: log per-step DPO accuracy, reward margin, per-class log-prob gap.
- Custom callback: every 100 steps, run a 50-prompt safety eval on dev (toxicity + over-refusal) to catch the over-refusal blowup early.
- Save checkpoint every 250 steps; final checkpoint = best dev Safety Score (Paper 2 formula).
- Training run logged to W&B under project `paper3-alignment` (private during embargo, public at release).

## 5. Evaluation harness

### 5.1 Reuse from Paper 2

- `papers/paper2-benchmark/src/judges.py` — judge-prompt definitions per dimension. Imported as a dependency, not copy-pasted.
- `papers/paper2-benchmark/src/llm_judge.py` — OpenRouter client + cache. Same.
- `gpt-5-mini` primary judge; `claude-opus-4.5` second-rater on a stratified 200-sample. Same as Paper 2 §3.4.

### 5.2 Harness (lives in the notebooks; helpers in `src/` only when shared)

Each long-running stage is a Colab notebook under `experiments/`:

- `experiments/01_refusal_probe.ipynb` — builds and caches the refusal-direction
  probe per §2. Idempotent.
- `experiments/02_generate_preferences.ipynb` — runs the two-stage preference
  pipeline per §3.
- `experiments/03_train_rd_dpo.ipynb` — one (anchor, condition, seed) training
  run.
- `experiments/04_eval_safety.ipynb` — generates + judges on the eval splits.
  Captures `finish_reason` and token usage on every call (Paper 2 R10 lesson:
  we never want to mis-diagnose an artefact for a behaviour again).
- `experiments/05_eval_capability.ipynb` — EN lm-eval-harness suite and
  Romanian capability suite (perplexity on a CulturaX-RO holdout; Flores-200
  RO-EN BLEU/chrF; the small RO-QA probe scored as accuracy with a separate
  judge).
- `experiments/06_aggregate_and_figures.ipynb` — collects per-run JSONs from
  `results/`, builds the headline table, the safety-vs-k figure, the
  alignment-tax scatter, and the scaling plot. Emits the LaTeX snippet for the
  manuscript.

`src/` is reserved for shared helpers that get used in more than one notebook
(judges shim, augmentation pipeline). No standalone CLIs, no Makefile
orchestration — Paper 2 demonstrated that notebooks-as-entrypoint is enough.

### 5.3 Evaluation protocol (locked)

For each `(anchor_model, condition, seed)`:

1. Generate responses on the holdout split (toxicity 53, jailbreak 20, over-refusal 20, bias 43, hallucination 39, cross-lingual 17 — 192 prompts) at temperature 1, top-p 0.95, 256 max-new-tokens. Capture `finish_reason` and token usage.
2. Generate responses on the held-out subcategory split (~60 prompts) at the same params.
3. Generate responses on the EN-HarmBench split (510 standard prompts) at the same params.
4. Generate responses on the Moldovan-RO probe (~50 prompts) at the same params.
5. Score everything with `gpt-5-mini`. Cache hit-rate target ≥ 90% by Paper 2's judge cache.
6. Run the EN capability suite via `lm-eval-harness`.
7. Run the RO capability suite per `src/eval_capability_ro.py`.
8. Aggregate into `results/<anchor>_<condition>_<seed>.json` with the schema below.
9. Cross-condition aggregation by `src/aggregate.py` produces `results/summary.json` and the LaTeX table for the manuscript.

### 5.4 Result schema (locked v1)

```json
{
  "anchor_model": "Qwen/Qwen2.5-3B-Instruct",
  "condition": "rd-dpo-k4",
  "seed": 17,
  "training_run_id": "wandb://paper3-alignment/abc123",
  "preference_dataset": "preferences_v1.jsonl@sha256:...",
  "selected_blocks": [22, 25, 27, 30],
  "training_compute": {
    "wallclock_seconds": 3812,
    "peak_gpu_memory_gb": 16.4,
    "device": "A100-40GB"
  },
  "safety": {
    "toxicity":    {"refusal_rate": 0.78, "wilson_95": [0.65, 0.87], "n": 53},
    "jailbreak":   {"refusal_rate": 0.85, "wilson_95": [0.64, 0.95], "n": 20},
    "overrefusal": {"refusal_rate": 0.10, "wilson_95": [0.02, 0.30], "n": 20},
    "bias":        {"stereotype_confirmation_rate": 0.07, "wilson_95": [...], "n": 43},
    "crosslingual":{"refusal_rate": 0.71, "wilson_95": [0.46, 0.88], "n": 17},
    "safety_score": 0.62
  },
  "capability_en": {"mmlu": 0.642, "arc_easy": 0.781, "arc_challenge": 0.464, "hellaswag": 0.733, "truthfulqa_mc2": 0.398},
  "capability_ro": {"culturax_ppl": 8.42, "flores_ro_en_chrf": 0.612, "ro_qa_accuracy": 0.51},
  "generalisation": {
    "harmbench_en_refusal_rate": 0.97,
    "moldovan_ro_refusal_rate": 0.74,
    "held_out_subcategory_refusal_rate": 0.66
  }
}
```

## 6. Public release

Following Paper 2's discipline:

- **Zenodo** — preferences (CC BY 4.0), trained adapters (Apache 2.0 with base-model licence note per anchor), probe artefacts, response JSONs, judge labels, training logs.
- **Hugging Face** — `rosafety-align/preferences-v1` dataset; per-anchor adapter repos; reproducibility scripts.
- **GitHub** — code, harness, configs, datasheet. `manuscript/` excluded.

Anonymised at submission. Public IDs at camera-ready.

## 7. Threats to validity (bake in early, don't add at submission)

1. **Generator overlap.** Teacher = Claude Opus 4.7 (frontier); evaluatees = open-weight 1-7B. No overlap. **Resolved.**
2. **Judge confounding.** Same `gpt-5-mini` is used during data generation (filtering rejected/chosen pairs) and at evaluation. Mitigation: report κ vs `claude-opus-4.5` second-rater on a 200-prompt stratified eval sample; second-rater is *not* used during data gen.
3. **Train/test contamination.** RoSafetyBench prompts appear in both Paper 2 and Paper 3 train. Mitigation: holdout 20% never seen during training; cross-lingual + held-out subcategory + EN-HarmBench + Moldovan-RO are all train-disjoint by construction.
4. **Capability tax measurement is RO-thin.** Mitigation: ship a small RO-QA probe with the release; report perplexity + Flores as redundant signals; commit to RoMath/RoSTS coverage in v2 if the benchmarks become available.
5. **Probe instability.** Mitigation: probe sample-size and probe-metric ablations (PAPER3_PLAN §9).
6. **Single-judge fragility.** Mitigation: same dual-judge protocol as Paper 2; cross-paper consistency.

## 8. Pre-registration

Before the first DPO run we commit, with git timestamps:

- `data/probes/probe_set.jsonl` (the 200-prompt `D_probe`).
- `configs/training.yaml` (locked hyperparams).
- This document with no further substantive edits to §2-§4.

The pre-registration discipline matches Paper 2's approach to the keyword-vs-judge question: lock the protocol, then run, then write the paper around what fell out.
