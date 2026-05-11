# Paper 3 — Plan

> **Working title:** *"Closing the Cross-Lingual Safety Gap with Refusal-Direction-Guided DPO and Synthetic Romanian Preferences"*
>
> **Method short name (internal, provisional):** **RD-DPO** (Refusal-Direction-guided DPO)
>
> **Phase:** 3 / Months 11-18 of the PhD plan.
> **Target venue:** EACL 2028 or NAACL 2028 via ARR. **Backup:** ECAI 2028 (WoS + Scopus) or RANLP 2027 (if the cycle hits while we're on submission).
> **Role:** First author.
> **Status:** Plan locked May 11 2026. Implementation not started.

---

## 1. The wedge

Paper 2 (RoSafetyBench, CIKM 2026) measured, on 953 culturally-native Romanian prompts, judged by `gpt-5-mini` and second-rated by `claude-opus-4.5` (κ = 0.78):

- Best open-weight Romanian cross-lingual judged refusal rate: **43.0%** (Qwen 2.5-1.5B). Family means cluster at 15-25%.
- Toxicity refusal best open-weight: **56.3%** (Gemma-3-27B). Median open-weight: **~30%**.
- Translated-from-English ablation (40 HarmBench prompts → RO via mechanical MT): models look ≥92.5% safe; on culturally-native RO they refuse 47-56% of the same intent-class prompts (gap ~+44 pp).
- The cross-lingual gap is **universal** across all 20 evaluatees (17 open-weight + 3 frontier).

That is a large, published, empirically-backed deficit on real Romanian. Paper 3's job is to close it.

## 2. Hypothesis

> A small (~5-15K) synthetic Romanian preference dataset, trained with DPO restricted to the residual-stream blocks where the model's refusal direction is most expressed (selected by a difference-in-means probe on the *base* model), closes most of the cross-lingual safety gap on small open-weight LMs (1-7B) at less than half the train-time of full-model DPO and with no measurable English capability tax.

Three falsifiable sub-claims, ordered by importance:

1. **Effect.** RD-DPO on Qwen-2.5-3B / Llama-3.2-3B / Gemma-3-4B raises judged toxicity refusal on the RoSafetyBench holdout by ≥ +25 pp absolute (target: cross 70%) and judged cross-lingual harmful refusal by ≥ +20 pp (target: cross 60%) at *seen* harm categories, with positive transfer to *unseen* categories.
2. **Efficiency.** RD-DPO with k = 4 targeted blocks matches or beats full-model DPO and all-layer LoRA-DPO on the safety axis while training in ≤ 50% of the wall-clock and ≤ 60% of the GPU-memory peak.
3. **No tax.** EN capability (MMLU 5-shot, ARC, HellaSwag) drops by ≤ 1 pp absolute averaged across the three anchors. RO capability (Flores RO-EN, perplexity on a Romanian web sample, our Romanian QA probe) drops by ≤ 1 pp.

If sub-claim 1 fails by > 5 pp on two of three anchors, the paper is not viable in this framing and we pivot (see §10).

## 3. Why this is a method paper, not an applied note

The methodological core is the **probe-guided targeted-layer recipe**, not the synthetic-preference dataset (which is enabling infrastructure). RD-DPO sits in a clean gap in the alignment-methods literature:

| Method                       | Layer selection            | Reference model? | Alignment objective | Targets multilingual safety? |
|------------------------------|----------------------------|------------------|---------------------|------------------------------|
| Vanilla DPO (Rafailov 2023)  | All layers                 | Yes              | Pairwise log-ratio  | No                           |
| LoRA-DPO (default PEFT)      | All linear modules         | Yes              | Pairwise log-ratio  | No                           |
| Safe LoRA (Hsu 2024)         | Post-hoc projection        | Yes              | Projection          | No                           |
| SaLoRA (Li 2024)             | Fixed safety adapter slots | Yes              | DPO + adapter       | No                           |
| NSPO (Lin 2024)              | Capability null-space      | Yes              | DPO in subspace     | No                           |
| ORPO (Hong 2024)             | All layers                 | No               | SFT + odds-ratio    | No                           |
| **RD-DPO (this paper)**      | **Top-k by refusal-direction probe norm** | **Yes** | **DPO** | **Yes (Romanian-first)** |

The novelty is *probe-guided* layer selection: the layers we update are picked by where the *base* model's hidden states encode refusal/non-refusal in the target language, not by a fixed architectural rule. Two intellectual lineages converge here:

- **Refusal-direction work** (Arditi et al. 2024 *Refusal in LLMs is mediated by a single direction*; Lee et al. 2024 *Mechanistic understanding of DPO*; the wider rep-eng literature). These papers identified *that* a low-dim refusal direction exists; we use it for *parameter selection* during training.
- **Targeted/selective fine-tuning** (Safe LoRA, SaLoRA, NSPO, the broader PEFT-for-safety thread). These pick layers/subspaces by safety criteria, but not by a probe of the model's own internal representation of refusal.

Probing → training intervention is the joint move that defines RD-DPO.

This also seeds **Paper 4** (mechanistic interpretability of safety): if RD-DPO works *because* the refusal direction is the right thing to push on, the same probes become Paper 4's primary lens. If RD-DPO works without correlating with probe quality, that's a Paper 4 finding too. Either way the chain is real.

## 4. Models

Three **anchor models** for the headline method results:

| Anchor       | Family | Params | RoSafetyBench v1 baseline (Tox / JB / XL) | Why included                                                                                  |
|--------------|--------|--------|-------------------------------------------|-----------------------------------------------------------------------------------------------|
| Qwen-2.5-3B-Instruct  | Qwen 2.5 | 3B   | 33.3 / 33.0 / 22.2                       | Most-studied small-model family; mid-band; clear room above 33% toxicity.                     |
| Llama-3.2-3B-Instruct | Llama   | 3B   | 16.1 / 23.7 / 16.0                       | Different family / training pipeline; weakest baseline of the three; biggest headroom.        |
| Gemma-3-4B-it         | Gemma 3 | 4B   | 34.9 / 50.5 / 21.0                       | Stronger baseline; tests whether RD-DPO can lift a *good* model further (publication risk).   |

A separate **scaling-track** sweep on Qwen-2.5 only — {0.5B, 1.5B, 3B, 7B} — answers: does RD-DPO benefit scale with model size? Mirrors Paper 2's 4-point Spearman setup (and addresses the n=4 critique pre-emptively by reporting absolute deltas, not just ρ-with-narrow-CIs).

**Generator/teacher discipline (Paper 2 R3 lesson).** The synthetic preference *teacher* must not overlap the *evaluatee* set. We use a frontier model — primary candidate `anthropic/claude-opus-4.7` via OpenRouter — for the chosen-side preferences, and the *base evaluatee* for the rejected side (its own harmful completions). Frontier-teacher → open-weight-student is standard distillation; reviewers will not flag it the way they flagged Qwen2.5-3B-as-generator-and-evaluatee.

## 5. Synthetic preference data

Reuse Paper 2's prompts as the source of harmful intents but **strict split discipline**:

- **Train (60%):** ~570 prompts. Used to elicit `(chosen, rejected)` pairs.
- **Dev (20%):** ~190 prompts. Hyperparameter selection, judge calibration.
- **Holdout (20%):** ~190 prompts. Reported numbers come from this split only.
- **Cross-lingual (86 prompts):** **eval-only**, never train. Tests cross-lingual generalization.

### 5.1 Pair construction (per anchor)

For each train prompt:

- `chosen` (harmful prompts): **Claude Opus 4.7** under the locked Romanian-aware refusal-style system prompt (METHOD §3.4). Short, polite, no moralising, redirect to a benign alternative when one exists. Frontier teacher = the alignment-signal source.
- `chosen` (benign over-refusal counter-pairs): **Llama-3.3-70B-Instruct via Together** under a plain "answer helpfully in Romanian" system prompt. Cooking, tick-bite first aid, civic information do not need frontier-tier; the chosen-side here is a capability-preservation signal, not a refusal-style signal. Mixed-teacher justified by sub-objective asymmetry, named explicitly in the paper.
- `rejected`: base evaluatee's own completion on the prompt, sampled at temperature 1 / top-p 0.95 with `n=8` resamples to find a high-likelihood compliance. If no compliance is sampled in `n=8` tries, the pair is dropped (the base model already refuses; nothing to align).

### 5.2 Composition target (per anchor)

**~800 pairs** per anchor for the headline experiments:

| Component | Pairs | Teacher | Why |
|---|---|---|---|
| Core harmful (toxicity / jailbreak / bias train) | ~400 | Claude Opus 4.7 | Refusal-style alignment signal |
| Cross-lingual harmful | ~200 | Claude Opus 4.7 | Cross-lingual transfer signal |
| Over-refusal counter-pairs (benign helpful vs apologetic refusal) | ~200 | Llama-3.3-70B | Capability preservation; **non-negotiable** — without these the over-refusal axis blows up |

This sits in the same operating point as Safe LoRA (Hsu 2024, ~800-2,000), SaLoRA (Li 2024, ~1,000), and NSPO (Lin 2024, ~600). LoRA on 4 selected blocks bounds parameter capacity, which bounds useful dataset size; we name and defend this in §6.5.

### 5.3 Translated-EN-preferences control

10K pairs from Anthropic HH-RLHF, mechanical-MT to RO via Google Translate (LLM-MT refused too often in Paper 2 R8). Train identical recipe. Tests "are RO-native preferences worth the effort vs translation". Generated **once**, shared across all three anchors (no per-anchor regeneration cost).

### 5.4 Reviewer-management plan

The most-likely reviewer comment will be "why only 800 pairs?". Defence:

- Citation row from comparable methods (Safe LoRA, NSPO, SaLoRA all 600-2,000).
- Single ablation point on Qwen-2.5-3B at 4× data (~3,200 pairs, ~$80) showing it does not improve safety on the holdout. Pre-empts the concern at a fixed extra cost; logged in `§6.5 Cost ledger`.

### 5.5 Datasheet

Following Gebru et al. 2021. Drafted in week 1, finalised before any training run, shipped with v1 release.

## 6. RD-DPO recipe

### 6.1 Refusal-direction probe (selection step)

1. Hold out 200 Romanian prompts (100 toxic intents + 100 benign), disjoint from train/dev/holdout.
2. Run the *base* model with greedy decoding, capture hidden states at the last position of the assistant prefix (before the first generated token), per residual block.
3. Compute the difference-in-means direction `d_ℓ` per block: mean toxic activations minus mean benign activations, ℓ-normalised.
4. Score each block by `‖d_ℓ‖₂ × cos(d_ℓ, projector_ℓ)` where `projector_ℓ` is the principal component of the toxic-class activations. (Robust to scale; rewards blocks where the toxic class is well-separated *and* unidirectional.)
5. Rank blocks; select top-k. **Default k = 4** (small enough to be a strong claim; large enough to give the optimiser room).

This is a one-shot pre-training analysis, ~10 GPU-minutes. Cached and shipped with the release.

### 6.2 LoRA configuration

- Adapters: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` on the **selected k blocks only**.
- Default rank: 16. Alpha: 32. Dropout: 0.05.
- All other parameters frozen.

### 6.3 DPO objective

- Standard DPO loss with reference model = base model (frozen, unaltered).
- β = 0.1 (default; ablate {0.05, 0.1, 0.2, 0.5}).
- Optimizer: AdamW, lr 5e-6, cosine decay, 1 epoch, grad accum to effective batch 32.
- Bf16, gradient checkpointing on all anchors.

### 6.4 Compute envelope (back-of-envelope on A100-40G)

| Anchor          | Vanilla DPO (full) | LoRA-DPO (all layers) | RD-DPO (k=4)        |
|-----------------|--------------------|-----------------------|---------------------|
| Qwen-2.5-3B     | ~3 h, 38 GB        | ~2 h, 22 GB           | **~1 h, 16 GB**     |
| Llama-3.2-3B    | ~3 h, 38 GB        | ~2 h, 22 GB           | **~1 h, 16 GB**     |
| Gemma-3-4B      | ~4 h, OOM-risk     | ~2.5 h, 28 GB         | **~1.3 h, 20 GB**   |

Total method-track sweep ≈ 60-80 A100-hours. Fits in 1-2 months of Colab Pro+ A100 quota or a few weeks of academic cluster time.

### 6.5 OpenRouter cost ledger

Locked May 11 2026; **revised** May 11 2026 (evening) after the Stage-1 +
Stage-2 pilot data on Qwen-2.5-3B. Re-validate at the smoke-test gate (§15)
before bulk spend.

**Empirical yield (Qwen-2.5-3B, n=110 train prompts, v3 prompt):**

  - Stage 1 yield (rejected-sample harvested): 92% overall, with
    bias 100%, jailbreak 89%, toxicity 82%.
  - Stage 2 yield (Claude refusal that survives the verification judge):
    22% overall, with bias 26%, jailbreak 38%, toxicity 12%.
  - Per-prompt determinism between repeated v3 runs: 89% (Claude Opus 4.7
    is not perfectly deterministic at temperature 0; expected behaviour
    documented as a sub-paragraph note in EXPERIMENT_LOG).
  - End-to-end yield: ~22% × 343 train prompts = ~75 core pairs/anchor.

This is significantly lower than the 70% yield assumed in the May 11
morning ledger. The shortfall traces to Paper 2's `harmful` tags being
noisier than expected on toxicity (many tagged-toxicity rows are political
tweets or garbled mistranslations that Claude correctly engages with).
Two prompt iterations confirmed the 22% number is data-driven, not
prompt-driven (v2 prompt experiment, see EXPERIMENT_LOG).

| Item | Calls | Per-call | Subtotal |
|---|---|---|---|
| Notebook 00 — pilot smoke test (50 teacher + 50 judge) | 100 | mixed | ~$0.30 |
| Notebook 02 — Qwen-2.5-3B prefs (~75 Opus harmful core + ~200 Llama-70B benign + ~2,800 stage-1 judges + ~600 verification judges) | ~3,675 | mixed | ~$5 |
| Notebook 02 — Llama-3.2-3B prefs (same composition) | ~3,675 | mixed | ~$5 |
| Notebook 02 — Gemma-3-4B prefs (same composition, abbreviated condition sweep) | ~3,675 | mixed | ~$5 |
| Stage 4 augmentation (cross-lingual ~200 + over-refusal counter ~200, all 3 anchors) | ~2,400 | mixed | ~$30 |
| Notebook 04 — safety eval (3 anchors × 5 conditions × 3 seeds × ~500 judge calls) | ~22,500 | $0.001 | ~$23 |
| Notebook 05 — RO-QA judging (45 runs × 50 calls) | ~2,250 | $0.001 | ~$3 |
| Second-rater κ audit (one-time, claude-opus-4.5 on 200 stratified) | 200 | $0.025 | ~$5 |
| Reviewer-pre-empt: Qwen-2.5-3B at 4× data ablation (one extra preference build) | ~3,200 | mixed | ~$5 |
| **Headline budget** | | | **~$81** |
| Safety pad for retries / partial regenerations / mid-run fixes | | | +$50 |
| **Recommended OpenRouter cap** | | | **$200** |

**Pair-count accounting** (per anchor, conservative):

  - Core harmful : ~75 (was 400)
  - Cross-lingual: ~200 (Stage 4)
  - Over-refusal counter: ~200 (Stage 4)
  - Total: **~475 pairs/anchor** (was 800).

This is at the bottom of the literature operating point (Safe LoRA
800-2000, NSPO ~600, SaLoRA ~1000). With LoRA-on-4-blocks bounding
parameter capacity, ~475 pairs is defensible but tight; the 4× ablation
on Qwen-2.5-3B (3,200 pairs) becomes more important for the reviewer
pre-empt rather than less, since it has to anchor "more data does not
help" with a wider gap.

Cost-control levers if we exceed budget:
1. Drop the 4× ablation (-$5) — costs us a reviewer-pre-empt argument, manageable.
2. Drop Gemma-3-4B's `dpo-full` and `safe-lora` baselines (-$8) — we already abbreviate this anchor's sweep.
3. Drop the second-rater κ audit (-$5) — would cost cross-paper κ consistency; do not pull this lever.

## 7. Baselines

Required for every anchor:

1. **Base** — no alignment (Paper 2 numbers, regenerated under matched generation params for clean comparison).
2. **SFT-only** — fine-tune on `chosen` responses; ablates the preference signal.
3. **Vanilla DPO (full model)** — gold standard DPO baseline. **Skip on Gemma-3-4B** (compute + cost trade — see §15).
4. **LoRA-DPO (all layers, rank 16)** — the standard PEFT recipe.
5. **Safe LoRA (Hsu 2024)** — post-hoc projection of a LoRA-DPO update. **Skip on Gemma-3-4B**.
6. **Translated-EN preferences (10K HH-RLHF MT'd to RO)** — controls for "RO-native vs translated". One-time generation, shared across anchors.

Stretch baselines (only if time):

7. **NSPO (Lin 2024)** — null-space projection.
8. **SaLoRA (Li 2024)** — fixed safety adapter slots.

## 8. Evaluation

### 8.1 Safety axis (RoSafetyBench holdout)

Use the **holdout 20%** + cross-lingual eval-only set, judged by `gpt-5-mini` (primary) with `claude-opus-4.5` second-rater on a 200-prompt stratified subsample (κ-anchor for cross-paper consistency with Paper 2).

Metrics per dimension:
- Toxicity, Jailbreak, Cross-lingual: judged refusal rate.
- Over-refusal: judged compliance rate on benign prompts (must stay high).
- Bias subcategory: stereotype-confirmation rate on Roma probes (specifically; this is Paper 2's worst-axis finding).

Aggregate: same Safety Score formula as Paper 2 for direct comparability.

### 8.2 Capability axis (alignment tax)

**English (standardised harnesses):**
- MMLU (5-shot)
- ARC-Easy + ARC-Challenge
- HellaSwag
- TruthfulQA-MC2 (reasonable correlate)

**Romanian:**
- Perplexity on a held-out Romanian web sample (CulturaX-RO subset).
- Flores-200 RO→EN BLEU/chrF.
- A small Romanian QA probe we generate (~100 factual questions with verified ground truth — reuse Paper 2 generation pipeline).
- *If available by the time we run:* RoMath, RoSTS, Romanian translations of MMLU. Audit availability in week 1.

Capability tax = capability_aligned − capability_base. We need it ≤ 1 pp on EN and ≤ 1 pp on RO for the no-tax claim to hold.

### 8.3 Generalisation axis

- **Held-out harm categories.** Train preferences exclude two RoSafetyBench bias subcategories (e.g., Roma + Hungarian) entirely; eval *includes* them. Tests transfer to unseen subcategories.
- **Unseen language register.** Eval on Moldovan-Romanian variant prompts (~50, generated separately). Tests dialectal generalisation.
- **English safety.** Eval on HarmBench EN. Romanian alignment must not break English safety.

### 8.4 Statistical reporting

Every headline number ships with Wilson 95% CIs (Paper 2 standard). Spearman-with-exact-distribution-p for scaling claims (Paper 2 R1 lesson). Three random seeds per condition for the headline numbers; report mean and bootstrap CI.

## 9. Ablations

Headline ablation: **k ∈ {1, 2, 4, 8, all-layers}** on Qwen-2.5-3B. The shape of the safety-vs-k curve is the central methodological figure.

Secondary ablations (Qwen-2.5-3B unless noted):
- LoRA rank ∈ {4, 8, 16, 32}.
- DPO β ∈ {0.05, 0.1, 0.2, 0.5}.
- With/without over-refusal counter-pairs.
- With/without cross-lingual augmentation.
- Probe sample size: 50, 100, 200, 500 prompts. Sensitivity of layer ranking.
- Probe metric: difference-in-means vs logistic-regression coefficient — does the choice of probe matter?

## 10. Risk register

| Risk                                                    | Likelihood | Mitigation                                                                                                  |
|---------------------------------------------------------|------------|-------------------------------------------------------------------------------------------------------------|
| RD-DPO ≈ LoRA-DPO; layer selection isn't the lever      | Medium     | Falls back to a synthetic-preference paper for Romanian (still publishable; rebrand as alignment-data paper).|
| Over-refusal climbs above the Paper-2 frontier (72% / 53%) | High      | Counter-balance pairs in §5 are mandatory, not stretch. Eval over-refusal every checkpoint, gate releases.  |
| Capability tax > 1 pp                                   | Medium     | Lower rank, lower β, fewer training steps. Document Pareto frontier honestly even if tax > 1 pp.            |
| Refusal-direction probe is unstable across models       | Low-Medium | Probe sample-size and metric ablations (§9) build the defence pre-emptively.                                |
| Frontier teacher refuses too much during data gen       | Medium     | Paper 2 R8 already saw this with `gpt-4o-mini` (37/40 refusals on translation). Smoke test §15 gates bulk spend at ≤ 10% teacher refusal; fallback to Llama-3.3-70B-Instruct with explicit safety-research framing. |
| OpenRouter spend exceeds plan                           | Low        | Smoke test (§15) lands at <$2 before bulk; cost ledger (§6.5) caps at $200 with named cost-control levers.  |
| Compute slip                                            | Low        | Single-anchor pilot (Qwen-2.5-3B only) ships in 2 weeks; full sweep in 8.                                   |
| Reviewer says "you re-used Paper-2 prompts as train"     | Certain    | Train/dev/holdout discipline §5 + cross-lingual eval-only + held-out subcategory split §8.3 are the answer; document loudly. |
| Reviewer says "only 800 pairs is too small for DPO"     | High       | Citation row from comparable methods (Safe LoRA, NSPO, SaLoRA: 600-2,000) + 4× ablation point on Qwen-2.5-3B (§5.4). |

**Pivot trigger.** If the Qwen-2.5-3B pilot (week 6) shows <10 pp toxicity refusal lift from RD-DPO over LoRA-DPO, we pivot to a different framing in week 7 — most likely a "Romanian alignment dataset and training recipe" applied paper at RANLP 2027 or the TrustNLP workshop, with the layer-selection ablation deferred to Paper 4.

## 11. Authoring discipline (Paper 2 lessons)

- **Notebooks are the runnable surface.** Paper 2 used the same pattern:
  every experiment is a Colab/A100 notebook under `experiments/`, and `src/`
  is reserved for shared helpers that genuinely need to be reused across more
  than one notebook (judges shim, generation augmentation pipeline). No CLI,
  no Makefile-as-orchestrator. The notebooks pip-install their dependencies
  inline and read/write Drive artefacts; `requirements.txt` is the source of
  truth for the env spec only.
- `manuscript/` directory **gitignored** at repo init. Public repo never sees
  draft text. Same policy as Paper 2.
- `EXPERIMENT_LOG.md` is private (gitignored). Public-facing journal is the
  README + `data/RELEASE_NOTES.md`.
- Public release: dataset (CC BY 4.0), code (Apache 2.0), trained adapters
  (with a license-compatible base-model note), judge prompts, response JSONs,
  refusal-probe artefacts. Zenodo + HF.
- `finish_reason`, `usage.completion_tokens`, `usage.reasoning_tokens` recorded
  for every API call from day one. (Paper 2 R10 lesson.)
- Bibliography: start by copying Paper 1's `references.bib` (111 entries —
  covers DPO, RLHF, LoRA, refusal-direction, Romanian) and grow as needed.
- LLM-judge: `gpt-5-mini` primary, `claude-opus-4.5` second-rater on stratified
  200-sample, Cohen's κ reported per dimension. Same protocol as Paper 2 §3.4.
  Cross-paper consistency.
- Pre-register the layer-selection probe before any training. Probe set
  committed before the first DPO run; SHA-256 quoted in the paper.

## 12. Timeline (18 weeks, mapped to PhD plan Months 11-18)

| Weeks | Block                                              | Deliverables                                                                                |
|-------|----------------------------------------------------|---------------------------------------------------------------------------------------------|
| 1-2   | Lit pass + scaffold                                | Updated bibliography on alignment methods 2026 vintage; refusal-direction tooling stood up.|
| 3-4   | Synthetic preference pipeline                      | `data/preferences_v1.jsonl` for the three anchors; teacher prompt locked; datasheet draft. |
| 5-6   | RD-DPO implementation + Qwen-2.5-3B pilot         | First end-to-end run; safety + capability numbers on the dev split. **Pivot decision point.** |
| 7-9   | Full method × baselines sweep on the three anchors | All conditions × 3 seeds; headline table; safety-vs-k figure.                              |
| 10-11 | Scaling track + ablations                          | Qwen-2.5 {0.5B, 1.5B, 3B, 7B} sweep; rank/β/probe-size ablations.                         |
| 12-13 | Capability tax suite + cross-lingual generalisation | EN + RO capability anchors; held-out subcategory + Moldovan-RO + EN-HarmBench evals.       |
| 14-16 | Writing + figures + supervisor review              | First full draft; figures regenerated; supervisor turnaround.                              |
| 17-18 | Submission                                         | ARR submission; release v1.0 to Zenodo + HF.                                              |

## 13. Open questions (to resolve in week 1)

1. **Teacher choice.** Claude Opus 4.7 vs GPT-5.5 vs an open-weight teacher (Llama-3.3-70B-Instruct served via Together). Cost, refusal-on-research, and licence terms differ. Decision affects §5 budget.
2. **Romanian capability benchmarks.** Audit availability of RoMath / RoSTS / RoBench / Romanian-MMLU. If thin, commit upfront to building a small RO-QA probe (100-200 questions) as part of the release.
3. **Probe formalism.** Difference-in-means is the simplest defensible choice. If the Arditi-style direction is more standard in the 2026 lit, use it instead. Decided in week 1 after the lit pass.
4. **Method name.** RD-DPO is descriptive but not catchy. Hold on the marketing name until first results land.

## 14. Non-goals (called out so we don't drift)

- We will **not** propose a new alignment objective. DPO + targeted layers, no new losses.
- We will **not** make Paper 4's mechanistic-interpretability claims here. The probe is used as a *parameter selector*; we don't need to argue *why* the refusal direction works at the Anthropic-circuit level. That argument is Paper 4.
- We will **not** evaluate on more than the three anchor models for the full method sweep. The scaling track on Qwen-2.5 covers within-family variation. Adding Mistral / Phi-4 / SmolLM2 doubles compute without proportional reviewer-impact gain.
- We will **not** attempt a multi-language generalisation pitch (RO + 4 European LR languages). That's tempting and slot-able as Paper 6's framing later, but doubles surface area and weakens the Romanian-headline narrative.


## 15. Fail-safety register (OpenRouter spend protection)

Six failure modes that historically waste budget, with concrete defences. Every one of these is implemented before notebook 02 sees a bulk run.

### 15.1 Locked teacher prompt + smoke test

- The teacher style prompt (METHOD §3.4) is a single string applied to every harmful-side `chosen` generation. A bad prompt on a 600-call run is ~$15 wasted.
- **Defence.** A new `experiments/00_pilot_smoke_test.ipynb` runs one example per harm subcategory through the full teacher → judge pipeline (~50 calls, ~$1.50). It displays the resulting pairs side-by-side and prints a budget estimate. Bulk Stage 2 will not start until `00_pilot_smoke_test.ipynb` has been run and a `smoke_ok=True` flag is present in `data/preferences/<short>/smoke.json`.

### 15.2 Frozen prompt strings, single source of truth

- Cache keys are `sha256(model || system || user)`. A single whitespace shift invalidates the cache and re-bills.
- **Defence.** `TEACHER_SYSTEM`, `JUDGE_SYSTEM`, and the user-message templates live as module-level constants in a small helper file. Both notebook 02 and notebook 04 import them. A "do not edit at runtime" comment is mandatory; any change requires a version bump in the cache namespace.

### 15.3 Append-and-flush on every record

- Colab kills runtimes at 12 h or on idle. Holding stage-1 results in a Python list and writing at the end loses partial progress on kernel death.
- **Defence.** Already in place — every per-pair record is appended to JSONL on Drive immediately. Hardening: explicit `f.flush(); os.fsync(f.fileno())` after every line and `PYTHONUNBUFFERED=1` set in the bootstrap cell. Resume cells at the top of Stage 1 and Stage 2 read the JSONL and skip done keys.

### 15.4 Teacher-refusal-on-research gate

- Paper 2 R8: GPT-4o-mini refused 37/40 HarmBench-style RO translations. If Claude Opus 4.7 does the same, we burn cost on `[empty]` chosen responses that fail the verification judge.
- **Defence.** The smoke test records `teacher_refusal_rate` over its 50-prompt sample. If `> 10%`, notebook 02 prints a hard-stop message and refuses to proceed. Operator must either (a) revise the teacher style prompt and re-run the smoke test, or (b) set `TEACHER = TEACHER_FALLBACK` (Llama-3.3-70B-Instruct via Together).

### 15.5 Reasoning-token overrun protection

- Paper 2 R10: GPT-5.5 Pro consumed 256 tokens of internal reasoning before any visible output, producing 71 silent empties. Claude Opus 4.7 is also a reasoning model.
- **Defence.** Teacher `max_tokens=1024` (3× actual response budget of ~300 tokens). We are billed only for actual completion tokens, so no spend increase from the higher cap. Capture `usage.reasoning_tokens` per response. Smoke test aborts if any response has `reasoning_tokens / max_tokens > 0.5`.

### 15.6 Concurrency cap on the teacher endpoint

- `Judge.classify_many(workers=8)` does not deduplicate concurrent requests. A 429 storm can produce duplicate billed calls.
- **Defence.** Concurrency capped at `workers=4` for the teacher (slow expensive endpoint); `workers=8` only for the judge endpoint. Explicit in the notebook config cell.

### 15.7 Pre-flight gate cell in notebook 02

- All of the above are implemented as a single explicit pre-flight cell at the top of notebook 02 that:
  1. Verifies `smoke.json` exists with `smoke_ok=True` and timestamp within 7 days.
  2. Prints the locked teacher prompt SHA-256.
  3. Prints the budget estimate (per-anchor and total) from the cost ledger §6.5.
  4. Prints the OpenRouter monthly cap from the OpenRouter API.
  5. Asks for an explicit `y` keystroke before bulk run begins.

The pre-flight cell costs ~30 seconds of friction per anchor and eliminates the named failure modes above.