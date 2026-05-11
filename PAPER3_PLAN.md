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

Training data construction per prompt (one harmful Romanian prompt → one preference pair):

- `chosen` = teacher (Claude Opus 4.7) response. Prompted with a Romanian-aware refusal style: short, polite, no moralizing, redirect to a benign alternative when one exists. We curate the teacher's refusal style as part of the contribution (datasheet-grade documentation).
- `rejected` = base evaluatee's own completion on the same prompt, sampled at temperature 1 with `n` resamples to find a high-likelihood compliance under the base model. If no compliance is sampled in `n=8` tries, we drop the pair (prompt is already refused; nothing to align). Documented per-model.

Augmentation expansions:
- **Cross-lingual preferences (~200 new pairs).** Generate Romanian harmful prompts paired with their EN translations; chosen is teacher RO refusal, rejected is base evaluatee EN compliance translated to RO. Tests cross-lingual transfer of alignment.
- **Over-refusal counter-pairs (~200 new pairs).** Critical: naive DPO on safety-only pairs raises over-refusal. We pair benign Romanian prompts (cooking, tick bites, language-learning) with `chosen=helpful answer` / `rejected=apologetic refusal` to keep the over-refusal rate bounded. Without this counter-balance the method will look like it just teaches the model to refuse everything.

Total dataset target: **~10-15K pairs** per anchor. Dataset will be released under CC BY 4.0 alongside the paper (anonymised at submission).

Translated-EN-preferences condition (control): take 10K pairs from Anthropic HH-RLHF, mechanical-MT to RO via Google Translate, train identical recipe. Tests "are RO-native preferences worth the effort vs translation".

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

## 7. Baselines

Required for every anchor:

1. **Base** — no alignment (Paper 2 numbers, regenerated under matched generation params for clean comparison).
2. **SFT-only** — fine-tune on `chosen` responses; ablates the preference signal.
3. **Vanilla DPO (full model)** — gold standard DPO baseline.
4. **LoRA-DPO (all layers, rank 16)** — the standard PEFT recipe.
5. **Safe LoRA (Hsu 2024)** — post-hoc projection of a LoRA-DPO update.
6. **Translated-EN preferences (10K HH-RLHF MT'd to RO)** — controls for "RO-native vs translated".

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
| Frontier teacher refuses too much during data gen       | Medium     | Paper 2 R8 already saw this with `gpt-4o-mini` (37/40 refusals on translation). Use Claude with a teacher-style system prompt; if Claude also refuses on >10% of prompts, fall back to a Llama-3.3-70B-Instruct teacher with explicit safety-research framing. |
| Compute slip                                            | Low        | Single-anchor pilot (Qwen-2.5-3B only) ships in 2 weeks; full sweep in 8.                                   |
| Reviewer says "you re-used Paper-2 prompts as train"     | Certain    | Train/dev/holdout discipline §5 + cross-lingual eval-only + held-out subcategory split §8.3 are the answer; document loudly. |

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
