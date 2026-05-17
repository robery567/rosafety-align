# Seed-Sweep Ablation — Operator Runbook

Goal: address the next strongest reviewer ask ("the load-bearing
cross-lingual non-lift rests on three single-seed numbers") by
training 12 new adapters at the e6 and x4 conditions on the
pre-registered seeds `{1729, 65537}` (in addition to the existing
seed-17 baselines). Result: the dissociation gets characterised as
a 9-cell sign test on x4 cross-lingual instead of a 3-cell one,
and every (anchor, scale, dim) cell in the manuscript carries a
mean +/- SE across three seeds.

The experimental cells are **already embedded in the notebooks**:

- `03_train_rd_dpo.ipynb` (last cell: seed-sweep training)
- `04_eval_safety.ipynb` (cells: seed-sweep eval + aggregate)

You don't need to paste anything. Just open the notebooks in Colab,
run end-to-end (or jump to the new cells at the bottom), and the
run will produce the new adapters, eval files, and the aggregate
delta JSON.

Prerequisites that should already be in place:
- `data/preferences/{short}/preferences_v2.jsonl` per anchor
  (the e6 source data; produced by nb02 + nb02b).
- `data/preferences/{short}/preferences_x4.jsonl` per anchor
  (the x4 source data; produced by nb02b x4 batch).
- `adapters/{short}/selected_blocks.json` per anchor
  (top-of-net probe selection; produced by nb01).
- The seed-17 baselines (`*-e6-lr2e5__seed17` for Qwen/Llama,
  `*-e6__seed17` for Gemma, and the three `*-e6-x4__seed17`
  adapters and safety.json files) — these are already in
  `adapters/` and `results/` and provide the third seed of the
  three-seed average.
- `HF_TOKEN` and `OPENROUTER_API_KEY` set in Colab Secrets.

---

## Step 1 — Train (notebook 03, ~7-9 A100-hours)

1. Open `03_train_rd_dpo.ipynb` in Colab on an A100 runtime.
2. If the previous batches have already trained successfully, skip
   to the new last cell ("Seed sweep: train each load-bearing
   condition at seed {1729, 65537}"). Otherwise run all cells from
   the top.
3. The cell trains 12 new adapters in this order:
   - seed=1729: e6 across all three anchors, then x4 across all three anchors.
   - seed=65537: e6 across all three anchors, then x4 across all three anchors.
   This ordering lets you stop after seed=1729 if compute tightens
   and still get a 6-cell two-seed sign test, which is already a
   meaningful upgrade from the current 3-cell single-seed one.
4. Per-run wall-clock estimate (A100-40G):
   - e6 conditions (~200 pairs): ~25 min each (3 anchors x 2 seeds = ~2.5h)
   - x4 conditions (~434 pairs): ~45 min each (3 anchors x 2 seeds = ~4.5h)
   Total: ~7-9 hours.
5. Output paths:
   - Qwen/Llama e6:  `adapters/{short}__rd-dpo-k4-bal-e6-lr2e5__seed{1729,65537}/`
   - Gemma e6:        `adapters/gemma-3-4b__rd-dpo-k4-bal-e6__seed{1729,65537}/`
   - All x4:          `adapters/{short}__rd-dpo-k4-bal-e6-x4__seed{1729,65537}/`

If you hit OOM at x4 on A100-40G:
- Switch to A100-80G (Colab Pro+ "high RAM" runtime).
- OR set `SS_BATCH = 2, SS_GA = 16` in the cell (effective batch
  unchanged) and re-run.

## Step 2 — Eval (notebook 04, ~60 min generation + ~$4-8 OpenRouter)

1. Open `04_eval_safety.ipynb` in Colab.
2. Run cells from the top through the existing prep cells (loads
   `eval_holdout`, defines `Judge`, etc.). You can stop after the
   per-anchor prep cell since later cells re-evaluate already-
   trained adapters.
3. Run the seed-sweep eval cell ("Seed sweep eval: load each
   seed-sweep adapter, generate, judge"). The cell is idempotent:
   it skips runs whose `safety.json` already exists.
4. Output: `results/{short}__{cond_tag}__seed{1729|65537}__safety.json`
   for 12 new files matching the 12 new adapters trained in Step 1.

Total cost: ~60 min generation + ~$4-8 OpenRouter for ~700
prompts x 12 adapters x ~$0.0005/judge.

## Step 3 — Aggregate (~10 seconds, no GPU)

1. Stay in `04_eval_safety.ipynb`.
2. Run the seed-sweep aggregate cell ("Seed sweep aggregate: build
   the multi-seed delta-vs-base JSON").
3. Output: `results/multi_anchor_delta_vs_base__seed-sweep.json`

The cell prints a human-readable summary with three sections:
- per-(anchor, scale) seed-aggregated mean +/- SE per dim;
- per-(anchor, scale, seed) cross-lingual delta line;
- a 9-cell sign-test summary on x4 cross-lingual.

## Step 4 — Sync results back from Drive to local workspace

After all steps finish, sync the new artefacts:

```bash
DRIVE=~/Library/CloudStorage/GoogleDrive-*/My\ Drive/PhD/paper3-alignment
LOCAL=~/phd/papers/paper3-alignment

# Adapters: the 12 new seed-sweep folders
rsync -av "$DRIVE/adapters/" "$LOCAL/adapters/" \
  --include='*__seed1729/' --include='*__seed1729/*' \
  --include='*__seed65537/' --include='*__seed65537/*' \
  --exclude='*'

# Per-anchor safety.json files for the 12 seed-sweep adapters + the
# aggregate delta JSON.
rsync -av "$DRIVE/results/" "$LOCAL/results/" \
  --include='*__seed1729__safety.json' \
  --include='*__seed65537__safety.json' \
  --include='*__seed-sweep.json' \
  --exclude='*'
```

Adjust the `DRIVE` glob to match your actual Drive mount path.

## Step 5 — Decide on paper update

The cell in step 3 prints the answer. For a richer summary with
auto-decision, run `_validate_seed_sweep.py` locally:

```bash
cd ~/phd/papers/paper3-alignment
python3 _validate_seed_sweep.py
```

The validator surfaces:
- which (anchor, scale, dim) cells now have signs robust to seed
  variance (i.e., all three seeds agree on the sign);
- which cells flip sign across seeds (these are the noise-bound
  ones; the multi-anchor-direction reading already covered them);
- a 9-cell sign test on x4 cross-lingual: the load-bearing
  dissociation claim becomes "0/9 positive" or "k/9 positive"
  depending on outcome.

**Outcome A: cross-lingual non-lift holds across seeds**
(0 or 1 positive cells out of 9 on x4 cross-lingual at the +5 pp
threshold):
- Update Tables 4 + 5 (`tables/multi-anchor-delta`,
  `tables/x4-comparison`) to report mean +/- SE across three
  seeds per cell instead of single-seed values.
- Update §5 prose ("the held-out cross-lingual delta does not")
  to ground the sign-test on N=9 instead of N=3.
- Update Limitations: drop the "single seed" paragraph (it's no
  longer a limitation); replace with a one-paragraph note
  describing the seed-sweep result.
- Headline strengthens: dissociation is now multi-seed-robust,
  not directional. Reviewer concern dissolves cleanly.

**Outcome B: cross-lingual lifts on at least one (anchor, seed)
pair at x4** (3+ positive cells out of 9, or any cell exceeds
+5 pp robustly):
- The dissociation claim weakens — at least some seeds escape
  the failure mode.
- Update §5 to report the per-seed split; reframe the
  dissociation as "non-lift on average, with seed-sensitive
  exceptions on anchor X."
- Investigate which anchor + seed combinations escape; that
  becomes a follow-up section, possibly with the rank-sweep
  story modified to "rank x seed" interaction.
- Even outcome B is publishable: the dissociation becomes
  conditional rather than universal, which is a more
  interesting empirical claim than the original.

In both cases, every claim in the abstract / intro / results that
currently says "every cell is non-positive" or similar should be
re-checked against the new mean +/- SE numbers and either
strengthened (Outcome A) or softened (Outcome B).

## Estimated total cost

| Resource | Estimate |
|---|---|
| GPU compute (A100) | ~7-9 hours |
| OpenRouter spend   | ~$4-8 |
| Wall-clock time    | ~10-12 hours total (training is the bottleneck) |
| Risk of failure    | Low — same recipe, more seeds, idempotent cells |

## Operator checklist

- [ ] Open `03_train_rd_dpo.ipynb` on A100 runtime; run final cell.
- [ ] Verify all 12 `run_meta.json` files exist in
      `adapters/*__seed1729/` and `adapters/*__seed65537/`.
- [ ] Open `04_eval_safety.ipynb`; run prep cells, then the
      seed-sweep eval cell, then the seed-sweep aggregate cell.
- [ ] Verify `results/multi_anchor_delta_vs_base__seed-sweep.json`
      exists and contains all 18 (anchor, scale, seed) entries.
- [ ] Sync adapters + results back from Drive to local workspace.
- [ ] Run `python3 _validate_seed_sweep.py` to print summary +
      auto-decision.
- [ ] Update manuscript per the Outcome A or B decision tree above.
