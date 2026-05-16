# LoRA-Rank Ablation — Operator Runbook

Goal: address the strongest reviewer ask ("does LoRA $r{=}16$
actually limit cross-lingual transfer?") by training six new
adapters at the x4 expanded data scale at LoRA $r \in \{64, 128\}$
and comparing held-out cross-lingual delta against the existing
$r{=}16$ baseline.

The experimental cells are **already embedded in the notebooks**:

- `03_train_rd_dpo.ipynb` (last cell: rank-sweep training)
- `04_eval_safety.ipynb` (cell 22: rank-sweep eval; cell 24: aggregation)

You don't need to paste anything. Just open the notebooks in Colab,
run end-to-end (or jump to the new cells), and the run will produce
the new adapters, eval files, and the aggregate delta JSON.

Prerequisites that should already be in place:
- `data/preferences/{short}/preferences_x4.jsonl` per anchor
  (produced by nb02 x4 batch; you've already run this).
- `adapters/{short}/selected_blocks.json` per anchor
  (produced by nb01).
- `HF_TOKEN` and `OPENROUTER_API_KEY` set in Colab Secrets.

---

## Step 1 — Train (notebook 03, ~5 A100-hours)

1. Open `03_train_rd_dpo.ipynb` in Colab on an A100 runtime.
2. Run all cells from top to the final cell, OR if the previous
   x4 batch has already trained successfully, just run the final
   cell ("Rank sweep: train Qwen / Llama / Gemma at LoRA r=64 and
   r=128 on x4 data"). The rank-sweep cell is idempotent: each
   anchor/rank pair is skipped if its `run_meta.json` already
   exists.
3. The cell trains six new adapters in this order: r=64 across
   all three anchors, then r=128 across all three anchors.
4. Per-run wall-clock estimate (A100-40G):
   - r=64 on Qwen / Llama: ~30 min each
   - r=64 on Gemma: ~35 min
   - r=128 on Qwen / Llama: ~45 min each
   - r=128 on Gemma: ~55 min
   Total: ~5 hours.
5. Output paths:
   `adapters/{short}__rd-dpo-k4-bal-e6-x4-r{rank}__seed17/`

If you hit OOM at r=128 on A100-40G:
- Switch to A100-80G (Colab Pro+ "high RAM" runtime).
- OR set `RS_BATCH = 2, RS_GA = 16` in the cell (effective batch
  unchanged) and re-run.

## Step 2 — Eval (notebook 04, ~30 min generation + ~$2-5 OpenRouter)

1. Open `04_eval_safety.ipynb` in Colab.
2. Run cells from the top through the existing prep cells (loads
   `eval_holdout`, defines `Judge`, etc.). You can stop after
   cell 14 since later cells re-evaluate already-trained adapters.
3. Run cell 22 ("Rank sweep eval: load each rank-sweep adapter,
   generate, judge"). The cell is idempotent: it skips runs whose
   `safety.json` already exists.
4. Output: `results/{short}__rd-dpo-k4-bal-e6-x4-r{rank}__seed17__safety.json`

Total cost: ~30 min generation + ~$2-5 OpenRouter for ~700
prompts × 6 adapters × ~$0.0005/judge.

## Step 3 — Aggregate (~10 seconds, no GPU)

1. Stay in `04_eval_safety.ipynb`.
2. Run cell 24 ("Rank sweep aggregate: build the multi-anchor
   delta-vs-base JSON").
3. Output: `results/multi_anchor_delta_vs_base__rd-dpo-k4-bal-e6-x4-rank-sweep.json`

The cell also prints a human-readable summary table to stdout.

## Step 4 — Sync results back from Drive to local workspace

After all steps finish, sync the new artefacts:

```bash
DRIVE=~/Library/CloudStorage/GoogleDrive-*/My\ Drive/PhD/paper3-alignment
LOCAL=~/phd/papers/paper3-alignment

# Adapters (the 6 new rank-sweep folders)
rsync -av "$DRIVE/adapters/" "$LOCAL/adapters/" \
  --include='*-r64*/' --include='*-r128*/' --include='*-r64*/*' \
  --include='*-r128*/*' --exclude='*'

# Per-anchor safety.json files for the 6 rank-sweep adapters + the
# aggregate delta JSON.
rsync -av "$DRIVE/results/" "$LOCAL/results/" \
  --include='*-r64*safety.json' --include='*-r128*safety.json' \
  --include='*rank-sweep*' --exclude='*'
```

Adjust the `DRIVE` glob to match your actual Drive mount path.

## Step 5 — Decide on paper update

The cell in step 3 prints a table that tells you the answer.
If you want a more verbose summary with auto-decision, run
`_validate_rank_sweep.py` locally:

```bash
cd ~/phd/papers/paper3-alignment
python3 _validate_rank_sweep.py
```

**If the rank sweep DOES lift cross-lingual** (any anchor moves
from $-X$ pp at r=16 to $+Y$ pp or $-Z<X$ pp at r=128):
- The discriminator-capacity claim is now empirically demonstrated.
- Update §why-it-fails to add "rank is the bottleneck" as a seventh
  ruled-in cause.
- Update abstract: replace "remaining most plausible hypothesis"
  with "ablation confirms the LoRA $r{=}16$ bottleneck; $r{=}64$
  moves cross-lingual delta from $-X.X$ to $+Y.Y$ pp on Qwen, etc."
- Add new table `tables/rank-sweep.tex` mirroring the lr-sweep
  table format.

**If the rank sweep DOES NOT lift cross-lingual** (deltas at
r=64/128 stay in the $[-7, 0]$ pp band like r=16):
- The discriminator-capacity claim is **falsified** — even better.
- Update §Limitations: remove "Larger LoRA capacity" as a candidate;
  add a new paragraph reporting the negative rank-sweep result.
- Update abstract: replace softer wording with "We further test
  larger LoRA ranks (64, 128) and confirm the cross-lingual
  bottleneck is not in the trainable subspace; the residual
  candidates are prompt-distribution gap and base-model size."
- Even cleaner negative result; eliminates one more candidate.

In both cases, add a new appendix subsection reporting the
rank-sweep numbers. Either result is publishable and improves
the paper's standing.

## Estimated total cost

| Resource | Estimate |
|---|---|
| GPU compute (A100) | ~5 hours |
| OpenRouter spend | ~$2-5 |
| Wall-clock time | ~6 hours total (training is the bottleneck) |
| Risk of failure | Low — same recipe, more parameters |

## Operator checklist

- [ ] Open `03_train_rd_dpo.ipynb` on A100 runtime; run final cell.
- [ ] Verify all 6 `run_meta.json` files exist in
      `adapters/*-r64-*` and `adapters/*-r128-*` folders.
- [ ] Open `04_eval_safety.ipynb`; run prep cells through cell 14;
      then run cells 22 and 24.
- [ ] Verify `results/multi_anchor_delta_vs_base__rd-dpo-k4-bal-e6-x4-rank-sweep.json`
      exists and contains all anchor × rank cells.
- [ ] Sync adapters + results back from Drive to local workspace.
- [ ] Run `python3 _validate_rank_sweep.py` to print summary.
- [ ] Update manuscript with new section + table based on outcome.
