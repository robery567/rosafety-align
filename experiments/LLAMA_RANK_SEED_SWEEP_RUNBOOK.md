# Llama Rank-x-Seed Sweep — Operator Runbook

Goal: address the last remaining single-seed concern in the v1.2 manuscript
by running Llama-3.2-3B at r in {64, 128} on the two missing seeds {1729,
65537}. Result: Llama's row of Table 7 (rank-sweep) becomes 3-seed mean +/-
SE at r=64 and r=128 (the cells that drive the contrary-anchor reading),
while Qwen and Gemma rows stay seed=17.

Why only Llama: Qwen and Gemma already showed clean cross-lingual lifts at
r=128 in the v1.1 single-seed rank sweep (+9.3 pp Qwen, +1.2 pp Gemma).
Llama is the one anchor where capacity-as-bottleneck failed; the reviewer
question is whether that result is robust or a single-seed artefact.

The experimental cells are **already embedded in the notebooks**:
- `03_train_rd_dpo.ipynb` (last cell: Llama rank-seed sweep training)
- `04_eval_safety.ipynb` (last 2 code cells: eval + aggregate)

You don't need to paste anything. Just open the notebooks in Colab, run
the new last cells, and the run will produce the new adapters, eval
files, and the aggregate JSON.

Prerequisites that should already be in place:
- `data/preferences/llama-3.2-3b/preferences_x4.jsonl`
  (produced by nb02b x4 batch; already done).
- `adapters/llama-3.2-3b/selected_blocks.json`
  (produced by nb01; already done).
- The seed=17 rank-sweep adapters for Llama (r=64, r=128) and their
  safety files. If they are not present, the aggregator will print
  "missing" warnings for those cells but still produce a 2-seed average.
- `HF_TOKEN` and `OPENROUTER_API_KEY` set in Colab Secrets.

---

## Step 1 — Train (notebook 03, ~3-4 A100-hours)

1. Open `03_train_rd_dpo.ipynb` in Colab on an A100 runtime.
2. Skip to the new last cell ("Llama rank-seed sweep: train Llama at r in
   {64, 128} x seed in {1729, 65537}"). All earlier cells are idempotent
   and can be re-run safely if needed.
3. The cell trains 4 new adapters in this order:
   - seed=1729: r=64, then r=128
   - seed=65537: r=64, then r=128
   The seeds-outermost order lets you stop after seed=1729 (2 runs) for
   a 2-seed mean if compute tightens.
4. Per-run wall-clock estimate (A100-40G):
   - r=64 on Llama:  ~30 min
   - r=128 on Llama: ~45 min
   Total: ~3-4 hours.
5. Output paths:
   `adapters/llama-3.2-3b__rd-dpo-k4-bal-e6-x4-r{64,128}__seed{1729,65537}/`

If you hit OOM at r=128 on A100-40G:
- Switch to A100-80G (Colab Pro+ "high RAM" runtime).
- OR set `LRS_BATCH = 2, LRS_GA = 16` in the cell (effective batch
  unchanged) and re-run.

## Step 2 — Eval (notebook 04, ~25 min generation + ~$1-2 OpenRouter)

1. Open `04_eval_safety.ipynb` in Colab.
2. Run cells from the top through the existing prep cells (loads
   `eval_holdout`, defines `Judge`, etc.). You can stop after cell 6.
3. Run the second-to-last code cell ("Llama rank-seed sweep eval"). The
   cell is idempotent: it skips runs whose `safety.json` already exists.
4. Output: `results/llama-3.2-3b__rd-dpo-k4-bal-e6-x4-r{64,128}__seed{1729,65537}__safety.json`
   (4 new files matching the 4 new adapters trained in Step 1).

Total cost: ~25 min generation + ~$1-2 OpenRouter (judge cache hits are
high since the same 262 prompts have been judged in previous evals).

## Step 3 — Aggregate (~10 seconds, no GPU)

1. Stay in `04_eval_safety.ipynb`.
2. Run the last code cell ("Llama rank-seed sweep aggregate").
3. Output:
   `results/multi_anchor_delta_vs_base__rd-dpo-k4-bal-e6-x4-rank-sweep-llama-multi-seed.json`

The cell prints two summary blocks to stdout:
- Llama cross-lingual deltas at r in {16, 64, 128} per seed plus the
  mean +/- SE across the three seeds at r=64 and r=128;
- Llama r=128 deltas across all four refusal dims (mean +/- SE).

## Step 4 — Sync results back from Drive to local workspace

```bash
DRIVE=~/Library/CloudStorage/GoogleDrive-*/My\ Drive/PhD/paper3-alignment
LOCAL=~/phd/papers/paper3-alignment

# 4 new safety files + the aggregate
rsync -av "$DRIVE/results/" "$LOCAL/results/" \
  --include='llama-3.2-3b__rd-dpo-k4-bal-e6-x4-r64__seed1729__safety.json' \
  --include='llama-3.2-3b__rd-dpo-k4-bal-e6-x4-r128__seed1729__safety.json' \
  --include='llama-3.2-3b__rd-dpo-k4-bal-e6-x4-r64__seed65537__safety.json' \
  --include='llama-3.2-3b__rd-dpo-k4-bal-e6-x4-r128__seed65537__safety.json' \
  --include='*rank-sweep-llama-multi-seed.json' \
  --exclude='*'
```

Adjust the `DRIVE` glob to match your actual Drive mount path.

## Step 5 — Decide on paper update

Run `_validate_llama_rank_seed_sweep.py` locally for the auto-decision:

```bash
cd ~/phd/papers/paper3-alignment
python3 _validate_llama_rank_seed_sweep.py
```

The validator surfaces:
- Llama r=128 cross-lingual mean +/- SE across the three seeds;
- whether the multi-seed result confirms or revises the v1.1 reading
  ("capacity is not the bottleneck on Llama");
- a side-by-side comparison of the v1.1 single-seed cell (-9.3 pp) with
  the v1.2 multi-seed mean.

**Outcome A: Llama r=128 stays in the [-7, 0] pp band across seeds.**
The probe-quality bound on Llama hardens: capacity is not the bottleneck
on the contrary anchor at any rank we tested, robustly across seeds.
Update Table 7 to mean +/- SE on the Llama row; tighten the §rank-sweep
"capacity hypothesis fails on Llama" prose.

**Outcome B: Llama r=128 multi-seed mean is materially less negative
than the v1.1 single-seed -9.3 pp** (e.g., -5 pp or better, or any cell
crosses zero):
- The v1.1 single-seed Llama cell was an unlucky draw, not a stable
  bound. Update Table 7 to mean +/- SE; rewrite the "capacity hypothesis
  fails on Llama" subsection to "capacity is the bottleneck on all three
  anchors at r=128, with Llama showing higher seed variance."
- The two-part bound in §Conclusion / §Abstract weakens to a one-part
  bound: capacity matters for cross-lingual transfer.
- Same word-count, stronger result.

In both cases, every claim in the abstract / intro / §rank-sweep that
currently says "Llama is the contrary anchor" should be re-checked
against the new mean +/- SE numbers.

## Estimated total cost

| Resource | Estimate |
|---|---|
| GPU compute (A100) | ~3-4 hours |
| OpenRouter spend   | ~$1-2 |
| Wall-clock time    | ~5 hours total |
| Risk of failure    | Very low — same recipe, same operator pattern as today |

## Operator checklist

- [ ] Open `03_train_rd_dpo.ipynb` on A100 runtime; run the last cell.
- [ ] Verify all 4 `run_meta.json` files exist in
      `adapters/llama-3.2-3b__*-r{64,128}__seed{1729,65537}/`.
- [ ] Open `04_eval_safety.ipynb`; run prep cells, then the last 2 code
      cells.
- [ ] Verify
      `results/multi_anchor_delta_vs_base__rd-dpo-k4-bal-e6-x4-rank-sweep-llama-multi-seed.json`
      exists.
- [ ] Sync adapters + results back from Drive to local workspace.
- [ ] Run `python3 _validate_llama_rank_seed_sweep.py` to print summary
      + auto-decision.
- [ ] Update manuscript Table 7 + §rank-sweep prose per the decision tree
      above.
