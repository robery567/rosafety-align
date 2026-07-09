# Qwen + Gemma Rank-x-Seed Sweep — Operator Runbook

Goal: make the Qwen and Gemma rows of Table 7 (rank-sweep) three-seed, so
the only positive cross-lingual cells in the rank ablation (Qwen r=128
**+9.3 pp**, Gemma r=128 **+1.2 pp**) are no longer single-seed. Addresses
reviewer hHpd (W2) and Xoi2 (W2). Mirrors the Llama rank-seed sweep exactly.

Runs: {Qwen-2.5-3B, Gemma-3-4B} x {r=64, r=128} x {seed=1729, seed=65537}
= **8 adapters**. (r=16 is reused from the seed sweep's `rd-dpo-k4-bal-e6-x4`
condition — already three-seed for both anchors.)

## Step 0 — inject the cells (once)

```bash
cd ~/phd/papers/paper3-alignment
python3 experiments/_inject_qwen_gemma_rank_seed_sweep_cells.py
```
Appends a training cell to `03_train_rd_dpo.ipynb` and an eval + aggregate
cell to `04_eval_safety.ipynb`. Idempotent. (Or paste the three
`_qwen_gemma_rank_seed_sweep_*_cell.py` files as cells manually.)

## Step 1 — Train (notebook 03, ~5-6 A100-hours)

1. Open `03_train_rd_dpo.ipynb` in Colab on an A100 runtime; run the prep
   cells, then the new last cell ("Qwen+Gemma rank-seed sweep: train ...").
2. Trains 8 adapters, seeds outermost: both anchors x both ranks at
   seed=1729 first, then seed=65537. You can stop after seed=1729 for a
   2-seed mean if compute tightens.
3. Per-run (A100-40G): Qwen r64 ~30 min / r128 ~45 min; Gemma r64 ~35 min /
   r128 ~55 min. OOM at r=128 on 40G -> use A100-80G, or set
   `LRS_BATCH=2, LRS_GA=16` in the cell (effective batch unchanged).
4. Output: `adapters/{qwen2.5-3b,gemma-3-4b}__rd-dpo-k4-bal-e6-x4-r{64,128}__seed{1729,65537}/`

## Step 2 — Eval (notebook 04, ~45 min + ~$2-4 OpenRouter)

1. Open `04_eval_safety.ipynb`; run prep cells (loads `eval_holdout`,
   `Judge`, etc.), then the "Qwen+Gemma rank-seed sweep eval" cell.
2. Idempotent: skips any run whose `safety.json` already exists.
3. Output: `results/{short}__rd-dpo-k4-bal-e6-x4-r{64,128}__seed{1729,65537}__safety.json`

## Step 3 — Aggregate (~10 s, no GPU)

1. Run the "Qwen+Gemma rank-seed sweep aggregate" cell.
2. Output:
   `results/multi_anchor_delta_vs_base__rd-dpo-k4-bal-e6-x4-rank-sweep-qwen-gemma-multi-seed.json`
3. Prints per-anchor cross-lingual delta by (rank, seed) + mean +/- SE.

## Step 4 — Sync Drive -> local

```bash
DRIVE=~/Library/CloudStorage/GoogleDrive-*/My\ Drive/PhD/paper3-alignment
LOCAL=~/phd/papers/paper3-alignment
rsync -av "$DRIVE/results/" "$LOCAL/results/" \
  --include='qwen2.5-3b__rd-dpo-k4-bal-e6-x4-r64__seed1729__safety.json' \
  --include='qwen2.5-3b__rd-dpo-k4-bal-e6-x4-r128__seed1729__safety.json' \
  --include='qwen2.5-3b__rd-dpo-k4-bal-e6-x4-r64__seed65537__safety.json' \
  --include='qwen2.5-3b__rd-dpo-k4-bal-e6-x4-r128__seed65537__safety.json' \
  --include='gemma-3-4b__rd-dpo-k4-bal-e6-x4-r64__seed1729__safety.json' \
  --include='gemma-3-4b__rd-dpo-k4-bal-e6-x4-r128__seed1729__safety.json' \
  --include='gemma-3-4b__rd-dpo-k4-bal-e6-x4-r64__seed65537__safety.json' \
  --include='gemma-3-4b__rd-dpo-k4-bal-e6-x4-r128__seed65537__safety.json' \
  --include='*rank-sweep-qwen-gemma-multi-seed.json' --exclude='*'
```

## Step 5 — Read the numbers into the rebuttal

```bash
python3 rebuttal/reanalysis_signcounts.py   # prints the 3-seed rank table
```
Then paste the Qwen/Gemma r=64/r=128 cross-lingual mean +/- SE into the
`[[INSERT ...]]` placeholders in `rebuttal/response-hHpd.md` (W2) and
`rebuttal/response-Xoi2.md` (W2).

## Decision (what the 3-seed numbers mean)

- **Qwen r=128 holds positive (mean > 0, e.g. still ~+9 pp):** capacity claim
  on Qwen is solid — report mean +/- SE, keep the "capacity is the bottleneck
  where the probe is clean" reading.
- **Qwen/Gemma regress toward 0 under seed averaging:** downgrade to
  "capacity helps on Qwen (single-seed only); Gemma's +1.2 pp was within
  single-prompt noise; no robust rank effect." Either outcome is honest and
  publishable; update Table 7 + the abstract/Section rank-sweep prose to match.
- Gemma's +1.2 pp is **1/86 prompts** — treat as noise unless the 3-seed mean
  is clearly positive with small SE.

## Estimated total cost

| Resource | Estimate |
|---|---|
| GPU compute (A100) | ~5-6 hours |
| OpenRouter spend | ~$2-4 |
| Wall-clock | ~6-7 hours (training dominates) |
| Risk | Low — same recipe as Llama sweep, more parameters |
