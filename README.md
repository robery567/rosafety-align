# Paper 3 — Refusal-Direction-Guided DPO for Romanian Safety Alignment

> **Working title:** *"Closing the Cross-Lingual Safety Gap with Refusal-
> Direction-Guided DPO and Synthetic Romanian Preferences"*
>
> **Method short name (provisional):** RD-DPO
>
> **Target venue:** EACL 2028 / NAACL 2028 (via ARR). Backup: ECAI 2028, RANLP 2027.
>
> **Status:** Plan locked; implementation in progress.

## What this is

A compute-efficient alignment recipe for small (1-7B) language models that
closes most of the Romanian cross-lingual safety gap measured in
[RoSafetyBench (Paper 2)][p2] without sacrificing English capability.

Two contributions:

1. **RD-DPO** — DPO restricted to the residual-stream blocks where the *base*
   model's refusal direction (difference-in-means between toxic and benign
   Romanian prompts) is most expressed. Selection step is a one-shot 10-GPU-
   minute probe on a frozen 200-prompt set; no annotation cost.
2. **A synthetic Romanian preference dataset** (~10-15K pairs per anchor),
   generated frontier-teacher → open-weight-student with strict generator/
   evaluatee separation. Released under CC BY 4.0.

Anchor models: Qwen-2.5-3B-Instruct, Llama-3.2-3B-Instruct, Gemma-3-4B-it.
Scaling track on Qwen-2.5 {0.5B, 1.5B, 3B, 7B}.

## Quick start

The full pipeline runs as a sequence of six Colab notebooks under
`experiments/`. Each is configured for an A100 high-RAM runtime, mounts Drive,
and reuses Paper 2's judge harness (`gpt-5-mini` primary, `claude-opus-4.5`
second-rater) so cross-paper numbers are directly comparable.

| # | Notebook                              | What it does                                         | Runtime (A100)         | Cost     |
|---|---------------------------------------|------------------------------------------------------|------------------------|----------|
| 1 | `01_refusal_probe.ipynb`              | Build refusal-direction probe; rank + cache top-k    | ~10 min (3-4B), 20m 7B | $0       |
| 2 | `02_generate_preferences.ipynb`       | Sample base completions, query teacher, emit pairs   | ~1 h + API time        | ~\$5-15  |
| 3 | `03_train_rd_dpo.ipynb`               | One (anchor, condition, seed) RD-DPO / DPO training  | 1-3 h                  | $0       |
| 4 | `04_eval_safety.ipynb`                | Generate + judge on holdout / XL / EN-HarmBench / MD | ~30-60 min             | <\$1     |
| 5 | `05_eval_capability.ipynb`            | EN lm-eval suite + RO ppl/Flores/RO-QA               | ~45-90 min             | <\$1     |
| 6 | `06_aggregate_and_figures.ipynb`      | Headline table, safety-vs-k, alignment-tax, LaTeX    | ~5 min, no GPU         | $0       |

Re-run notebook 3 → 5 once per `(anchor, condition, seed)`. Notebook 6 picks
up everything from `results/` automatically. All notebooks are idempotent —
re-running skips already-cached artefacts.

Set `HF_TOKEN` and `OPENROUTER_API_KEY` in **Colab → 🔑 → Secrets** with
notebook access enabled before running notebook 2 onwards.

`requirements.txt` lists everything the notebooks pip-install on their own;
keep it in sync if a notebook adds a new dependency.

## Project structure

```
paper3-alignment/
├── configs/
│   ├── models.yaml          # Anchor + scaling track registry
│   └── training.yaml        # DPO / LoRA / RD-DPO hyperparams
├── data/                     # operational Drive-only state (gitignored)
│   ├── preferences/<short>/      # per-anchor synthetic pairs (released to Zenodo)
│   ├── probes/<short>/            # per-anchor refusal-direction artefacts
│   ├── augmentation/<short>/      # per-anchor Stage 4 stream caches
│   ├── smoke/                    # smoke-gate audit JSONLs
│   └── splits/                   # train/dev/holdout assignment
├── src/
│   └── (small shared library helpers; sys.path-imported from notebooks
│        when reused across more than one notebook — judges shim, augmentation
│        pipeline, etc. Notebooks remain the runnable entrypoint.)
├── experiments/              # Notebooks per experimental block
│   ├── 01_refusal_probe.ipynb
│   ├── 02_generate_preferences.ipynb
│   ├── 03_train_rd_dpo.ipynb
│   ├── 04_eval_safety.ipynb
│   ├── 05_eval_capability.ipynb
│   └── 06_aggregate_and_figures.ipynb
├── results/                  # Per-run JSON outputs
├── figures/                  # Generated figures (released)
├── manuscript/               # Paper LaTeX source (private until submission)
├── logs/                     # Training + eval logs
├── PAPER3_PLAN.md           # The plan
├── METHOD_DESIGN.md         # The method spec
├── Makefile
├── requirements.txt
└── README.md
```

## Reproducibility

Following Paper 2's discipline:

- All API calls record `finish_reason`, `usage.completion_tokens`, and
  `usage.reasoning_tokens` (Paper 2 R10 lesson).
- Probe set, training configs, and judge prompts are pre-registered before
  any training run; commit hash quoted in the manuscript.
- Three random seeds per condition; mean and bootstrap-95% CI reported.
- Teacher = frontier model; evaluatees = open-weight 1-7B. No overlap.
- Same `gpt-5-mini` primary / `claude-opus-4.5` second-rater protocol as
  Paper 2 (Cohen's κ = 0.78 reference).
- Public release: dataset, code, trained adapters, probe artefacts, judge
  labels, raw response JSONs.

## Citation

```bibtex
@inproceedings{colca2028rdpo,
  title={Closing the Cross-Lingual Safety Gap with Refusal-Direction-Guided
         {DPO} and Synthetic {R}omanian Preferences},
  author={Colca, Robert-Mihai},
  booktitle={TBD},
  year={2028}
}
```

[p2]: ../paper2-benchmark/README.md
