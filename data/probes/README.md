# `data/probes/`

Operational state directory. Holds the frozen probe set
(`probe_set.jsonl`, 200 Romanian prompts: 100 toxic + 100 benign) and
per-anchor subdirectories with refusal-direction probe artefacts:

- `hidden_states.pt` — captured residual-stream activations at the
  last assistant-prefix position, per layer.
- `block_directions.pt` — difference-in-means refusal direction per
  layer, normalised.
- `block_scores.json` — per-layer `(separation, unidirectionality,
  score)`.
- `selected_blocks.json` — top-k selections for `k ∈ {1, 2, 4, 8}`.
- `meta.json` — anchor id, n_blocks, d_model, sanity-check ratio,
  build timestamp.
- `score_curve.png` — per-block score bar plot.

All of the above are gitignored. Public release will copy
`selected_blocks.json` and `meta.json` into the Zenodo deposit so the
manuscript's reproducibility appendix can cite the exact layer indices;
the bulky `hidden_states.pt` stays Drive-only.

This file exists only so the directory survives Drive sync (which
silently drops empty `.gitkeep` files).
