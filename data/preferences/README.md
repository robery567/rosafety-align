# `data/preferences/`

Operational state directory. Per-anchor subdirectories
(`qwen2.5-3b/`, `llama-3.2-3b/`, `gemma-3-4b/`) live on Drive and hold:

- `smoke.json` — pre-flight gate marker (notebook 00 writes; notebook 02
  pre-flight reads).
- `stage1_rejected.jsonl` — base-model sample harvest (notebook 02
  Stage 1).
- `stage2_chosen.jsonl` — teacher refusal pairs (notebook 02 Stage 2).
- `preferences_v*.jsonl` + `.meta.json` — assembled DPO pairs.

All of the above are gitignored. Public release copies the assembled
preferences to Zenodo + HF under a dedicated v1 tag, not from this
working directory.

This file exists only so the directory survives Drive sync (which
silently drops empty `.gitkeep` files).
