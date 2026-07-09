"""Throwaway: append Qwen + Gemma rank x seed sweep cells to nb03 (training)
and nb04 (eval + aggregate). Mirrors _inject_llama_rank_seed_sweep_cells.py.

Idempotent: if a markdown cell with the marker already exists, the script
prints a warning and exits without modifying the notebook.

Usage:
    python3 experiments/_inject_qwen_gemma_rank_seed_sweep_cells.py
"""

import json
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parent

NB03 = EXP / "03_train_rd_dpo.ipynb"
NB04 = EXP / "04_eval_safety.ipynb"

TRAIN_PY = EXP / "_qwen_gemma_rank_seed_sweep_train_cell.py"
EVAL_PY  = EXP / "_qwen_gemma_rank_seed_sweep_eval_cell.py"
AGG_PY   = EXP / "_qwen_gemma_rank_seed_sweep_aggregate_cell.py"

NB03_MARKER = "## Qwen+Gemma rank-seed sweep: train"
NB04_MARKER = "## Qwen+Gemma rank-seed sweep eval"

NB03_MD = (
    "## Qwen+Gemma rank-seed sweep: train {Qwen, Gemma} at r in {64, 128} x seed in {1729, 65537}\n\n"
    "Adds 8 new training runs to convert the Qwen and Gemma rows of Table 7\n"
    "(rank-sweep) from single-seed (only seed=17) to three-seed mean +/- SE\n"
    "at r=64 and r=128. Closes the reviewer concern that the only positive\n"
    "cross-lingual cells in the rank ablation (Qwen r=128 +9.3 pp, Gemma\n"
    "r=128 +1.2 pp) rest on a single seed.\n\n"
    "Per-anchor best-LR (Qwen 2e-5, Gemma 5e-6); all other hyperparameters\n"
    "identical to the seed=17 rank-sweep baselines. r=16 is NOT retrained --\n"
    "the x4 r=16 cell is the seed sweep's `rd-dpo-k4-bal-e6-x4` condition,\n"
    "already three-seed for both anchors. The cell is idempotent: each\n"
    "(anchor, rank, seed) is skipped if its `run_meta.json` already exists.\n"
    "Estimated cost: ~5-6 A100-hours.\n\n"
    "Source mirror at `experiments/_qwen_gemma_rank_seed_sweep_train_cell.py`.\n"
)

NB04_EVAL_MD = (
    "## Qwen+Gemma rank-seed sweep eval: load each adapter, generate, judge\n\n"
    "Loads the 8 adapters produced by the Qwen+Gemma rank-seed sweep training\n"
    "cell in nb03, generates on the held-out split with a per-seed-keyed\n"
    "decoder RNG, judges with gpt-5-mini, and writes per-(anchor, rank, seed)\n"
    "`safety.json`. Estimated cost: ~45 min generation + ~$2-4 OpenRouter.\n\n"
    "Source mirror at `experiments/_qwen_gemma_rank_seed_sweep_eval_cell.py`.\n"
)

NB04_AGG_MD = (
    "## Qwen+Gemma rank-seed sweep aggregate: build the multi-seed rows\n\n"
    "Combines the 8 new safety files plus the existing seed=17 rank-sweep\n"
    "baselines and the three-seed x4 r=16 cells into\n"
    "`results/multi_anchor_delta_vs_base__rd-dpo-k4-bal-e6-x4-rank-sweep-qwen-gemma-multi-seed.json`.\n"
    "Qwen and Gemma r=16/64/128 cells become mean +/- SE across\n"
    "{17, 1729, 65537}.\n\n"
    "Source mirror at `experiments/_qwen_gemma_rank_seed_sweep_aggregate_cell.py`.\n"
)


def _read_source(path: Path) -> list:
    return path.read_text().splitlines(keepends=True)


def _md(source_text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source_text.splitlines(keepends=True),
    }


def _code(py_path: Path) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _read_source(py_path),
    }


def _has_marker(nb: dict, marker: str) -> bool:
    for cell in nb["cells"]:
        if cell["cell_type"] != "markdown":
            continue
        src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        if marker in src:
            return True
    return False


def _save(path: Path, nb: dict) -> None:
    path.write_text(json.dumps(nb, indent=1) + "\n")


def append_nb03():
    if not TRAIN_PY.exists():
        sys.exit(f"missing: {TRAIN_PY}")
    nb = json.loads(NB03.read_text())
    if _has_marker(nb, NB03_MARKER):
        print("[nb03] Qwen+Gemma rank-seed-sweep cells already present; skipping.")
        return
    nb["cells"].append(_md(NB03_MD))
    nb["cells"].append(_code(TRAIN_PY))
    _save(NB03, nb)
    print(f"[nb03] appended Qwen+Gemma rank-seed-sweep markdown + code cells -> {NB03}")


def append_nb04():
    if not EVAL_PY.exists() or not AGG_PY.exists():
        sys.exit(f"missing: {EVAL_PY} or {AGG_PY}")
    nb = json.loads(NB04.read_text())
    if _has_marker(nb, NB04_MARKER):
        print("[nb04] Qwen+Gemma rank-seed-sweep cells already present; skipping.")
        return
    nb["cells"].append(_md(NB04_EVAL_MD))
    nb["cells"].append(_code(EVAL_PY))
    nb["cells"].append(_md(NB04_AGG_MD))
    nb["cells"].append(_code(AGG_PY))
    _save(NB04, nb)
    print(f"[nb04] appended 2 markdown + 2 code cells -> {NB04}")


if __name__ == "__main__":
    append_nb03()
    append_nb04()
