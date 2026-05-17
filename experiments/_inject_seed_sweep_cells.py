"""Throwaway: append seed-sweep cells to nb03 (training) and nb04
(eval + aggregate). Mirrors the rank-sweep convention where the
notebooks contain the cells AND a standalone .py file lives in
`experiments/_seed_sweep_*.py` for paste-into-Colab usage.

Idempotent: if a markdown cell with the seed-sweep marker already
exists, the script prints a warning and exits without modifying
the notebook. Run once after writing the three _seed_sweep_*.py
helpers.

Usage:
    python3 experiments/_inject_seed_sweep_cells.py
"""

import json
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parent

NB03 = EXP / "03_train_rd_dpo.ipynb"
NB04 = EXP / "04_eval_safety.ipynb"

TRAIN_PY = EXP / "_seed_sweep_train_cell.py"
EVAL_PY  = EXP / "_seed_sweep_eval_cell.py"
AGG_PY   = EXP / "_seed_sweep_aggregate_cell.py"

NB03_MARKER = "## Seed sweep: train"
NB04_MARKER = "## Seed sweep eval"

NB03_MD = (
    "## Seed sweep: train each load-bearing condition at seed {1729, 65537}\n\n"
    "Adds 12 new training runs to convert the load-bearing dissociation cells\n"
    "(3 anchors x {e6, x4}) from single-seed to three-seed (17 + 1729 + 65537,\n"
    "the seeds pre-registered in `configs/models.yaml`). Hyperparameters are\n"
    "identical to the seed=17 baselines listed in `tables/multi-anchor-delta`\n"
    "and `tables/x4-comparison`; only the seed varies.\n\n"
    "The cell is idempotent: each (anchor, scale, seed) triple is skipped if\n"
    "its `run_meta.json` already exists. Estimated cost: ~7-9 A100-hours.\n\n"
    "Source mirror at `experiments/_seed_sweep_train_cell.py`.\n"
)

NB04_EVAL_MD = (
    "## Seed sweep eval: load each seed-sweep adapter, generate, judge\n\n"
    "Loads the 12 adapters produced by the seed-sweep training cell in nb03,\n"
    "generates on the held-out split with a per-seed-keyed decoder RNG, judges\n"
    "with gpt-5-mini, and writes per-(anchor, scale, seed) `safety.json`.\n\n"
    "Estimated cost: ~60 min generation + ~$4-8 OpenRouter.\n\n"
    "Source mirror at `experiments/_seed_sweep_eval_cell.py`.\n"
)

NB04_AGG_MD = (
    "## Seed sweep aggregate: build the multi-seed delta-vs-base JSON\n\n"
    "Combines the 12 new seed-sweep `safety.json` files plus the existing\n"
    "seed=17 baselines with Paper 2 base anchor refusal labels. Produces\n"
    "per-(anchor, scale, seed) deltas, seed-aggregated mean +/- SE per\n"
    "(anchor, scale, dim), and a cross-anchor sign-test summary on the\n"
    "x4 cross-lingual cells (the load-bearing dissociation claim).\n\n"
    "Output: `results/multi_anchor_delta_vs_base__seed-sweep.json`\n\n"
    "Source mirror at `experiments/_seed_sweep_aggregate_cell.py`.\n"
)


def _read_source(path: Path) -> list:
    """Return file contents as a list of newline-terminated lines, matching
    Jupyter's preferred `source` field representation."""
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
        print("[nb03] seed-sweep cells already present; skipping.")
        return
    nb["cells"].append(_md(NB03_MD))
    nb["cells"].append(_code(TRAIN_PY))
    _save(NB03, nb)
    print(f"[nb03] appended seed-sweep markdown + code cells -> {NB03}")


def append_nb04():
    if not EVAL_PY.exists() or not AGG_PY.exists():
        sys.exit(f"missing: {EVAL_PY} or {AGG_PY}")
    nb = json.loads(NB04.read_text())
    if _has_marker(nb, NB04_MARKER):
        print("[nb04] seed-sweep cells already present; skipping.")
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
