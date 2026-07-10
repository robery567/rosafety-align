"""Append the full-DPO / all-layers-LoRA baseline cells to nb03 (train) and
nb04 (eval + aggregate). Format-safe: preserves each notebook's exact JSON
serialization (indent / ensure_ascii / trailing newline) so the diff is
additive only. Idempotent via markdown markers.

Usage:
    python3 experiments/_inject_full_dpo_baseline_cells.py
"""

import json
from pathlib import Path

EXP = Path(__file__).resolve().parent
NB03 = EXP / "03_train_rd_dpo.ipynb"
NB04 = EXP / "04_eval_safety.ipynb"
TRAIN_PY = EXP / "_full_dpo_baseline_train_cell.py"
EVAL_PY  = EXP / "_full_dpo_baseline_eval_cell.py"
AGG_PY   = EXP / "_full_dpo_baseline_aggregate_cell.py"

NB03_MARKER = "## Full-DPO / all-layers-LoRA baseline: train"
NB04_MARKER = "## Full-DPO / all-layers-LoRA baseline: eval"

NB03_MD = (
    "## Full-DPO / all-layers-LoRA baseline: train (Xoi2 suggestion 3)\n\n"
    "Upper-bound baselines at the x4 scale. all-layers LoRA (r=16 on every\n"
    "layer, matched to RD-DPO except for the probe-layer restriction) runs\n"
    "across the three pre-registered seeds {17, 1729, 65537} on all three\n"
    "anchors; full-model DPO (seed 17, Qwen + Llama) is the heavier literal\n"
    "upper bound. Answers whether the flat cross-lingual result is caused by\n"
    "restricting training to the 4 probe layers. Idempotent per (anchor,\n"
    "mode, seed). Source: experiments/_full_dpo_baseline_train_cell.py.\n"
)
NB04_EVAL_MD = (
    "## Full-DPO / all-layers-LoRA baseline: eval\n\n"
    "Evaluates the baseline runs on the held-out split and writes per-run\n"
    "safety.json (handles both PEFT-adapter and full-model saves).\n"
    "Source: experiments/_full_dpo_baseline_eval_cell.py.\n"
)
NB04_AGG_MD = (
    "## Full-DPO / all-layers-LoRA baseline: aggregate\n\n"
    "Seed-aggregates the all-layers-LoRA control (three seeds), computes\n"
    "delta-vs-base, and prints cross-lingual next to RD-DPO-k4 (x4, 3-seed)\n"
    "and full-model DPO. Writes\n"
    "results/multi_anchor_delta_vs_base__dpo-baselines.json.\n"
    "Source: experiments/_full_dpo_baseline_aggregate_cell.py.\n"
)


def detect_fmt(raw, nb):
    for indent in (1, 2, 4):
        for ea in (True, False):
            for tnl in (True, False):
                if json.dumps(nb, indent=indent, ensure_ascii=ea) + ("\n" if tnl else "") == raw:
                    return indent, ea, tnl
    return 1, True, True


def _md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def _code(py_path):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": py_path.read_text().splitlines(keepends=True)}


def _has_marker(nb, marker):
    return any(marker in "".join(c.get("source", []))
               for c in nb["cells"] if c["cell_type"] == "markdown")


def _load(path):
    raw = path.read_text(encoding="utf-8")
    return raw, json.loads(raw)


def _save(path, nb, fmt):
    indent, ea, tnl = fmt
    path.write_text(json.dumps(nb, indent=indent, ensure_ascii=ea) + ("\n" if tnl else ""),
                    encoding="utf-8")


def main():
    raw3, nb3 = _load(NB03)
    fmt3 = detect_fmt(raw3, json.loads(raw3))   # detect BEFORE modifying nb3
    if _has_marker(nb3, NB03_MARKER):
        print("[nb03] full-DPO baseline cell already present; skipping.")
    else:
        nb3["cells"] += [_md(NB03_MD), _code(TRAIN_PY)]
        _save(NB03, nb3, fmt3)
        print(f"[nb03] appended full-DPO baseline train cell (fmt={fmt3}).")

    raw4, nb4 = _load(NB04)
    fmt4 = detect_fmt(raw4, json.loads(raw4))   # detect BEFORE modifying nb4
    if _has_marker(nb4, NB04_MARKER):
        print("[nb04] full-DPO baseline cells already present; skipping.")
    else:
        nb4["cells"] += [_md(NB04_EVAL_MD), _code(EVAL_PY), _md(NB04_AGG_MD), _code(AGG_PY)]
        _save(NB04, nb4, fmt4)
        print(f"[nb04] appended full-DPO baseline eval + aggregate cells (fmt={fmt4}).")

    for p, m in [(NB03, NB03_MARKER), (NB04, NB04_MARKER)]:
        nb = json.loads(p.read_text(encoding="utf-8"))
        assert _has_marker(nb, m), f"marker missing in {p}"
    print("OK: both notebooks reparse and contain the full-DPO baseline cells.")


if __name__ == "__main__":
    main()
