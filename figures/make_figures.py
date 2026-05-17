"""Regenerate Paper 3 figures from results/ + adapters/.

Run from the repo root:
    python3 figures/make_figures.py

Outputs (white background, PDF + PNG, mirrored into manuscript/figures/):
    figures/fig1-trajectories.pdf      training trajectories on 7 trained adapters
    figures/fig2-multi-anchor-delta.pdf delta vs Paper 2 base, by (anchor, LR, dim)
    figures/fig3-toxicity-vs-lr.pdf     monotonic toxicity-degradation curve

Style mirrors Paper 2's regen_figures.py for visual consistency across the trilogy.
"""

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIGS_REPO = ROOT / "figures"
FIGS_MS   = ROOT / "manuscript" / "figures"
RES_DIR   = ROOT / "results"
ADP_DIR   = ROOT / "adapters"
P2_JUDGED = ROOT.parent / "paper2-benchmark" / "experiments" / "judged"

mpl.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "savefig.facecolor": "white",
    "savefig.edgecolor": "white",
    "font.size": 9,
})

# ---------------------------------------------------------------------------
# Inventory of trained adapters and their tags. The convention is:
#   <short>__<condition>__seed17
#
# We pull Gemma e=6 from rd-dpo-k4-bal-e6 (LR=5e-6, the only LR that
# produced training on Gemma -- mid-depth was identical and degraded LR
# sweeps were not run for Gemma).
# Qwen and Llama come from the LR sweep.
# ---------------------------------------------------------------------------
TRAINED_RUNS = [
    # (anchor_short, lr_label, condition_string, lr_value)
    ("gemma-3-4b",   "5e-6",  "rd-dpo-k4-bal-e6",      5e-6),
    ("qwen2.5-3b",   "1e-5",  "rd-dpo-k4-bal-e6-lr1e5", 1e-5),
    ("qwen2.5-3b",   "2e-5",  "rd-dpo-k4-bal-e6-lr2e5", 2e-5),
    ("qwen2.5-3b",   "5e-5",  "rd-dpo-k4-bal-e6-lr5e5", 5e-5),
    ("llama-3.2-3b", "1e-5",  "rd-dpo-k4-bal-e6-lr1e5", 1e-5),
    ("llama-3.2-3b", "2e-5",  "rd-dpo-k4-bal-e6-lr2e5", 2e-5),
    ("llama-3.2-3b", "5e-5",  "rd-dpo-k4-bal-e6-lr5e5", 5e-5),
]

ANCHOR_DISPLAY = {
    "qwen2.5-3b":   "Qwen 2.5 3B",
    "llama-3.2-3b": "Llama 3.2 3B",
    "gemma-3-4b":   "Gemma 3 4B",
}
ANCHOR_COLOR = {
    "qwen2.5-3b":   "#1f77b4",  # blue
    "llama-3.2-3b": "#ff7f0e",  # orange
    "gemma-3-4b":   "#2ca02c",  # green
}
LR_LINESTYLE = {
    "5e-6": ":",
    "1e-5": "-.",
    "2e-5": "--",
    "5e-5": "-",
}

P2_TAG = {  # paper2-benchmark file naming
    "qwen2.5-3b":   "qwen2.5-3b",
    "llama-3.2-3b": "llama3.2-3b",
    "gemma-3-4b":   "gemma3-4b",
}


def _load_trajectory(short, condition):
    """Read trainer_state.json for the given run and return a dict with
    step, epoch, loss, margins, accuracy lists."""
    base = ADP_DIR / f"{short}__{condition}__seed17"
    ts = next(base.glob("checkpoint-*/trainer_state.json"), None)
    if ts is None:
        ts = base / "trainer_state.json"
    if not ts.exists():
        return None
    log = json.loads(ts.read_text()).get("log_history", [])
    log = [e for e in log if "loss" in e]
    return {
        "step":    [e.get("step", 0)             for e in log],
        "epoch":   [e.get("epoch", 0)            for e in log],
        "loss":    [e.get("loss", float("nan"))  for e in log],
        "margins": [e.get("rewards/margins", float("nan")) for e in log],
        "acc":     [e.get("rewards/accuracies", float("nan")) for e in log],
    }


# ---------------------------------------------------------------------------
# Figure 1: convergence trajectories
# Three subplots (loss / margins / accuracy), all anchors and LRs overlaid.
# ---------------------------------------------------------------------------

def fig_trajectories():
    runs = []
    for short, lr_label, condition, lr_val in TRAINED_RUNS:
        traj = _load_trajectory(short, condition)
        if traj is None:
            print(f"  missing trajectory for {short} {condition}; skipping")
            continue
        runs.append((short, lr_label, traj))

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.4), sharex=True)

    for ax, (key, ylabel, title) in zip(
        axes,
        [("loss",    "DPO loss",                "Loss"),
         ("margins", "rewards/margins",         "Margins"),
         ("acc",     "rewards/accuracies",      "Accuracy")],
    ):
        for short, lr_label, traj in runs:
            ax.plot(
                traj["step"], traj[key],
                color=ANCHOR_COLOR[short],
                linestyle=LR_LINESTYLE[lr_label],
                marker="o", markersize=2.5, linewidth=1.2,
                label=f"{ANCHOR_DISPLAY[short]} (LR={lr_label})",
            )
        ax.set_xlabel("training step")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)

    # baselines on the loss panel
    axes[0].axhline(0.6931, color="gray", linestyle=":", linewidth=0.9)
    axes[0].text(0.5, 0.6931, " random ($-\\log 0.5$)",
                 va="bottom", ha="left", fontsize=8, color="gray")

    # baselines on the accuracy panel
    axes[2].axhline(0.5, color="gray", linestyle=":", linewidth=0.9)
    axes[2].axhline(1.0, color="gray", linestyle=":", linewidth=0.9)

    # Single shared legend below the panels
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.05),
               ncol=4, fontsize=8, frameon=False)
    fig.suptitle("Training trajectories: in-distribution convergence on 200 preference pairs",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0.06, 1, 0.97])

    for out_dir in (FIGS_REPO, FIGS_MS):
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / "fig1-trajectories.pdf", bbox_inches="tight")
        fig.savefig(out_dir / "fig1-trajectories.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"wrote fig1-trajectories ({len(runs)} runs)")


# ---------------------------------------------------------------------------
# Figure 2: multi-anchor delta vs base, four held-out dims
# Grouped bars per dim, hue by anchor, overlay LR variants for Qwen + Llama
# (Gemma has just one trained run -- LR=5e-6).
# ---------------------------------------------------------------------------

def _delta_for(short, condition):
    """Return dict dim -> {n, base, adp, delta} computed from intersecting
    Paper 3 holdout IDs with Paper 2 per-id refusal judgments."""
    p2 = json.loads((P2_JUDGED / f"refusal_{P2_TAG[short]}.json").read_text())
    p2_by = {(j["dim"], j["id"]): j["judge_label"] for j in p2["judgments"]}
    p3_path = RES_DIR / f"{short}__{condition}__seed17__safety.json"
    if not p3_path.exists():
        return None
    p3 = json.loads(p3_path.read_text())
    out = {}
    for dim in ("toxicity", "jailbreak", "overrefusal", "crosslingual"):
        items = p3.get("judged", {}).get(dim, [])
        labels = {x["id"]: x["judge_label"] for x in items}
        ids = [i for i in labels if (dim, i) in p2_by]
        if not ids:
            out[dim] = None; continue
        n = len(ids)
        base_r = sum(1 for i in ids if p2_by[(dim, i)] == "refusal") / n
        adp_r  = sum(1 for i in ids if labels[i] == "refusal") / n
        out[dim] = {"n": n, "base": base_r, "adp": adp_r, "delta": adp_r - base_r}
    return out


def fig_multi_anchor_delta():
    # Per-anchor representative trained run: pick the LR that gave the
    # highest in-distribution accuracy without dropping below random
    # baseline. Qwen+Llama: best is lr=2e-5 (clean convergence, not
    # over-fit). Gemma: only trained run is lr=5e-6.
    rep_runs = [
        ("qwen2.5-3b",   "2e-5", "rd-dpo-k4-bal-e6-lr2e5"),
        ("llama-3.2-3b", "2e-5", "rd-dpo-k4-bal-e6-lr2e5"),
        ("gemma-3-4b",   "5e-6", "rd-dpo-k4-bal-e6"),
    ]

    dims = ["toxicity", "jailbreak", "overrefusal", "crosslingual"]
    dim_labels = ["Toxicity", "Jailbreak", "Over-refusal", "Cross-lingual"]

    deltas = {}  # short -> {dim -> delta_pp}
    bases  = {}  # short -> {dim -> base_rate}
    ns     = {}  # short -> {dim -> n}
    for short, lr_label, condition in rep_runs:
        d = _delta_for(short, condition)
        if d is None:
            print(f"  missing delta for {short}; skipping")
            continue
        deltas[short] = {dim: d[dim]["delta"] * 100 for dim in dims if d[dim]}
        bases[short]  = {dim: d[dim]["base"]   * 100 for dim in dims if d[dim]}
        ns[short]     = {dim: d[dim]["n"]            for dim in dims if d[dim]}

    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    x = np.arange(len(dims))
    width = 0.25
    offsets = {"qwen2.5-3b": -width, "llama-3.2-3b": 0.0, "gemma-3-4b": +width}

    for short, off in offsets.items():
        ys = [deltas.get(short, {}).get(dim, 0.0) for dim in dims]
        ax.bar(
            x + off, ys, width,
            color=ANCHOR_COLOR[short],
            label=ANCHOR_DISPLAY[short],
            edgecolor="black", linewidth=0.6,
        )
        # annotate the delta in pp
        for xi, y in zip(x + off, ys):
            va = "bottom" if y >= 0 else "top"
            ax.text(xi, y + (0.6 if y >= 0 else -0.6),
                    f"{y:+.1f}", ha="center", va=va, fontsize=7.5)

    ax.axhline(0, color="black", linewidth=0.7)
    # Wilson-noise band guide on n=20 (jailbreak / over-refusal); roughly +/- 22pp at p=0.5
    # We show only a vertical reference around dims with small n.
    ax.set_xticks(x)
    ax.set_xticklabels(dim_labels)
    ax.set_ylabel("Δ refusal rate (rd-dpo − base, pp)")
    ax.set_ylim(-30, 30)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)

    # n annotations under x-axis
    for xi, dim in zip(x, dims):
        n_for_dim = next(
            (ns[s].get(dim) for s in ns if dim in ns[s]), None)
        if n_for_dim is not None:
            ax.text(xi, -33, f"n={n_for_dim}", ha="center", va="top",
                    fontsize=7.5, color="gray")

    ax.set_title("OOD held-out refusal rate change vs Paper 2 base "
                 "(best-trained adapter per anchor)")
    fig.tight_layout()

    for out_dir in (FIGS_REPO, FIGS_MS):
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / "fig2-multi-anchor-delta.pdf", bbox_inches="tight")
        fig.savefig(out_dir / "fig2-multi-anchor-delta.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"wrote fig2-multi-anchor-delta ({len(deltas)} anchors x {len(dims)} dims)")


# ---------------------------------------------------------------------------
# Figure 3: toxicity refusal vs LR (Qwen + Llama, three LRs each)
# Companion line plot showing the monotonic in-distribution-vs-OOD trade.
# ---------------------------------------------------------------------------

def fig_toxicity_vs_lr():
    # x-axis: in-distribution accuracy peak (proxy for "training strength")
    # y-axis: OOD toxicity Δ vs base
    # colour: anchor; marker shape: LR

    rows = []  # list of (short, lr_label, lr_val, acc_peak, tox_delta_pp)
    for short, lr_label, condition, lr_val in TRAINED_RUNS:
        traj = _load_trajectory(short, condition)
        if traj is None: continue
        d = _delta_for(short, condition)
        if d is None or d.get("toxicity") is None: continue
        acc_peak = max(traj["acc"]) if traj["acc"] else float("nan")
        rows.append((short, lr_label, lr_val, acc_peak,
                     d["toxicity"]["delta"] * 100))

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    LR_MARKER = {"5e-6": "s", "1e-5": "v", "2e-5": "o", "5e-5": "^"}

    for short, lr_label, lr_val, acc_peak, tox_d in rows:
        ax.scatter(
            acc_peak, tox_d,
            color=ANCHOR_COLOR[short],
            marker=LR_MARKER[lr_label],
            s=85, edgecolor="black", linewidth=0.6,
            label=f"{ANCHOR_DISPLAY[short]} (LR={lr_label})",
            zorder=3,
        )
        ax.annotate(
            f"LR={lr_label}",
            xy=(acc_peak, tox_d),
            xytext=(7, 4), textcoords="offset points",
            fontsize=7.5, color=ANCHOR_COLOR[short],
        )

    # connect Qwen LR-sweep with a thin line to make the monotonic
    # degradation visible
    for short in ("qwen2.5-3b", "llama-3.2-3b"):
        pts = sorted(
            [(acc, tox) for s, _, _, acc, tox in rows if s == short],
            key=lambda p: p[0],
        )
        if len(pts) >= 2:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=ANCHOR_COLOR[short], linewidth=1.0,
                    alpha=0.4, zorder=1)

    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xlabel("In-distribution preference accuracy (peak)")
    ax.set_ylabel("Δ toxicity refusal vs base (pp, n=53)")
    ax.set_title("Training harder makes OOD toxicity refusal worse")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()

    for out_dir in (FIGS_REPO, FIGS_MS):
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / "fig3-toxicity-vs-lr.pdf", bbox_inches="tight")
        fig.savefig(out_dir / "fig3-toxicity-vs-lr.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"wrote fig3-toxicity-vs-lr ({len(rows)} points)")


# ---------------------------------------------------------------------------
# Figure 4: data scale lifts near-OOD (toxicity) but not far-OOD
# (cross-lingual). Two-panel comparison at the e6 baseline (~200
# pairs/anchor, locked LR per anchor) versus the x4 expansion (~434
# pairs/anchor, same LR per anchor). Every cell is delta-vs-base on the
# Paper 3 holdout split.
# ---------------------------------------------------------------------------

def fig_x4_dissociation():
    # Per-anchor representative trained run at each pair count.
    # e6 baseline: best-trained representative from the LR sweep
    #   (Qwen+Llama at lr=2e-5; Gemma at lr=5e-6 from rd-dpo-k4-bal-e6).
    # x4 expansion: rd-dpo-k4-bal-e6-x4 at the per-anchor working LR.
    e6_runs = {
        "qwen2.5-3b":   "rd-dpo-k4-bal-e6-lr2e5",
        "llama-3.2-3b": "rd-dpo-k4-bal-e6-lr2e5",
        "gemma-3-4b":   "rd-dpo-k4-bal-e6",
    }
    x4_run = "rd-dpo-k4-bal-e6-x4"

    dims = ["toxicity", "jailbreak", "overrefusal", "crosslingual"]
    dim_labels = ["Toxicity", "Jailbreak", "Over-refusal", "Cross-lingual"]

    e6_deltas, x4_deltas, ns = {}, {}, {}
    for short in ANCHOR_COLOR:
        e6 = _delta_for(short, e6_runs[short])
        x4 = _delta_for(short, x4_run)
        if e6 is None or x4 is None:
            print(f"  missing data for {short}; skipping in fig4")
            continue
        e6_deltas[short] = {dim: e6[dim]["delta"] * 100 for dim in dims if e6[dim]}
        x4_deltas[short] = {dim: x4[dim]["delta"] * 100 for dim in dims if x4[dim]}
        ns[short]        = {dim: x4[dim]["n"] for dim in dims if x4[dim]}

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6), sharey=True)
    width = 0.36
    x = np.arange(len(dims))

    for ax, short in zip(axes, ANCHOR_COLOR):
        if short not in e6_deltas: continue
        e6_y = [e6_deltas[short].get(d, 0.0) for d in dims]
        x4_y = [x4_deltas[short].get(d, 0.0) for d in dims]
        ax.bar(x - width/2, e6_y, width,
               color=ANCHOR_COLOR[short], alpha=0.45,
               edgecolor="black", linewidth=0.6, label="e6 (~200 pairs)")
        ax.bar(x + width/2, x4_y, width,
               color=ANCHOR_COLOR[short],
               edgecolor="black", linewidth=0.6, label="x4 (~434 pairs)")
        # Annotate values in pp
        for xi, y in zip(x - width/2, e6_y):
            va = "bottom" if y >= 0 else "top"
            ax.text(xi, y + (0.4 if y >= 0 else -0.4),
                    f"{y:+.0f}", ha="center", va=va, fontsize=7)
        for xi, y in zip(x + width/2, x4_y):
            va = "bottom" if y >= 0 else "top"
            ax.text(xi, y + (0.4 if y >= 0 else -0.4),
                    f"{y:+.0f}", ha="center", va=va, fontsize=7)
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(dim_labels, fontsize=8, rotation=20, ha="right")
        ax.set_title(ANCHOR_DISPLAY[short], fontsize=10)
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_ylim(-25, 18)
        ax.legend(loc="lower left", fontsize=7, frameon=False)

    axes[0].set_ylabel("Δ refusal rate (rd-dpo − base, pp)")

    fig.suptitle("Data scale lifts near-OOD but not far-OOD on Romanian",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    for out_dir in (FIGS_REPO, FIGS_MS):
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / "fig4-x4-dissociation.pdf", bbox_inches="tight")
        fig.savefig(out_dir / "fig4-x4-dissociation.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"wrote fig4-x4-dissociation ({len(e6_deltas)} anchors x 2 conditions x {len(dims)} dims)")


if __name__ == "__main__":
    fig_trajectories()
    fig_multi_anchor_delta()
    fig_toxicity_vs_lr()
    fig_x4_dissociation()
