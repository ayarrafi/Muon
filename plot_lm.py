"""Plot Muon vs AdamW learning curves for the LLM experiment.

Usage:
    python plot_lm.py --runs-dir runs_lm --out-dir figs_lm
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


COLORS = {"muon": "#e07b39", "adamw": "#4c72b0"}
LABELS = {"muon": "Muon + AdamW", "adamw": "AdamW"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", default="runs_lm")
    p.add_argument("--out-dir", default="figs_lm")
    args = p.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9})

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))

    for opt in ("muon", "adamw"):
        csv_path = runs_dir / f"{opt}.csv"
        if not csv_path.exists():
            print(f"  missing {csv_path}, skipping")
            continue
        df = pd.read_csv(csv_path)
        color = COLORS[opt]
        label = LABELS[opt]
        axes[0].plot(df["step"], df["train_loss"], color=color, label=label, lw=1.5)
        axes[1].plot(df["step"], df["val_loss"], color=color, label=label, lw=1.5)

    for ax, title, ylabel in zip(
        axes,
        ["Train loss vs step", "Val loss vs step"],
        ["cross-entropy loss", "cross-entropy loss"],
    ):
        ax.set_xlabel("optimizer step")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Muon vs AdamW — Tiny Shakespeare (char-level GPT)", fontsize=10)
    fig.tight_layout()
    out_path = out_dir / "muon_vs_adamw_lm.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
