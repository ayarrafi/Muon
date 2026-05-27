"""Plot Muon vs AdamW fine-tuning curves from two CSV files.

Examples:
    python plot_finetune.py runs_ft/muon.csv runs_ft/adamw.csv --out figs/finetune.png
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def load_csv(path: str) -> dict[str, list]:
    rows = {"step": [], "train_loss": [], "val_loss": [], "wallclock": []}
    with open(path) as f:
        for row in csv.DictReader(f):
            for k in rows:
                rows[k].append(float(row[k]))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("muon_csv",  help="CSV from --optimizer muon run")
    p.add_argument("adamw_csv", help="CSV from --optimizer adamw run")
    p.add_argument("--out", default="figs/finetune_comparison.png")
    args = p.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("pip install matplotlib")

    muon  = load_csv(args.muon_csv)
    adamw = load_csv(args.adamw_csv)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, y_key, title in [
        (axes[0], "train_loss", "Train loss (EMA)"),
        (axes[1], "val_loss",   "Val loss"),
    ]:
        ax.plot(muon["step"],  muon[y_key],  color="tomato",    label="Muon",  linewidth=2)
        ax.plot(adamw["step"], adamw[y_key], color="steelblue", label="AdamW", linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.legend()
        ax.grid(alpha=0.3)

    fig.suptitle("Muon vs AdamW — LLM fine-tuning", fontsize=13)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"saved: {args.out}")

    print("\n── Summary ──────────────────────────────")
    for data, name in [(muon, "Muon"), (adamw, "AdamW")]:
        if data["val_loss"]:
            print(f"  {name:5s}  final_val={data['val_loss'][-1]:.4f}  best_val={min(data['val_loss']):.4f}")


if __name__ == "__main__":
    main()
