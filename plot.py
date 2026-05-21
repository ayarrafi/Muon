"""Plotting script for the Muon batch-size x momentum sweep.

Produces:
  * Learning curves (test accuracy vs. epoch and vs. optimizer step).
  * Final-accuracy heatmap over the (batch_size, momentum) grid.
  * Sample-efficiency: steps and samples to reach a target test accuracy.

Usage:
    python plot.py --runs-dir runs --out-dir figs --target 0.90
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUN_RE = re.compile(r"bs(?P<bs>\d+)_m(?P<m>0p\d+)")
CIFAR_TRAIN_SIZE = 50_000


def parse_name(name: str) -> tuple[int, float] | None:
    m = RUN_RE.search(name)
    if not m:
        return None
    bs = int(m.group("bs"))
    mom = float(m.group("m").replace("p", "."))
    return bs, mom


def load_runs(runs_dir: Path) -> dict[tuple[int, float], pd.DataFrame]:
    runs: dict[tuple[int, float], pd.DataFrame] = {}
    for csv in sorted(runs_dir.glob("bs*_m*.csv")):
        key = parse_name(csv.stem)
        if key is None:
            continue
        df = pd.read_csv(csv)
        bs, _ = key
        df["epoch"] = df["samples_seen"] / CIFAR_TRAIN_SIZE
        df["batch_size"] = bs
        runs[key] = df
    return runs


def steps_to_target(df: pd.DataFrame, target: float) -> dict[str, float] | None:
    """Return the first eval row whose test_acc crosses `target`; None if never."""
    hit = df[df["test_acc"] >= target]
    if hit.empty:
        return None
    row = hit.iloc[0]
    return {
        "step": int(row["step"]),
        "samples_seen": int(row["samples_seen"]),
        "epoch": float(row["epoch"]),
        "wallclock": float(row["wallclock"]),
    }


def best_acc(df: pd.DataFrame) -> float:
    return float(df["test_acc"].max())


def final_acc(df: pd.DataFrame) -> float:
    return float(df["test_acc"].iloc[-1])


# ---------------- Plots ----------------

def _colormap(momenta: list[float]):
    cmap = plt.cm.viridis
    return {m: cmap(i / max(1, len(momenta) - 1)) for i, m in enumerate(sorted(momenta))}


def plot_curves(
    runs: dict[tuple[int, float], pd.DataFrame],
    out_dir: Path,
    x_axis: str = "epoch",
) -> None:
    """One panel per batch size; lines = momenta. Axis is `epoch` (fair across bs)."""
    batch_sizes = sorted({k[0] for k in runs})
    momenta = sorted({k[1] for k in runs})
    colors = _colormap(momenta)

    ncols = 3
    nrows = int(np.ceil(len(batch_sizes) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.5 * nrows), sharey=True)
    axes = np.atleast_2d(axes)

    for idx, bs in enumerate(batch_sizes):
        ax = axes[idx // ncols, idx % ncols]
        for mom in momenta:
            df = runs.get((bs, mom))
            if df is None:
                continue
            ax.plot(df[x_axis], df["test_acc"] * 100, color=colors[mom], label=f"$\\mu={mom}$", lw=1.4)
        ax.set_title(f"batch size = {bs}")
        ax.set_xlabel(x_axis.replace("_", " "))
        ax.grid(True, alpha=0.3)
        ax.set_ylim(20, 100)
    for idx in range(len(batch_sizes), nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)
    for r in range(nrows):
        axes[r, 0].set_ylabel("test accuracy (%)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(momenta), bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out_path = out_dir / f"curves_by_bs_{x_axis}.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_curves_step(
    runs: dict[tuple[int, float], pd.DataFrame],
    out_dir: Path,
) -> None:
    """Single panel: test acc vs optimizer step, all (bs, mom) lines.

    Per-step view is the key research question: how does each optimizer
    update progress when bs/momentum change?
    """
    batch_sizes = sorted({k[0] for k in runs})
    momenta = sorted({k[1] for k in runs})
    fig, axes = plt.subplots(1, len(momenta), figsize=(3.0 * len(momenta), 2.7), sharey=True)
    cmap = plt.cm.plasma
    bs_colors = {bs: cmap(i / max(1, len(batch_sizes) - 1)) for i, bs in enumerate(batch_sizes)}

    for j, mom in enumerate(momenta):
        ax = axes[j]
        for bs in batch_sizes:
            df = runs.get((bs, mom))
            if df is None:
                continue
            ax.plot(df["step"], df["test_acc"] * 100, color=bs_colors[bs], label=f"bs={bs}", lw=1.4)
        ax.set_title(f"$\\mu = {mom}$")
        ax.set_xscale("log")
        ax.set_xlabel("optimizer step")
        ax.grid(True, which="both", alpha=0.3)
        ax.set_ylim(20, 100)
    axes[0].set_ylabel("test accuracy (%)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(batch_sizes), bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out_path = out_dir / "curves_per_step.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_heatmap(
    runs: dict[tuple[int, float], pd.DataFrame],
    out_dir: Path,
    target: float,
) -> None:
    batch_sizes = sorted({k[0] for k in runs})
    momenta = sorted({k[1] for k in runs})

    final = np.full((len(batch_sizes), len(momenta)), np.nan)
    best = np.full_like(final, np.nan)
    epochs_to = np.full_like(final, np.nan)
    steps_to = np.full_like(final, np.nan)
    for i, bs in enumerate(batch_sizes):
        for j, mom in enumerate(momenta):
            df = runs.get((bs, mom))
            if df is None:
                continue
            final[i, j] = final_acc(df) * 100
            best[i, j] = best_acc(df) * 100
            hit = steps_to_target(df, target)
            if hit is not None:
                epochs_to[i, j] = hit["epoch"]
                steps_to[i, j] = hit["step"]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    panels = [
        ("Best test acc (%)", best, "viridis", "{:.1f}"),
        ("Final test acc (%)", final, "viridis", "{:.1f}"),
        (f"Epochs to {target*100:.0f}%", epochs_to, "viridis_r", "{:.1f}"),
    ]
    for ax, (title, mat, cmap, fmt) in zip(axes, panels):
        im = ax.imshow(mat, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(momenta)))
        ax.set_xticklabels([f"{m}" for m in momenta])
        ax.set_yticks(range(len(batch_sizes)))
        ax.set_yticklabels([str(bs) for bs in batch_sizes])
        ax.set_xlabel("momentum $\\mu$")
        ax.set_ylabel("batch size")
        ax.set_title(title)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if np.isnan(v):
                    txt = "—"
                else:
                    txt = fmt.format(v)
                ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                        color="white" if (np.isnan(v) or v < np.nanmean(mat)) else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    out_path = out_dir / "heatmap.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_steps_to_target(
    runs: dict[tuple[int, float], pd.DataFrame],
    out_dir: Path,
    target: float,
) -> None:
    batch_sizes = sorted({k[0] for k in runs})
    momenta = sorted({k[1] for k in runs})
    colors = _colormap(momenta)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))

    for mom in momenta:
        steps = []
        samples = []
        xs = []
        for bs in batch_sizes:
            df = runs.get((bs, mom))
            if df is None:
                continue
            hit = steps_to_target(df, target)
            if hit is None:
                continue
            steps.append(hit["step"])
            samples.append(hit["samples_seen"])
            xs.append(bs)
        axes[0].plot(xs, steps, marker="o", color=colors[mom], label=f"$\\mu={mom}$", lw=1.4)
        axes[1].plot(xs, samples, marker="o", color=colors[mom], label=f"$\\mu={mom}$", lw=1.4)

    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("batch size")
    axes[0].set_ylabel(f"steps to {target*100:.0f}% test acc")
    axes[0].grid(True, which="both", alpha=0.3)

    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("batch size")
    axes[1].set_ylabel(f"samples to {target*100:.0f}% test acc")
    axes[1].grid(True, which="both", alpha=0.3)

    axes[0].legend(fontsize=8, loc="best")
    fig.tight_layout()
    out_path = out_dir / "sample_efficiency.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def write_summary_csv(
    runs: dict[tuple[int, float], pd.DataFrame],
    out_dir: Path,
    target: float,
) -> None:
    rows = []
    for (bs, mom), df in sorted(runs.items()):
        hit = steps_to_target(df, target)
        rows.append({
            "batch_size": bs,
            "momentum": mom,
            "best_test_acc": best_acc(df),
            "final_test_acc": final_acc(df),
            "steps_to_target": hit["step"] if hit else None,
            "samples_to_target": hit["samples_seen"] if hit else None,
            "epochs_to_target": hit["epoch"] if hit else None,
            "total_steps": int(df["step"].iloc[-1]),
            "total_epochs": float(df["epoch"].iloc[-1]),
        })
    summary = pd.DataFrame(rows)
    out_path = out_dir / "summary.csv"
    summary.to_csv(out_path, index=False)
    print(f"  wrote {out_path}")
    print(summary.to_string(index=False))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--out-dir", default="figs")
    p.add_argument("--target", type=float, default=0.90)
    args = p.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = load_runs(runs_dir)
    if not runs:
        raise SystemExit(f"No runs found in {runs_dir}")
    print(f"loaded {len(runs)} runs from {runs_dir}")

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9})

    plot_curves(runs, out_dir, x_axis="epoch")
    plot_curves_step(runs, out_dir)
    plot_heatmap(runs, out_dir, args.target)
    plot_steps_to_target(runs, out_dir, args.target)
    write_summary_csv(runs, out_dir, args.target)


if __name__ == "__main__":
    main()
