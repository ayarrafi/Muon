"""Sweep driver: runs train.py over a grid of (batch_size, momentum).

Examples:
    # Default grid (5 batch sizes x 4 momenta = 20 runs)
    python sweep.py --steps 5000 --out-dir runs/

    # Custom grid
    python sweep.py --batch-sizes 128,512 --momenta 0.9,0.95,0.98 --steps 3000

    # See what would run without actually launching anything
    python sweep.py --dry-run

    # Pass through extra args to train.py (e.g. --muon-lr, --num-workers)
    python sweep.py --extra "--num-workers 8 --dropout 0.1"
"""
from __future__ import annotations

import argparse
import itertools
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path


def _parse_list(s: str, type_fn):
    return [type_fn(x.strip()) for x in s.split(",") if x.strip()]


def _run_name(batch_size: int, momentum: float) -> str:
    return f"bs{batch_size}_m{str(momentum).replace('.', 'p')}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--batch-sizes", type=str, default="64,128,256,512,1024,2048")
    p.add_argument("--momenta", type=str, default="0.8,0.9,0.95,0.99")
    p.add_argument("--seeds", type=str, default="0", help="comma-separated seed list")
    p.add_argument("--epochs", type=int, default=25,
                   help="if set, every run trains for this many passes over the train set; "
                        "smaller batch sizes therefore take more optimizer steps")
    p.add_argument("--steps", type=int, default=None,
                   help="optimizer-step budget (ignored if --epochs is set)")
    p.add_argument("--muon-lr", type=float, default=0.02)
    p.add_argument("--eval-per-epoch", type=int, default=2,
                   help="number of test evaluations per epoch (>=1)")
    p.add_argument("--out-dir", type=str, default="runs")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--rerun", action="store_true", help="re-run even if CSV exists")
    p.add_argument("--extra", type=str, default="", help="extra args passed through to train.py")
    args = p.parse_args()

    batch_sizes = _parse_list(args.batch_sizes, int)
    momenta = _parse_list(args.momenta, float)
    seeds = _parse_list(args.seeds, int)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = list(itertools.product(seeds, batch_sizes, momenta))
    extra_args = shlex.split(args.extra) if args.extra else []

    budget_desc = f"{args.epochs} epochs" if args.epochs is not None else f"{args.steps} steps"
    print(f"sweep: {len(grid)} configurations, {budget_desc} each")
    print(f"  batch_sizes={batch_sizes}")
    print(f"  momenta={momenta}")
    print(f"  seeds={seeds}")
    print(f"  out_dir={out_dir.resolve()}")
    print()

    manifest = {
        "batch_sizes": batch_sizes,
        "momenta": momenta,
        "seeds": seeds,
        "steps": args.steps,
        "epochs": args.epochs,
        "muon_lr": args.muon_lr,
        "extra": args.extra,
        "runs": [],
    }

    sweep_t0 = time.time()
    for i, (seed, bs, mom) in enumerate(grid, 1):
        base_name = _run_name(bs, mom)
        name = f"{base_name}_s{seed}" if len(seeds) > 1 else base_name
        csv_path = out_dir / f"{name}.csv"

        if csv_path.exists() and not args.rerun:
            print(f"[{i}/{len(grid)}] skip {name} (csv exists, use --rerun to overwrite)")
            manifest["runs"].append({"name": name, "status": "skipped", "csv": str(csv_path)})
            continue

        cmd = [
            sys.executable, "train.py",
            "--batch-size", str(bs),
            "--momentum", str(mom),
            "--muon-lr", str(args.muon_lr),
            "--seed", str(seed),
            "--eval-per-epoch", str(args.eval_per_epoch),
            "--log-csv", str(csv_path),
            "--tag", name,
        ] + extra_args
        if args.epochs is not None:
            cmd.extend(["--epochs", str(args.epochs)])
        else:
            cmd.extend(["--steps", str(args.steps)])

        print(f"[{i}/{len(grid)}] run {name}")
        print(f"  $ {' '.join(shlex.quote(c) for c in cmd)}")
        if args.dry_run:
            manifest["runs"].append({"name": name, "status": "dry-run", "cmd": cmd})
            continue

        run_t0 = time.time()
        try:
            subprocess.run(cmd, check=True)
            status = "ok"
        except subprocess.CalledProcessError as e:
            print(f"  !! {name} failed (exit {e.returncode}), continuing")
            status = f"failed (exit {e.returncode})"
        except KeyboardInterrupt:
            print("  !! interrupted")
            manifest["runs"].append({"name": name, "status": "interrupted"})
            break
        elapsed = time.time() - run_t0
        print(f"  done in {elapsed:.1f}s")
        manifest["runs"].append({
            "name": name,
            "status": status,
            "csv": str(csv_path),
            "seconds": elapsed,
        })

    manifest["total_seconds"] = time.time() - sweep_t0
    manifest_path = out_dir / "sweep_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nmanifest -> {manifest_path}")
    print(f"sweep done in {manifest['total_seconds']:.1f}s")


if __name__ == "__main__":
    main()
