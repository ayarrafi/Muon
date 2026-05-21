"""Train WRN-28-10 on CIFAR-10 with Muon (+ AdamW for non-matrix params).

Designed for sweeping batch size and Muon momentum, with per-step logging.

Example:
    python train.py --batch-size 128 --momentum 0.95 --steps 5000 --log-csv runs/bs128_m095.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data import build_loaders
from model import wrn_28_10
from muon import Muon, split_params_for_muon


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader, device: torch.device) -> tuple[float, float]:
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    model.train()
    return loss_sum / total, correct / total


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # Sweep variables.
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--momentum", type=float, default=0.95, help="Muon momentum")
    # Training budget — pass either --steps (optimizer steps) or --epochs (passes over
    # the train set). If --epochs is given it overrides --steps.
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--epochs", type=int, default=None,
                   help="if set, training budget is epochs * steps_per_epoch; overrides --steps")
    p.add_argument("--eval-per-epoch", type=int, default=2,
                   help="number of test evaluations per epoch (>=1)")
    p.add_argument("--log-every", type=int, default=20)
    # Optimizer hyperparameters.
    p.add_argument("--muon-lr", type=float, default=0.02)
    p.add_argument("--adamw-lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--ns-steps", type=int, default=5)
    p.add_argument("--nesterov", action="store_true", default=True)
    # Misc.
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--log-csv", type=str, default=None, help="Path to per-step CSV log")
    p.add_argument("--tag", type=str, default="")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    train_loader, test_loader = build_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=not args.no_augment,
    )

    steps_per_epoch = len(train_loader)
    if args.epochs is not None:
        args.steps = args.epochs * steps_per_epoch
        print(f"epochs mode: {args.epochs} epochs * {steps_per_epoch} steps/epoch = {args.steps} steps")

    eval_every = max(1, steps_per_epoch // max(1, args.eval_per_epoch))
    print(f"eval cadence: every {eval_every} steps "
          f"({args.eval_per_epoch}x per epoch of {steps_per_epoch} steps)")

    model = wrn_28_10(num_classes=10, dropout=args.dropout).to(device)
    model = model.to(memory_format=torch.channels_last)

    muon_params, adamw_params = split_params_for_muon(model)
    muon_opt = Muon(
        muon_params,
        lr=args.muon_lr,
        momentum=args.momentum,
        nesterov=args.nesterov,
        ns_steps=args.ns_steps,
        weight_decay=args.weight_decay,
    )
    adamw_opt = torch.optim.AdamW(
        adamw_params,
        lr=args.adamw_lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    n_muon = sum(p.numel() for p in muon_params)
    n_adam = sum(p.numel() for p in adamw_params)
    print(f"params: muon={n_muon:,}  adamw={n_adam:,}  total={n_muon + n_adam:,}")
    print(f"config: {json.dumps(vars(args))}")

    log_rows: list[dict] = []
    csv_writer = None
    csv_file = None
    if args.log_csv:
        Path(args.log_csv).parent.mkdir(parents=True, exist_ok=True)
        csv_file = open(args.log_csv, "w", newline="")
        csv_writer = csv.DictWriter(
            csv_file,
            fieldnames=["step", "samples_seen", "train_loss", "train_acc", "test_loss", "test_acc", "wallclock"],
        )
        csv_writer.writeheader()

    train_iter = cycle(train_loader)
    model.train()
    t0 = time.time()
    loss_ema, acc_ema = None, None
    ema_decay = 0.98

    for step in range(1, args.steps + 1):
        x, y = next(train_iter)
        x = x.to(device, non_blocking=True, memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = F.cross_entropy(logits, y)

        muon_opt.zero_grad(set_to_none=True)
        adamw_opt.zero_grad(set_to_none=True)
        loss.backward()
        muon_opt.step()
        adamw_opt.step()

        with torch.no_grad():
            acc = (logits.argmax(1) == y).float().mean().item()
        loss_v = loss.item()
        loss_ema = loss_v if loss_ema is None else ema_decay * loss_ema + (1 - ema_decay) * loss_v
        acc_ema = acc if acc_ema is None else ema_decay * acc_ema + (1 - ema_decay) * acc

        if step % args.log_every == 0:
            elapsed = time.time() - t0
            print(
                f"step {step:6d} | samples {step * args.batch_size:>9,} | "
                f"loss {loss_ema:.4f} | acc {acc_ema*100:5.2f}% | "
                f"{elapsed:6.1f}s"
            )

        do_eval = (step % eval_every == 0) or (step == args.steps)
        if do_eval:
            test_loss, test_acc = evaluate(model, test_loader, device)
            print(f"  >> eval @ step {step}: test_loss {test_loss:.4f} | test_acc {test_acc*100:5.2f}%")
            row = {
                "step": step,
                "samples_seen": step * args.batch_size,
                "train_loss": loss_ema,
                "train_acc": acc_ema,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "wallclock": time.time() - t0,
            }
            log_rows.append(row)
            if csv_writer is not None:
                csv_writer.writerow(row)
                csv_file.flush()

    if csv_file is not None:
        csv_file.close()

    final = log_rows[-1] if log_rows else None
    print(f"done. final test_acc = {final['test_acc']*100:.2f}%" if final else "done.")


if __name__ == "__main__":
    main()
