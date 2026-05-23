"""Train a small GPT on Tiny Shakespeare with Muon or AdamW.

Downloads the dataset automatically on first run (~1 MB).

Examples:
    python train_lm.py --optimizer muon --log-csv runs_lm/muon.csv
    python train_lm.py --optimizer adamw --log-csv runs_lm/adamw.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model_lm import GPT, split_params_for_muon_gpt
from muon import Muon

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = Path(__file__).resolve().parent / "data_shakespeare.txt"


def get_data() -> tuple[torch.Tensor, int, dict]:
    if not DATA_PATH.exists():
        print(f"downloading Tiny Shakespeare -> {DATA_PATH}")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    text = DATA_PATH.read_text(encoding="utf-8")
    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    return data, vocab_size, itos


def make_batch(
    data: torch.Tensor,
    batch_size: int,
    context_len: int,
    split_end: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(split_end - context_len, (batch_size,))
    x = torch.stack([data[i : i + context_len] for i in ix]).to(device)
    y = torch.stack([data[i + 1 : i + context_len + 1] for i in ix]).to(device)
    return x, y


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    data: torch.Tensor,
    batch_size: int,
    context_len: int,
    val_start: int,
    device: torch.device,
    n_batches: int = 20,
) -> float:
    model.eval()
    losses = []
    for _ in range(n_batches):
        ix = torch.randint(val_start, len(data) - context_len, (batch_size,))
        x = torch.stack([data[i : i + context_len] for i in ix]).to(device)
        y = torch.stack([data[i + 1 : i + context_len + 1] for i in ix]).to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--optimizer", choices=["muon", "adamw"], default="muon")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--context-len", type=int, default=256)
    p.add_argument("--d-model", type=int, default=384)
    p.add_argument("--n-heads", type=int, default=6)
    p.add_argument("--n-layers", type=int, default=6)
    p.add_argument("--muon-lr", type=float, default=0.02)
    p.add_argument("--adamw-lr", type=float, default=3e-3)
    p.add_argument("--momentum", type=float, default=0.95)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--log-csv", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    data, vocab_size, _ = get_data()

    # 90/10 train/val split
    val_start = int(0.9 * len(data))

    model = GPT(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        context_len=args.context_len,
    ).to(device)
    print(f"GPT: vocab={vocab_size}, params={model.num_params():,}")
    print(f"optimizer: {args.optimizer}")

    if args.optimizer == "muon":
        muon_params, adamw_params = split_params_for_muon_gpt(model)
        opt_muon = Muon(muon_params, lr=args.muon_lr, momentum=args.momentum, weight_decay=0.0)
        opt_adamw = torch.optim.AdamW(adamw_params, lr=args.adamw_lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
        optimizers = [opt_muon, opt_adamw]
    else:
        all_params = [p for p in model.parameters() if p.requires_grad]
        # deduplicate tied weights
        seen: set[int] = set()
        unique_params = []
        for p in all_params:
            if id(p) not in seen:
                seen.add(id(p))
                unique_params.append(p)
        opt_adamw = torch.optim.AdamW(unique_params, lr=args.adamw_lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
        optimizers = [opt_adamw]

    csv_file = None
    csv_writer = None
    if args.log_csv:
        Path(args.log_csv).parent.mkdir(parents=True, exist_ok=True)
        csv_file = open(args.log_csv, "w", newline="")
        csv_writer = csv.DictWriter(csv_file, fieldnames=["step", "train_loss", "val_loss", "wallclock"])
        csv_writer.writeheader()

    model.train()
    t0 = time.time()
    loss_ema = None

    for step in range(1, args.steps + 1):
        x, y = make_batch(data, args.batch_size, args.context_len, val_start, device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        loss.backward()
        for opt in optimizers:
            opt.step()

        loss_v = loss.item()
        loss_ema = loss_v if loss_ema is None else 0.98 * loss_ema + 0.02 * loss_v

        if step % args.log_every == 0:
            print(f"step {step:5d} | train_loss {loss_ema:.4f} | {time.time()-t0:.1f}s")

        if step % args.eval_every == 0 or step == args.steps:
            val_loss = evaluate(model, data, args.batch_size, args.context_len, val_start, device)
            print(f"  >> val_loss {val_loss:.4f} @ step {step}")
            if csv_writer:
                csv_writer.writerow({"step": step, "train_loss": loss_ema, "val_loss": val_loss, "wallclock": time.time() - t0})
                csv_file.flush()

    if csv_file:
        csv_file.close()


if __name__ == "__main__":
    main()
