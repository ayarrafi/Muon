"""Fine-tune a pretrained LLM with Muon or AdamW and compare convergence.

This extends the from-scratch GPT experiment (train_lm.py) to the more
practical setting of fine-tuning a pretrained HuggingFace model, which was
the original motivation for Muon (Keller Jordan, 2024).

Dataset : HuggingFaceH4/ultrachat_200k  (default)  or  teknium/OpenHermes-2.5
Model   : any HuggingFace causal LM, default Qwen/Qwen3-0.6B

Examples:
    python train_finetune.py --optimizer muon  --log-csv runs_ft/muon.csv
    python train_finetune.py --optimizer adamw --log-csv runs_ft/adamw.csv
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from muon import Muon

# ── Helpers ───────────────────────────────────────────────────────────────────

def best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def split_params_for_muon_lm(model: nn.Module):
    """Split pretrained LLM params into (muon_params, adamw_params).

    Muon handles hidden 2-D weight matrices.
    AdamW handles embeddings, lm_head, biases, and layer norms — anything
    that is either 1-D or at the input/output boundary of the network.
    """
    muon_params, adamw_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Input embedding and output head: treat like input/output, use AdamW
        if any(k in name for k in ("embed", "lm_head", "wte", "wpe")):
            adamw_params.append(p)
        elif p.ndim >= 2:
            muon_params.append(p)
        else:
            adamw_params.append(p)
    return muon_params, adamw_params


# ── Data ──────────────────────────────────────────────────────────────────────

def _tokenize_conversation(messages, tokenizer, max_length):
    """Tokenize a chat conversation; mask prompt so loss is on assistant only."""
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    enc = tokenizer(full_text, truncation=True, max_length=max_length)
    input_ids = enc["input_ids"]
    labels    = input_ids[:]

    if len(messages) >= 2:
        prompt_text = tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True
        )
        prompt_len = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
        labels[: min(prompt_len, len(labels))] = [-100] * min(prompt_len, len(labels))

    return {"input_ids": input_ids, "attention_mask": enc["attention_mask"], "labels": labels}


def load_ultrachat(tokenizer, max_length, n_train, n_eval, cache_dir):
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", cache_dir=cache_dir)
    ds = ds.shuffle(seed=0).select(range(n_train + n_eval))

    def tokenize(batch):
        out = {"input_ids": [], "attention_mask": [], "labels": []}
        for msgs in batch["messages"]:
            tok = _tokenize_conversation(msgs, tokenizer, max_length)
            for k in out:
                out[k].append(tok[k])
        return out

    ds = ds.map(tokenize, batched=True, batch_size=64, remove_columns=ds.column_names)
    return ds.select(range(n_train)), ds.select(range(n_train, n_train + n_eval))


def load_openhermes(tokenizer, max_length, n_train, n_eval, cache_dir):
    ds = load_dataset("teknium/OpenHermes-2.5", split="train", cache_dir=cache_dir)
    ds = ds.shuffle(seed=0)

    def to_messages(row):
        msgs = []
        for turn in (row.get("conversations") or []):
            role = "user" if turn.get("from") in ("human", "user") else "assistant"
            msgs.append({"role": role, "content": turn.get("value", "")})
        return {"messages": msgs}

    ds = ds.map(to_messages, remove_columns=ds.column_names)
    ds = ds.filter(lambda x: len(x["messages"]) >= 2).select(range(n_train + n_eval))

    def tokenize(batch):
        out = {"input_ids": [], "attention_mask": [], "labels": []}
        for msgs in batch["messages"]:
            tok = _tokenize_conversation(msgs, tokenizer, max_length)
            for k in out:
                out[k].append(tok[k])
        return out

    ds = ds.map(tokenize, batched=True, batch_size=64, remove_columns=["messages"])
    return ds.select(range(n_train)), ds.select(range(n_train, n_train + n_eval))


def collate(pad_id):
    def _fn(batch):
        max_len = max(len(x["input_ids"]) for x in batch)
        ids, masks, lbls = [], [], []
        for x in batch:
            p = max_len - len(x["input_ids"])
            ids.append(list(x["input_ids"]) + [pad_id] * p)
            masks.append(list(x["attention_mask"]) + [0] * p)
            lbls.append(list(x["labels"]) + [-100] * p)
        return {
            "input_ids":      torch.tensor(ids,   dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
            "labels":         torch.tensor(lbls,  dtype=torch.long),
        }
    return _fn


# ── Eval ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device, n_batches=30):
    model.eval()
    total, n = 0.0, 0
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        loss = model(**batch).loss
        if loss is not None:
            total += loss.item()
            n += 1
    model.train()
    return total / max(n, 1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    # ── Model & tokenizer ────────────────────────────────────────────────────
    print(f"loading {args.model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, trust_remote_code=True, cache_dir=args.cache_dir
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # bfloat16 requires Ampere (A100+); V100 only supports float16/float32
    cc = torch.cuda.get_device_capability() if device.type == "cuda" else (0, 0)
    dtype = torch.bfloat16 if (device.type == "mps" or cc[0] >= 8) else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, dtype=dtype,
        trust_remote_code=True, cache_dir=args.cache_dir,
    )
    model.config.use_cache = False
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  {n_params:.0f}M params on {device}")

    # ── Dataset ──────────────────────────────────────────────────────────────
    print(f"loading {args.dataset} ...")
    loader_fn = load_ultrachat if args.dataset == "ultrachat" else load_openhermes
    train_ds, eval_ds = loader_fn(
        tokenizer, args.max_length, args.n_train, args.n_eval, args.cache_dir
    )
    train_ds.set_format("torch")
    eval_ds.set_format("torch")
    pad_id       = tokenizer.pad_token_id
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate(pad_id), num_workers=0)
    eval_loader  = DataLoader(eval_ds,  batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate(pad_id), num_workers=0)
    print(f"  train={len(train_ds)}  eval={len(eval_ds)}")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    print(f"optimizer: {args.optimizer}")
    if args.optimizer == "muon":
        muon_p, adamw_p = split_params_for_muon_lm(model)
        print(f"  muon params:  {sum(p.numel() for p in muon_p)/1e6:.1f}M")
        print(f"  adamw params: {sum(p.numel() for p in adamw_p)/1e6:.1f}M")
        opt_muon  = Muon(muon_p, lr=args.muon_lr, momentum=args.momentum, weight_decay=0.0)
        opt_adamw = torch.optim.AdamW(adamw_p, lr=args.adamw_lr,
                                      betas=(0.9, 0.95), weight_decay=args.weight_decay)
        optimizers = [opt_muon, opt_adamw]
    else:
        all_p = list({id(p): p for p in model.parameters() if p.requires_grad}.values())
        opt_adamw  = torch.optim.AdamW(all_p, lr=args.adamw_lr,
                                       betas=(0.9, 0.95), weight_decay=args.weight_decay)
        optimizers = [opt_adamw]

    # ── CSV logging ──────────────────────────────────────────────────────────
    csv_file, csv_writer = None, None
    if args.log_csv:
        Path(args.log_csv).parent.mkdir(parents=True, exist_ok=True)
        csv_file   = open(args.log_csv, "w", newline="")
        csv_writer = csv.DictWriter(
            csv_file, fieldnames=["step", "train_loss", "val_loss", "wallclock"]
        )
        csv_writer.writeheader()

    # ── Training loop ─────────────────────────────────────────────────────────
    model.train()
    loss_ema = None
    t0       = time.time()
    step     = 0

    for _ in range(9999):
        for batch in train_loader:
            if step >= args.steps:
                break
            batch = {k: v.to(device) for k, v in batch.items()}

            for opt in optimizers:
                opt.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            for opt in optimizers:
                opt.step()

            loss_ema = loss.item() if loss_ema is None else 0.98 * loss_ema + 0.02 * loss.item()
            step += 1

            if step % args.log_every == 0:
                print(f"step {step:5d} | train_loss {loss_ema:.4f} | {time.time()-t0:.1f}s")

            if step % args.eval_every == 0 or step == args.steps:
                val_loss = evaluate(model, eval_loader, device)
                print(f"  >> val_loss {val_loss:.4f} @ step {step}")
                if csv_writer:
                    csv_writer.writerow({
                        "step": step, "train_loss": round(loss_ema, 5),
                        "val_loss": round(val_loss, 5), "wallclock": round(time.time() - t0, 1),
                    })
                    csv_file.flush()

        if step >= args.steps:
            break

    if csv_file:
        csv_file.close()
    print("done.")


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--optimizer",    choices=["muon", "adamw"], default="muon")
    p.add_argument("--model-id",     default="Qwen/Qwen3-0.6B")
    p.add_argument("--dataset",      choices=["ultrachat", "openhermes"], default="ultrachat")
    p.add_argument("--steps",        type=int,   default=1000)
    p.add_argument("--batch-size",   type=int,   default=4)
    p.add_argument("--max-length",   type=int,   default=512)
    p.add_argument("--n-train",      type=int,   default=5000)
    p.add_argument("--n-eval",       type=int,   default=500)
    p.add_argument("--muon-lr",      type=float, default=0.02)
    p.add_argument("--adamw-lr",     type=float, default=3e-4)
    p.add_argument("--momentum",     type=float, default=0.95)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--eval-every",   type=int,   default=100)
    p.add_argument("--log-every",    type=int,   default=50)
    p.add_argument("--log-csv",      default=None)
    p.add_argument("--cache-dir",    default="/scratch/hf_cache")
    p.add_argument("--seed",         type=int,   default=0)
    p.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


if __name__ == "__main__":
    main()
