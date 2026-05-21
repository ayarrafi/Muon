# Muon optimizer study on CIFAR-10

An EPFL *Optimization for Machine Learning* (OptML) project investigating how
**batch size** and the **momentum** parameter influence the performance of the
[Muon](https://kellerjordan.github.io/posts/muon/) optimizer (Keller Jordan,
2024), trained on CIFAR-10 with a Wide ResNet 28-10.

## Research question

> How does the batch size influence the performance of Muon per training step?
> How does the momentum parameter interact with that?

To answer this we sweep over a grid of `(batch_size, momentum)` configurations,
log test-set metrics throughout training, and compare:

- learning curves (test accuracy vs. step / samples seen / epoch),
- final and best test accuracy across the grid,
- sample efficiency (steps / samples / epochs to reach a target accuracy).

## Files

| File              | Purpose                                                       |
| ----------------- | ------------------------------------------------------------- |
| `data.py`         | CIFAR-10 loader (from local `cifar-10-batches-py/`) + augment |
| `model.py`        | Wide ResNet 28-10 (~36.5 M parameters)                        |
| `muon.py`         | Muon optimizer (Newton-Schulz orthogonalization) + param split |
| `train.py`        | Single-config training loop with per-step CSV logging         |
| `sweep.py`        | Grid driver over `(batch_size, momentum, seed)`               |
| `plot.py`         | Curves, heatmap, and sample-efficiency plots                  |
| `requirements.txt`| Python dependencies                                           |

Non-2D parameters (biases, BatchNorm, stem conv, final classifier) are trained
with AdamW; Muon updates only the hidden 2D / 4D weight matrices, as in the
original paper.

## Setup

```bash
pip install -r requirements.txt
```

The CIFAR-10 pickle files are expected in `cifar-10-batches-py/` next to the
scripts (already present in this repository).

## Usage

### Single training run

```bash
python train.py --batch-size 128 --momentum 0.95 --epochs 10 \
    --log-csv runs/bs128_m0p95.csv
```

Use `--steps N` instead of `--epochs E` if you prefer a fixed optimizer-step
budget. When `--epochs` is given, smaller batch sizes mechanically run more
steps for the same number of passes over the data.

### Full sweep

```bash
python sweep.py --epochs 10 \
    --batch-sizes 64,128,256,512 \
    --momenta 0.8,0.9,0.95,0.98 \
    --out-dir runs/
```

The driver skips configurations whose CSV already exists, so the sweep is
resumable after interruptions. Use `--rerun` to overwrite, `--dry-run` to
preview commands, `--seeds 0,1,2` for multiple seeds.

### Plotting

```bash
python plot.py --runs-dir runs --out-dir figs --x-axis epoch --target 0.85
```

`--x-axis` controls the curve plots: `step` (default), `samples_seen`, or
`epoch`. **Use `epoch` or `samples_seen` when comparing batch sizes trained for
the same number of epochs** — plotting against `step` unfairly favors small
batches because they take more optimizer updates per epoch.

## Hardware requirements

Training was developed on an **NVIDIA GeForce GTX 1660 Ti Mobile (6 GB VRAM)**.
WRN-28-10 in channels-last format fits comfortably at small/medium batch
sizes; the largest batch sizes need more VRAM.

| Batch size | Approx. peak VRAM | 
| ---------: | ----------------: |
|         64 |             ~1 GB |
|        128 |           ~1.6 GB |
|        256 |           ~2.7 GB |
|        512 |           ~5.0 GB |
|       1024 |          ~9-10 GB |     

### Software

- Python ≥ 3.10
- PyTorch ≥ 2.1 with CUDA (developed against 2.12 + CUDA 13)
- numpy, pandas, matplotlib, torchvision (see `requirements.txt`)

## Notes on the Muon implementation

PyTorch does not ship Muon in `torch.optim` (as of early 2026). Third-party
implementations exist (`pytorch-optimizer` package, Keller Jordan's reference
repo), but for a *study of* the optimizer it is more transparent to keep the
implementation in-repo (`muon.py`, ~50 lines). This makes the Newton-Schulz
iteration, the Nesterov path, and the RMS scaling visible — and therefore
swappable — when interpreting sweep results.
