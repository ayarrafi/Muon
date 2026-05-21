"""Muon optimizer (Keller Jordan, 2024).

Newton-Schulz orthogonalization of momentum-buffered 2D gradients.
Use Muon for hidden 2D (and 4D conv) weights; pair with AdamW for everything else
(biases, normalization params, input embedding / stem, final classifier).

Reference: https://kellerjordan.github.io/posts/muon/
"""
from __future__ import annotations

import torch
from torch.optim import Optimizer


@torch.no_grad()
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Quintic Newton-Schulz iteration approximating G -> U S^0 V^T (orthogonal factor)."""
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.to(torch.bfloat16)
    X = X / (X.norm() + eps)
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class Muon(Optimizer):
    """Muon: momentum SGD + per-step Newton-Schulz orthogonalization.

    Only pass 2D (or 4D conv) hidden weights here. Conv weights of shape
    (C_out, C_in, kh, kw) are reshaped to (C_out, C_in*kh*kw) for the NS step.
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
    ):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                assert g.ndim in (2, 4), f"Muon expects 2D/4D params, got shape {tuple(p.shape)}"

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                update = g.add(buf, alpha=momentum) if nesterov else buf

                if update.ndim == 4:
                    update_2d = update.reshape(update.size(0), -1)
                else:
                    update_2d = update
                ortho = zeropower_via_newtonschulz5(update_2d, steps=ns_steps)
                ortho = ortho.reshape(p.shape)

                # RMS-preserving scale; see Keller's nanogpt-muon reference.
                fan_out, fan_in = p.size(0), p.numel() // p.size(0)
                scale = max(1.0, (fan_out / fan_in) ** 0.5)

                if wd != 0.0:
                    p.mul_(1 - lr * wd)
                p.add_(ortho, alpha=-lr * scale)

        return loss


def split_params_for_muon(model: torch.nn.Module) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """Split a model's parameters into (muon_params, adamw_params).

    Muon group: hidden 2D linear weights and 4D conv weights, excluding the conv stem
    and the final classifier (which are treated as input/output embeddings).
    AdamW group: everything else (biases, BN, stem conv, final linear).
    """
    muon, adamw = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # The stem conv and final classifier are treated as input/output embeddings -> AdamW.
        if name.startswith("conv1.") or name.startswith("fc."):
            adamw.append(p)
        elif p.ndim >= 2:
            muon.append(p)
        else:
            adamw.append(p)
    return muon, adamw
