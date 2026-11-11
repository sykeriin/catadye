"""Per-sub-block update periods.

A sub-block at depth i of every MLP chain gets period periods[i]. Modes:

- "none":      every sub-block updates every step (baseline).
- "cms":       sub-block i updates every periods[i] steps using the mean of its accumulated
               gradients (gradients are left to accumulate between updates).
- "lr_matched": every sub-block updates every step, but with lr scaled by 1/periods[i]. Same blocks,
               same smaller effective step, no staggering. Separates "frequency" from "step size".

Everything outside the MLP chains (embeddings, attention, norms) always updates every step.
"""
from __future__ import annotations

import re

import torch

_SUB = re.compile(r"blocks\.\d+\.mlp\.subs\.(\d+)\.")


def sub_index(name: str) -> int | None:
    m = _SUB.search(name)
    return int(m.group(1)) if m else None


class StaggeredOptimizer:
    """Wraps one torch optimizer per period group."""

    def __init__(self, model, mode: str, periods=(1, 8, 64), lr: float = 3e-4, weight_decay: float = 0.1,
                 betas=(0.9, 0.95), optimizer_cls=torch.optim.AdamW, optimizer_kwargs=None):
        assert mode in ("none", "cms", "lr_matched")
        self.mode, self.periods = mode, tuple(periods)
        self.base_lr = lr
        okw = dict(betas=betas, weight_decay=weight_decay, **(optimizer_kwargs or {}))
        groups: dict[int, list] = {}
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            si = sub_index(n)
            period = 1 if (si is None or mode == "none") else self.periods[si]
            groups.setdefault(period, []).append((p, si))
        self.groups = []
        for period, items in sorted(groups.items()):
            params = [p for p, _ in items]
            if mode == "cms":
                eff_period, lr_scale = period, 1.0
            elif mode == "lr_matched":
                eff_period, lr_scale = 1, 1.0 / period
            else:
                eff_period, lr_scale = 1, 1.0
            opt = optimizer_cls(params, lr=lr * lr_scale, **okw)
            self.groups.append(dict(period=eff_period, lr_scale=lr_scale, params=params, opt=opt, accum=0))

    def set_lr(self, lr: float):
        for g in self.groups:
            for pg in g["opt"].param_groups:
                pg["lr"] = lr * g["lr_scale"]

    def step(self, step: int, clip: float | None = 1.0):
        """step is 1-indexed. Updates groups whose period divides `step`; others keep accumulating."""
        updated = []
        for g in self.groups:
            g["accum"] += 1
            if step % g["period"] != 0:
                continue
            n = g["accum"]
            if n > 1:
                for p in g["params"]:
                    if p.grad is not None:
                        p.grad.div_(n)
            if clip:
                torch.nn.utils.clip_grad_norm_(g["params"], clip)
            g["opt"].step()
            for p in g["params"]:
                p.grad = None
            g["accum"] = 0
            updated.append(g["period"])
        return updated

    def fraction_updated_per_step(self, model) -> float:
        """Average fraction of parameters updated per step (for overhead reporting)."""
        total = sum(p.numel() for p in model.parameters())
        return sum(sum(p.numel() for p in g["params"]) / g["period"] for g in self.groups) / total
