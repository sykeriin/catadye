"""Stage 1b: test-time updates of the MLP sub-blocks on a token stream (see README declaration).

Each step: take 4 consecutive 256-token windows of the stream, score them with the current weights
(online loss), then do one (possibly staggered) update. Only MLP sub-blocks are trainable.
"""
from __future__ import annotations

import numpy as np
import torch

from .staggered import StaggeredOptimizer
from .train import evaluate

ARM_MODE = {"frozen": None, "ttt_single": "none", "ttt_cms": "cms", "ttt_lrm": "lr_matched"}


def freeze_all_but_mlp(model):
    for n, p in model.named_parameters():
        p.requires_grad = ".mlp.subs." in n and (".up." in n or ".down." in n or ".ln." in n)


def stream_batch(arr, step, bs_windows, ctx, device):
    """Step is 1-indexed; windows are consecutive and non-overlapping across the whole stream."""
    base = (step - 1) * bs_windows * ctx
    xs = np.stack([arr[base + j * ctx : base + (j + 1) * ctx] for j in range(bs_windows)]).astype(np.int64)
    ys = np.stack([arr[base + j * ctx + 1 : base + (j + 1) * ctx + 1] for j in range(bs_windows)]).astype(np.int64)
    return torch.from_numpy(xs).to(device), torch.from_numpy(ys).to(device)


def run_ttt(model, stream, old_ev, new_ev, arm, lr, device, periods=(1, 8, 64), n_steps=1000, bs_windows=4,
            ctx=256, eval_every=250):
    mode = ARM_MODE[arm]
    freeze_all_but_mlp(model)
    so = StaggeredOptimizer(model, mode, periods=periods, lr=lr, weight_decay=0.0) if mode else None
    curve, online = [], []

    def snap(step):
        curve.append(dict(step=step, old=evaluate(model, old_ev, device), new=evaluate(model, new_ev, device)))

    snap(0)
    for s in range(1, n_steps + 1):
        x, y = stream_batch(stream, s, bs_windows, ctx, device)
        with torch.autocast(device, dtype=torch.bfloat16, enabled=(device == "cuda")), \
                torch.set_grad_enabled(so is not None):
            _, loss = model(x, y)
        online.append(loss.item())
        if so is not None:
            loss.backward()
            so.step(s)
        if s % eval_every == 0:
            snap(s)
    return dict(curve=curve, online=online)
