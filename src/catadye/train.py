"""Sequential-domain training. Writes one JSON per run with the full loss matrix and cost stats."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from .data import Batcher, eval_windows, load_domain
from .model import GPT, GPTConfig
from .ewc import ewc_grad_add, fisher_diag
from .slowmom import SlowMomAdamW
from .staggered import StaggeredOptimizer


@torch.no_grad()
def evaluate(model, windows, device, bs=8):
    model.eval()
    xs, ys = windows
    tot, n = 0.0, 0
    for i in range(0, len(xs), bs):
        with torch.autocast(device, dtype=torch.bfloat16, enabled=(device == "cuda")):
            _, loss = model(xs[i : i + bs], ys[i : i + bs])
        tot += loss.item() * len(xs[i : i + bs])
        n += len(xs[i : i + bs])
    model.train()
    return tot / n


def lr_at(step, total, base, warmup, min_frac=0.1):
    if step < warmup:
        return base * (step + 1) / warmup
    prog = (step - warmup) / max(1, total - warmup)
    return base * (min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * prog)))


def run(cfg: dict, out_dir: str | Path = "results") -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{cfg['name']}_s{cfg['seed']}"
    out_path = out_dir / f"{name}.json"
    if out_path.exists():
        return json.loads(out_path.read_text())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    mc = GPTConfig(**cfg.get("model", {}))
    model = GPT(mc).to(device)
    def make_opt():
        return StaggeredOptimizer(model, mode=cfg["mode"], periods=cfg.get("periods", (1, 8, 64)), lr=cfg["lr"],
                                  weight_decay=cfg.get("weight_decay", 0.1),
                                  **({"optimizer_cls": SlowMomAdamW, "optimizer_kwargs": dict(
                                      slow_beta=cfg.get("slow_beta", 0.999), alpha=cfg.get("slow_alpha", 1.0),
                                      slow_norm=cfg.get("slow_norm", False))}
                                     if cfg.get("slowmom") else {}))
    so = make_opt()
    domains = cfg["domains"]
    data = {d: load_domain(d) for d in domains}
    ev = {d: eval_windows(data[d][1], mc.ctx, cfg.get("eval_windows", 64), device) for d in domains}
    T = len(domains)

    R0 = np.array([evaluate(model, ev[d], device) for d in domains])  # untrained model
    R = np.zeros((T, T))
    steps, bs = cfg["steps_per_domain"], cfg["batch"]
    log, tokens, t0 = [], 0, time.time()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    gstep = 0
    ewc_state = None
    for i, d in enumerate(domains):
        micro = cfg.get("micro_batches", 1)  # same gradient, lower peak memory (4 GB card)
        assert bs % micro == 0
        batcher = Batcher(data[d][0], mc.ctx, bs // micro, seed=cfg["seed"] * 1000 + i, device=device)
        if i > 0 and cfg.get("reset_opt"):  # post-hoc option: fresh optimizer state at the domain boundary
            so = make_opt()
        ewc = ewc_state if (i > 0 and cfg.get("ewc_lambda")) else None
        replay = Batcher(data[domains[0]][0], mc.ctx, bs // micro, seed=cfg["seed"] * 1000 + 500 + i, device=device)             if (i > 0 and cfg.get("replay_seqs")) else None
        for s in range(steps):
            gstep += 1
            so.set_lr(lr_at(s, steps, cfg["lr"], cfg.get("warmup", 100)))
            for _ in range(micro):
                x, y = batcher.get()
                if replay is not None:  # replace the first k sequences of every micro-batch with old-domain ones
                    rx, ry = replay.get()
                    k = cfg["replay_seqs"]
                    x, y = torch.cat([rx[:k], x[k:]]), torch.cat([ry[:k], y[k:]])
                with torch.autocast(device, dtype=torch.bfloat16, enabled=(device == "cuda")):
                    _, loss = model(x, y)
                (loss / micro).backward()
                if ewc is not None:
                    ewc_grad_add(model, ewc[0], ewc[1], cfg["ewc_lambda"] / micro)
                tokens += x.numel()
            so.step(gstep)
            if (s + 1) % 200 == 0:
                log.append(dict(domain=d, step=s + 1, loss=loss.item()))
        for j, dj in enumerate(domains):
            R[i, j] = evaluate(model, ev[dj], device)
        if i == 0 and cfg.get("ewc_lambda"):
            fb = Batcher(data[d][0], mc.ctx, 8, seed=cfg["seed"] * 1000 + 900, device=device)
            ewc_state = (fisher_diag(model, fb, 64, device), {n: p.detach().clone() for n, p in model.named_parameters()})
        if cfg.get("ckpt_dir"):
            ck = Path(cfg["ckpt_dir"])
            ck.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ck / f"{cfg['mode']}_s{cfg['seed']}_d{i + 1}.pt")
    wall = time.time() - t0
    res = dict(
        name=cfg["name"], seed=cfg["seed"], mode=cfg["mode"], domains=domains, cfg=cfg,
        n_params=model.n_params(),
        sub_param_share=sum(p.numel() for n, p in model.named_parameters() if ".mlp.subs." in n) / model.n_params(),
        R0=R0.tolist(), R=R.tolist(), wall_s=wall, tokens=tokens,
        tokens_per_s=tokens / wall, frac_params_updated_per_step=so.fraction_updated_per_step(model),
        peak_vram_mb=(torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else None), log=log,
    )
    out_path.write_text(json.dumps(res))
    return res


def load_cfg(path: str | Path, **overrides) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    cfg.update(overrides)
    return cfg
