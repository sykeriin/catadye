import copy

import numpy as np
import torch

from catadye.metrics import final_avg, forgetting, forward_transfer, holm, paired_bootstrap_ci
from catadye.model import GPT, GPTConfig
from catadye.staggered import StaggeredOptimizer, sub_index

CFG = GPTConfig(vocab_size=100, n_layer=2, n_head=2, d_model=32, ctx=16, n_sub=3)


def _batch():
    g = torch.Generator().manual_seed(0)
    x = torch.randint(0, 100, (4, 16), generator=g)
    return x, torch.roll(x, -1, 1)


def _snapshot(model):
    return {n: p.detach().clone() for n, p in model.named_parameters()}


def _step(model, so, step):
    x, y = _batch()
    _, loss = model(x, y)
    loss.backward()
    so.step(step)


def test_param_counts_identical_across_modes():
    counts = {m: GPT(CFG).n_params() for m in ("none", "cms", "lr_matched")}
    assert len(set(counts.values())) == 1


def test_period_k_block_changes_only_on_multiples():
    torch.manual_seed(0)
    model = GPT(CFG)
    so = StaggeredOptimizer(model, "cms", periods=(1, 4, 8), lr=1e-2)
    prev = _snapshot(model)
    for step in range(1, 17):
        _step(model, so, step)
        cur = _snapshot(model)
        for n in cur:
            si = sub_index(n)
            period = 1 if si is None else (1, 4, 8)[si]
            changed = not torch.equal(cur[n], prev[n])
            assert changed == (step % period == 0), (n, step)
        prev = cur


def test_accumulated_grad_is_mean_of_per_step_grads():
    torch.manual_seed(0)
    model = GPT(CFG)
    sub2 = [p for n, p in model.named_parameters() if sub_index(n) == 2]
    ref = []
    for _ in range(4):
        x, y = _batch()
        _, loss = model(x, y)
        loss.backward()
        ref.append([p.grad.clone() for p in sub2])
        model.zero_grad()
    # same params, no update in between => accumulated grad / 4 equals mean of per-step grads
    seen = {}

    class Spy(torch.optim.AdamW):
        def step(self, *a, **k):
            seen.update({id(p): p.grad.clone() for g in self.param_groups for p in g["params"]})
            return super().step(*a, **k)

    so = StaggeredOptimizer(model, "cms", periods=(1, 4, 4), lr=0.0, weight_decay=0.0, optimizer_cls=Spy)
    for step in range(1, 5):
        x, y = _batch()
        _, loss = model(x, y)
        loss.backward()
        so.step(step, clip=None)
    for k, p in enumerate(sub2):
        mean = sum(r[k] for r in ref) / 4
        torch.testing.assert_close(seen[id(p)], mean, rtol=1e-4, atol=1e-6)


def test_lr_matched_updates_every_step_with_scaled_lr():
    model = GPT(CFG)
    so = StaggeredOptimizer(model, "lr_matched", periods=(1, 8, 64), lr=1e-2)
    assert all(g["period"] == 1 for g in so.groups)
    scales = sorted(g["lr_scale"] for g in so.groups)
    assert scales == [1 / 64, 1 / 8, 1.0]


def test_none_mode_updates_everything_every_step():
    model = GPT(CFG)
    so = StaggeredOptimizer(model, "none", lr=1e-2)
    prev = _snapshot(model)
    _step(model, so, 1)
    cur = _snapshot(model)
    assert all(not torch.equal(cur[n], prev[n]) for n in cur)


def test_overhead_fraction():
    model = GPT(CFG)
    f_none = StaggeredOptimizer(model, "none").fraction_updated_per_step(model)
    f_cms = StaggeredOptimizer(model, "cms", periods=(1, 8, 64)).fraction_updated_per_step(model)
    assert f_none == 1.0 and f_cms < f_none


def test_metrics():
    R = np.array([[1.0, 5.0], [2.0, 1.5]])
    assert forgetting(R) == 1.0          # domain 0: 2.0 - 1.0
    assert final_avg(R) == 1.75
    assert forward_transfer(R, np.array([6.0, 6.0])) == 1.0   # 6.0 - R[0,1]=5.0
    m, lo, hi = paired_bootstrap_ci([2, 2, 2], [1, 1, 1], n_boot=200)
    assert m == 1.0 and lo == hi == 1.0
    adj = holm([0.01, 0.04, 0.03])
    assert adj[0] == 0.03 and adj.max() <= 1.0
