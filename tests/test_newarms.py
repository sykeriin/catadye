import torch

from catadye.ewc import ewc_grad_add, fisher_diag
from catadye.slowmom import SlowMomAdamW


def test_slowmom_alpha0_equals_adamw():
    torch.manual_seed(0)
    a = torch.nn.Parameter(torch.randn(5, 3)); b = torch.nn.Parameter(a.detach().clone())
    o1 = torch.optim.AdamW([a], lr=1e-2, betas=(0.9, 0.95), weight_decay=0.1)
    o2 = SlowMomAdamW([b], lr=1e-2, betas=(0.9, 0.95), weight_decay=0.1, alpha=0.0)
    for _ in range(10):
        g = torch.randn(5, 3)
        a.grad = g.clone(); b.grad = g.clone(); o1.step(); o2.step()
    assert torch.allclose(a, b, atol=1e-6)


def test_slowmom_alpha_changes_update():
    a = torch.nn.Parameter(torch.ones(4)); b = torch.nn.Parameter(torch.ones(4))
    o1 = SlowMomAdamW([a], alpha=0.0); o2 = SlowMomAdamW([b], alpha=1.0)
    for i in range(5):
        g = torch.full((4,), 1.0 if i < 3 else -1.0); a.grad = g.clone(); b.grad = g.clone(); o1.step(); o2.step()
    assert not torch.allclose(a, b)


def test_ewc_zero_at_anchor_and_pulls_back():
    m = torch.nn.Linear(3, 1)
    anchor = {n: p.detach().clone() for n, p in m.named_parameters()}
    F = {n: torch.ones_like(p) for n, p in m.named_parameters()}
    for p in m.parameters():
        p.grad = torch.zeros_like(p)
    ewc_grad_add(m, F, anchor, 10.0)
    assert all((p.grad == 0).all() for p in m.parameters())
    with torch.no_grad():
        m.weight.add_(1.0)
    ewc_grad_add(m, F, anchor, 10.0)
    assert torch.allclose(m.weight.grad, torch.full_like(m.weight, 10.0))


def test_slow_norm_bounds_slow_term():
    # With slow_norm, |s| / sqrt(vs) <= 1 (Cauchy-Schwarz, same beta), so the slow term cannot outgrow
    # the fast denominator when gradients shrink (the failure of the unnormalised variant).
    from catadye.slowmom import SlowMomAdamW
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.zeros(50))
    opt = SlowMomAdamW([p], lr=1e-3, alpha=1.0, slow_norm=True)
    worst = 0.0
    for t in range(400):
        scale = 10.0 if t < 100 else 1e-3  # gradients shrink sharply, as after a domain switch
        p.grad = torch.randn(50) * scale
        opt.step()
        st = opt.state[p]
        s = st["s"] / (1 - 0.999 ** st["t"])
        vs = (st["vs"] / (1 - 0.999 ** st["t"])).sqrt() + 1e-8
        worst = max(worst, (s / vs).abs().max().item())
    assert worst <= 1.0 + 1e-4
