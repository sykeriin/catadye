"""AdamW with an added slow momentum buffer (M3-inspired, without Newton-Schulz orthogonalisation).

update = (m_fast + alpha * m_slow) / (1 + alpha) / (sqrt(v) + eps), decoupled weight decay.
alpha = 0 reduces exactly to AdamW (tested).
"""
import torch


class SlowMomAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1, eps=1e-8, slow_beta=0.999, alpha=1.0):
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay, eps=eps,
                                      slow_beta=slow_beta, alpha=alpha))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            b1, b2 = g["betas"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if not st:
                    st["t"] = 0
                    st["m"], st["s"], st["v"] = (torch.zeros_like(p) for _ in range(3))
                st["t"] += 1
                t = st["t"]
                st["m"].mul_(b1).add_(p.grad, alpha=1 - b1)
                st["s"].mul_(g["slow_beta"]).add_(p.grad, alpha=1 - g["slow_beta"])
                st["v"].mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
                m = st["m"] / (1 - b1 ** t)
                s = st["s"] / (1 - g["slow_beta"] ** t)
                den = (st["v"] / (1 - b2 ** t)).sqrt().add_(g["eps"])
                p.mul_(1 - g["lr"] * g["weight_decay"])
                p.addcdiv_((m + g["alpha"] * s) / (1 + g["alpha"]), den, value=-g["lr"])
