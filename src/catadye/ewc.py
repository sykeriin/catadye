"""Diagonal-Fisher EWC. Fisher = mean squared gradient of the LM loss over batches of the old domain,
normalised to mean 1 so lambda is comparable across models."""
import torch


def fisher_diag(model, batcher, n_batches, device):
    F = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    model.train()
    for _ in range(n_batches):
        x, y = batcher.get()
        model.zero_grad(set_to_none=True)
        with torch.autocast(device, dtype=torch.bfloat16, enabled=(device == "cuda")):
            _, loss = model(x, y)
        loss.backward()
        for n, p in model.named_parameters():
            if p.grad is not None:
                F[n] += p.grad.detach().float() ** 2 / n_batches
    model.zero_grad(set_to_none=True)
    mean = sum(f.sum() for f in F.values()) / sum(f.numel() for f in F.values())
    return {n: f / mean.clamp_min(1e-30) for n, f in F.items()}


def ewc_grad_add(model, F, anchor, lam):
    """Adds the gradient of (lam/2) * sum F (theta - anchor)^2 to p.grad (call after backward)."""
    for n, p in model.named_parameters():
        if p.grad is not None:
            p.grad.add_(lam * F[n] * (p.detach() - anchor[n]))
