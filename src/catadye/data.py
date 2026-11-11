"""Domain token streams, cached as uint16 .npy (GPT-2 BPE, vocab 50257 < 65536).

Each domain has a train array and a held-out eval array. Documents are separated by the EOT token.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# name -> (hf dataset, config, split, text field). Streamed; only the first documents are read.
DOMAINS = {
    "tinystories": ("roneneldan/TinyStories", None, "train", "text"),
    "python": ("codeparrot/codeparrot-clean-valid", None, "train", "content"),
    "wikitext": ("Salesforce/wikitext", "wikitext-103-raw-v1", "train", "text"),
}


def build_domain(name: str, n_train: int, n_eval: int, force: bool = False) -> None:
    """Stream documents, tokenize, and write {name}_train.npy / {name}_eval.npy."""
    import tiktoken
    from datasets import load_dataset

    DATA_DIR.mkdir(exist_ok=True)
    tr, ev = DATA_DIR / f"{name}_train.npy", DATA_DIR / f"{name}_eval.npy"
    if tr.exists() and ev.exists() and not force:
        return
    repo, cfg, split, field = DOMAINS[name]
    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token
    ds = load_dataset(repo, cfg, split=split, streaming=True)
    need = n_train + n_eval
    chunks, total = [], 0
    for ex in ds:
        text = ex[field]
        if not text or not text.strip():
            continue
        ids = enc.encode_ordinary(text) + [eot]
        chunks.append(np.asarray(ids, dtype=np.uint16))
        total += len(ids)
        if total >= need:
            break
    arr = np.concatenate(chunks)
    # eval = the *last* n_eval tokens, disjoint from train (split at document-stream level)
    np.save(tr, arr[:n_train])
    np.save(ev, arr[-n_eval:] if len(arr) >= need else arr[n_train:])


def load_domain(name: str):
    return np.load(DATA_DIR / f"{name}_train.npy"), np.load(DATA_DIR / f"{name}_eval.npy")


class Batcher:
    """Random contiguous windows from a token array, with a private RNG for reproducibility."""

    def __init__(self, arr: np.ndarray, ctx: int, batch: int, seed: int, device: str):
        self.arr, self.ctx, self.batch, self.device = arr, ctx, batch, device
        self.rng = np.random.default_rng(seed)

    def get(self):
        import torch

        hi = len(self.arr) - self.ctx - 1
        ix = self.rng.integers(0, hi, size=self.batch)
        x = np.stack([self.arr[i : i + self.ctx] for i in ix]).astype(np.int64)
        y = np.stack([self.arr[i + 1 : i + 1 + self.ctx] for i in ix]).astype(np.int64)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        if self.device == "cuda":
            return x.pin_memory().to("cuda", non_blocking=True), y.pin_memory().to("cuda", non_blocking=True)
        return x, y


def eval_windows(arr: np.ndarray, ctx: int, n_windows: int, device: str):
    """Deterministic, non-overlapping eval windows."""
    import torch

    n = min(n_windows, (len(arr) - 1) // ctx)
    xs = np.stack([arr[i * ctx : i * ctx + ctx] for i in range(n)]).astype(np.int64)
    ys = np.stack([arr[i * ctx + 1 : i * ctx + ctx + 1] for i in range(n)]).astype(np.int64)
    return torch.from_numpy(xs).to(device), torch.from_numpy(ys).to(device)
