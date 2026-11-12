"""Post-hoc (exploratory, NOT pre-registered) diagnostics: where does forgetting live?

1. Drift: relative weight movement ||theta_d2 - theta_d1|| / ||theta_d1|| per parameter group.
2. Revert test: start from the model after domain 2 and revert one group at a time to its domain-1
   weights; report old-domain (TinyStories) loss recovered and new-domain (Python) loss lost.
Needs checkpoints from scripts/run_ckpts.py.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from catadye.data import eval_windows, load_domain  # noqa: E402
from catadye.model import GPT, GPTConfig  # noqa: E402
from catadye.train import evaluate, load_cfg  # noqa: E402

dev = "cuda" if torch.cuda.is_available() else "cpu"
cfg = load_cfg(ROOT / "configs" / "pilot_rev.yaml")
mc = GPTConfig(**cfg["model"])
ev = {d: eval_windows(load_domain(d)[1], mc.ctx, cfg["eval_windows"], dev) for d in ("tinystories", "python")}


def group(name: str) -> str:
    if name in ("wte.weight", "wpe.weight", "lm_head.weight"):
        return "embeddings"
    m = re.search(r"mlp\.subs\.(\d)\.(up|down)", name)
    if m:
        return f"mlp_sub{m.group(1)}"
    if ".attn." in name:
        return "attention"
    return "layernorm"


GROUPS = ["embeddings", "attention", "layernorm", "mlp_sub0", "mlp_sub1", "mlp_sub2"]
REVERTS = {g: [g] for g in GROUPS}
REVERTS["all_mlp"] = ["mlp_sub0", "mlp_sub1", "mlp_sub2"]
REVERTS["all_non_mlp"] = ["embeddings", "attention", "layernorm"]


def load(mode, seed, d):
    sd = torch.load(ROOT / "checkpoints" / f"{mode}_s{seed}_d{d}.pt", map_location=dev, weights_only=True)
    m = GPT(mc).to(dev)
    m.load_state_dict(sd)
    return m, sd


def analyze(mode, seed):
    m, sd2 = load(mode, seed, 2)
    _, sd1 = load(mode, seed, 1)
    names = [n for n, _ in m.named_parameters()]
    drift = {}
    for g in GROUPS:
        num = den = 0.0
        for n in names:
            if group(n) == g:
                num += (sd2[n].float() - sd1[n].float()).pow(2).sum().item()
                den += sd1[n].float().pow(2).sum().item()
        drift[g] = float(np.sqrt(num / den))
    old = lambda mm: evaluate(mm, ev["tinystories"], dev)  # noqa: E731
    new = lambda mm: evaluate(mm, ev["python"], dev)  # noqa: E731
    m1, _ = load(mode, seed, 1)
    ref = dict(d1_old=old(m1), d1_new=new(m1), d2_old=old(m), d2_new=new(m))
    forgot = ref["d2_old"] - ref["d1_old"]
    rev = {}
    for key, gs in REVERTS.items():
        mm, _ = load(mode, seed, 2)
        with torch.no_grad():
            for n, p in mm.named_parameters():
                if group(n) in gs:
                    p.copy_(sd1[n])
        o, nw = old(mm), new(mm)
        rev[key] = dict(old=o, new=nw, recovered=(ref["d2_old"] - o) / forgot, new_cost=nw - ref["d2_new"])
    return dict(mode=mode, seed=seed, drift=drift, ref=ref, forgetting=forgot, revert=rev)


if __name__ == "__main__":
    out = []
    for f in sorted((ROOT / "checkpoints").glob("*_d2.pt")):
        mode, s, _ = f.stem.rsplit("_", 2)
        out.append(analyze(mode, int(s[1:])))
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "diagnostics.json").write_text(json.dumps(out, indent=1))
    for r in out:
        print(f"\n== {r['mode']} s{r['seed']}: forgetting {r['forgetting']:.3f} nats "
              f"(old {r['ref']['d1_old']:.3f} -> {r['ref']['d2_old']:.3f}; new {r['ref']['d1_new']:.3f} -> {r['ref']['d2_new']:.3f})")
        print("  drift:", {k: round(v, 3) for k, v in r["drift"].items()})
        for k, v in r["revert"].items():
            print(f"  revert {k:12s}: old-loss recovered {v['recovered']:+6.1%}  new-loss cost {v['new_cost']:+.3f}")
