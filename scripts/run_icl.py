"""In-context baseline (eval only, not pre-registered): the domain-1 (TinyStories) model is frozen; Python text is
supplied as context. Compares Python loss on the same 128-token targets with (a) no context and (b) a 128-token
Python prefix. Weights never change, so forgetting of TinyStories is zero by construction.
Uses the none-arm domain-1 checkpoints (seeds 0-2, lr 6e-4, pilot_rev recipe).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from catadye.data import load_domain  # noqa: E402
from catadye.model import GPT, GPTConfig  # noqa: E402

base = yaml.safe_load((ROOT / "configs" / "pilot_rev.yaml").read_text())
mc = GPTConfig(**base["model"])
device = "cuda" if torch.cuda.is_available() else "cpu"
_, py_eval = load_domain("python")
arr = py_eval.astype(np.int64)
P = T = 128
starts = [k * 512 for k in range(64) if k * 512 + P + T <= len(arr)]
out = {}
for seed in (0, 1, 2):
    model = GPT(mc).to(device)
    model.load_state_dict(torch.load(ROOT / "checkpoints" / f"none_s{seed}_d1.pt", map_location=device))
    model.eval()
    nc, icl = [], []
    with torch.no_grad():
        for a in starts:
            tgt = torch.tensor(arr[a + P : a + P + T], device=device)[None]
            pre = torch.tensor(arr[a : a + P], device=device)[None]
            # (a) no context: score the target from its own start
            lg = model(tgt[:, :-1])
            nc.append(torch.nn.functional.cross_entropy(lg[0], tgt[0, 1:]).item())
            # (b) 128-token Python prefix as context: score only the target positions
            seq = torch.cat([pre, tgt], dim=1)
            lg = model(seq[:, :-1])
            icl.append(torch.nn.functional.cross_entropy(lg[0, P - 1 :], seq[0, P:]).item())
    out[seed] = dict(no_context=float(np.mean(nc)), with_python_context=float(np.mean(icl)), n_windows=len(starts))
    print(seed, out[seed], flush=True)
(ROOT / "results" / "icl").mkdir(parents=True, exist_ok=True)
(ROOT / "results" / "icl" / "icl_python.json").write_text(json.dumps(out, indent=1))
