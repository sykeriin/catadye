"""Stage 1b: 3 ttt arms x 3 lrs x 3 seeds + frozen x 3 seeds, from the *_d1.pt TinyStories-only bases.
Skips finished runs. Results: results/ttt/<arm>_lr<lr>_s<seed>.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from catadye.data import eval_windows, load_domain  # noqa: E402
from catadye.model import GPT, GPTConfig  # noqa: E402
from catadye.testtime import run_ttt  # noqa: E402
from catadye.train import load_cfg  # noqa: E402

dev = "cuda" if torch.cuda.is_available() else "cpu"
cfg = load_cfg(ROOT / "configs" / "pilot_rev.yaml")
mc = GPTConfig(**cfg["model"])
old_ev = eval_windows(load_domain("tinystories")[1], mc.ctx, cfg["eval_windows"], dev)
py_train, py_eval = load_domain("python")
new_ev = eval_windows(py_eval, mc.ctx, cfg["eval_windows"], dev)
N_STEPS, BS = 1000, 4
assert len(py_train) >= N_STEPS * BS * mc.ctx + 1, "python stream too short"
LRS = [1e-4, 3e-4, 1e-3]
out_dir = ROOT / "results" / "ttt"
if "--low" in sys.argv:  # Stage 1b-low: extra lower lrs, separate dir
    LRS = [1e-5, 3e-5]
    out_dir = ROOT / "results" / "ttt_low"
out_dir.mkdir(parents=True, exist_ok=True)

jobs = []
for seed in (0, 1, 2):
    if "--low" not in sys.argv:
        jobs.append(("frozen", 0.0, seed))
    for lr in LRS:
        for arm in ("ttt_single", "ttt_cms", "ttt_lrm"):
            jobs.append((arm, lr, seed))

for arm, lr, seed in jobs:
    f = out_dir / f"{arm}_lr{lr:g}_s{seed}.json"
    if f.exists():
        continue
    torch.manual_seed(seed)
    m = GPT(mc).to(dev)
    m.load_state_dict(torch.load(ROOT / "checkpoints" / f"none_s{seed}_d1.pt", map_location=dev, weights_only=True))
    r = run_ttt(m, py_train, old_ev, new_ev, arm, lr, dev, n_steps=N_STEPS, bs_windows=BS, ctx=mc.ctx)
    r.update(arm=arm, lr=lr, seed=seed)
    f.write_text(json.dumps(r))
    c = r["curve"]
    print(f"{arm} lr={lr:g} s{seed}: old {c[0]['old']:.3f}->{c[-1]['old']:.3f}  new {c[0]['new']:.3f}->{c[-1]['new']:.3f}  "
          f"online {np.mean(r['online']):.3f}", flush=True)
