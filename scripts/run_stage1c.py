"""Stage 1c runner: replay, EWC, slow momentum, CMS+slow momentum (see docs/PREREGISTRATION.md). Seed-major; skips finished."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from catadye.train import load_cfg, run  # noqa: E402

base = load_cfg(ROOT / "configs" / "pilot_rev.yaml")
D = ["tinystories", "python"]
jobs = []
for seed in (0, 1, 2):
    jobs.append((seed, "slowmom", dict(mode="none", slowmom=True, lr=6e-4)))
    jobs.append((seed, "cms_slowmom", dict(mode="cms", slowmom=True, lr=6e-4)))
for seed in (0, 1, 2):
    for lam in (10, 1000):
        jobs.append((seed, f"ewc{lam}", dict(mode="none", ewc_lambda=lam, lr=6e-4)))
for seed in (0, 1, 2):
    for lr in (3e-4, 6e-4, 1.2e-3):
        jobs.append((seed, f"replay_lr{lr:g}", dict(mode="none", replay_seqs=1, lr=lr)))
for seed, name, o in jobs:
    res = run(dict(base, name=f"s1c_{name}", seed=seed, domains=D, **o), ROOT / "results" / "stage1c")
    print(name, seed, np.round(np.array(res["R"]), 3).tolist(), f"{res['wall_s']/60:.1f}min", flush=True)
