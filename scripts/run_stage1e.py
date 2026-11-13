"""Stage 1e: stable slow momentum (slow_norm) in the 2x2, lr 6e-4, order A, seeds 0-4 (docs/PREREGISTRATION.md).
Baseline and CMS arms reuse results/pilot_rev (same recipe). Only the slow-momentum arms run here. Skips finished runs.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from catadye.train import load_cfg, run  # noqa: E402

base = load_cfg(ROOT / "configs" / "pilot_rev.yaml")
D = ["tinystories", "python"]
for seed in (0, 1, 2, 3, 4):
    for name, o in [("slowmom_norm", dict(mode="none", slowmom=True, slow_norm=True, lr=6e-4)),
                    ("cms_slowmom_norm", dict(mode="cms", slowmom=True, slow_norm=True, lr=6e-4))]:
        res = run(dict(base, name=f"s1e_{name}", seed=seed, domains=D, **o), ROOT / "results" / "stage1e")
        print(name, seed, np.round(np.array(res["R"]), 3).tolist(), f"{res['wall_s']/60:.1f}min", flush=True)
