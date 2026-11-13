"""Joint-training upper bound: both domains interleaved, same lr (6e-4), same total steps (2 x 2000), seeds 0-4.
Not pre-registered (see docs/RESULTS.md). Skips finished runs.
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
    res = run(dict(base, name="joint", seed=seed, domains=D, mode="none", joint=True, lr=6e-4),
              ROOT / "results" / "joint")
    print("joint", seed, np.round(np.array(res["joint_final"]), 3).tolist(), f"{res['wall_s']/60:.1f}min", flush=True)
