"""POST-HOC (not pre-registered): does resetting optimizer state at the domain boundary fix the slow-momentum instability?
Seed 0 only, lr 6e-4: slowmom, cms_slowmom, and plain AdamW as the control for the reset itself."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from catadye.train import load_cfg, run  # noqa: E402

base = load_cfg(ROOT / "configs" / "pilot_rev.yaml")
for name, o in [("slowmom_reset", dict(mode="none", slowmom=True)), ("cms_slowmom_reset", dict(mode="cms", slowmom=True)),
                ("none_reset", dict(mode="none"))]:
    r = run(dict(base, name=f"posthoc_{name}", seed=0, lr=6e-4, reset_opt=True, domains=["tinystories", "python"], **o),
            ROOT / "results" / "stage1c_posthoc")
    print(name, np.round(np.array(r["R"]), 3).tolist(), flush=True)
