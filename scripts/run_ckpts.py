"""Train models with the revised-pilot recipe (order A, lr 6e-4) and save a checkpoint after each domain.
Used for (1) the Stage 1b base models (mode none, seeds 0-2, *_d1.pt) and (2) the post-hoc drift /
revert diagnostics (all modes, *_d1.pt and *_d2.pt). Same config and seed as pilot_rev, so the loss
matrix should reproduce results/pilot_rev/rev_<mode>_lr0.0006_A_s<seed>.json (consistency check).
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from catadye.train import load_cfg, run  # noqa: E402

JOBS = [("none", 0), ("cms", 0), ("lr_matched", 0), ("none", 1), ("none", 2)]
base = load_cfg(ROOT / "configs" / "pilot_rev.yaml")
for mode, seed in JOBS:
    cfg = dict(base, name=f"ckpt_{mode}", mode=mode, seed=seed, lr=6.0e-4, ckpt_dir=str(ROOT / "checkpoints"),
               order="A")
    res = run(cfg, ROOT / "results" / "ckpt_runs")
    ref = ROOT / "results" / "pilot_rev" / f"rev_{mode}_lr0.0006_A_s{seed}.json"
    msg = ""
    if ref.exists():
        import json
        d = np.abs(np.array(json.loads(ref.read_text())["R"]) - np.array(res["R"])).max()
        msg = f" | max |R - pilot_rev R| = {d:.4f}"
    print(f"{mode} s{seed}: R={np.round(res['R'], 3).tolist()} {res['wall_s']/60:.1f}min{msg}", flush=True)
