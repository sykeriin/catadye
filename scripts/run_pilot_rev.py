"""Pilot revision 1: 3 arms x 3 lrs x 3 seeds (order A) + 3 arms x 3 seeds at the middle lr (order B).
Seed-major so an interrupted run still leaves complete seeds. Skips finished runs.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from catadye.data import build_domain  # noqa: E402
from catadye.train import load_cfg, run  # noqa: E402

LRS = [3.0e-4, 6.0e-4, 1.2e-3]
ARMS = ["none", "cms", "lr_matched"]
ORDERS = {"A": ["tinystories", "python"], "B": ["python", "tinystories"]}

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("--steps", type=int, default=None)
ap.add_argument("--out", default=str(ROOT / "results" / "pilot_rev"))
args = ap.parse_args()

base = load_cfg(ROOT / "configs" / "pilot_rev.yaml")
if args.steps:
    base["steps_per_domain"] = args.steps
need = base["steps_per_domain"] * base["batch"] * base["model"]["ctx"]
for d in ORDERS["A"]:
    build_domain(d, n_train=int(need * 1.1), n_eval=300_000)

jobs = []
for seed in args.seeds:
    for lr in LRS:
        for arm in ARMS:
            jobs.append((seed, "A", lr, arm))
for seed in args.seeds:
    for arm in ARMS:
        jobs.append((seed, "B", 6.0e-4, arm))

for seed, order, lr, arm in jobs:
    cfg = dict(base, name=f"rev_{arm}_lr{lr:g}_{order}", mode=arm, seed=seed, lr=lr, domains=ORDERS[order], order=order)
    res = run(cfg, args.out)
    R = np.array(res["R"])
    print(f"{cfg['name']} s{seed}: params={res['n_params']/1e6:.1f}M sub_share={res['sub_param_share']:.2f} "
          f"R={np.round(R, 3).tolist()} {res['wall_s']/60:.1f}min {res['tokens_per_s']:.0f} tok/s", flush=True)
