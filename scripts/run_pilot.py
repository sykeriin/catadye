"""Stage 0 pilot: 3 arms x 3 seeds. Skips finished runs (results/*.json), so it is safe to rerun.

Usage: python scripts/run_pilot.py [--steps N] [--seeds 0 1 2] [--arms none cms lr_matched]
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from catadye.data import build_domain  # noqa: E402
from catadye.train import load_cfg, run  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=None)
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("--arms", nargs="+", default=["none", "cms", "lr_matched"])
ap.add_argument("--out", default=str(ROOT / "results" / "pilot"))
args = ap.parse_args()

base = load_cfg(ROOT / "configs" / "pilot.yaml")
if args.steps:
    base["steps_per_domain"] = args.steps
need = base["steps_per_domain"] * base["batch"] * base["model"]["ctx"]
for d in base["domains"]:
    build_domain(d, n_train=int(need * 1.1), n_eval=300_000)

for seed in args.seeds:
    for arm in args.arms:
        cfg = dict(base, name=f"pilot_{arm}", mode=arm, seed=seed)
        res = run(cfg, args.out)
        import numpy as np
        R = np.array(res["R"])
        print(f"{cfg['name']} s{seed}: params={res['n_params']/1e6:.1f}M "
              f"R={np.round(R, 3).tolist()} {res['wall_s']/60:.1f}min {res['tokens_per_s']:.0f} tok/s "
              f"vram={res['peak_vram_mb']}", flush=True)
