"""Stage 1c readout as declared. Excess forgetting vs the baseline frontier interpolated at each arm's new-domain loss."""
import glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from catadye.metrics import paired_bootstrap_ci  # noqa: E402


def fn(r):
    R = np.array(r["R"]); return R[1, 1], R[1, 0] - R[0, 0]  # (new-domain loss, forgetting)


base = defaultdict(list)
for f in glob.glob(str(ROOT / "results/pilot_rev/rev_none_lr*_A_s*.json")):
    r = json.load(open(f)); base[r["seed"]].append(fn(r))
groups = defaultdict(dict)
for f in glob.glob(str(ROOT / "results/pilot_rev/rev_cms_lr0.0006_A_s*.json")):
    r = json.load(open(f)); groups["cms (pilot_rev, ref)"][r["seed"]] = fn(r)
for f in glob.glob(str(ROOT / "results/stage1c/*.json")):
    r = json.load(open(f)); groups[re.sub(r"^s1c_", "", r["name"])][r["seed"]] = fn(r)
for name in sorted(groups):
    seeds = sorted(groups[name])
    if name.startswith("replay"):
        pass
    ex, bf, bad = [], [], 0
    for s in seeds:
        pts = sorted(base[s]); xs, ys = zip(*pts); n, f = groups[name][s]
        if not xs[0] <= n <= xs[-1]:
            bad += 1; continue
        b = np.interp(n, xs, ys); ex.append(f - b); bf.append(b)
    raw = np.mean([groups[name][s][1] for s in seeds]); rawn = np.mean([groups[name][s][0] for s in seeds])
    line = f"{name:24s} seeds={len(seeds)} raw forgetting {raw:.3f} new-loss {rawn:.3f}"
    if bad or len(ex) < 2:
        print(line + f"  -> outside baseline range for {bad} seed(s); no matched readout"); continue
    m, lo, hi = paired_bootstrap_ci(np.array(ex), np.zeros(len(ex)))
    rel = m / np.mean(bf)
    print(line + f"  excess {m:+.3f} [{lo:+.3f},{hi:+.3f}] ({rel:+.1%} of baseline) "
          + ("PASS" if hi < 0 and rel <= -0.10 else "fail"))
