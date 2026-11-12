"""Revised-pilot readout, exactly as declared in README ("Pilot revision 1").

Per arm and seed: forgetting is linearly interpolated as a function of new-domain loss across the 3
learning rates, evaluated at the median baseline new-domain loss, only if that target lies inside
every arm's range for every seed; otherwise "no overlap". Seed-paired bootstrap CIs; criteria
(a) CMS vs baseline and (b) CMS vs LR-matched need >=10% reduction with CI below 0.
Order B is a robustness check and is not used in the gate.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from catadye.metrics import forgetting, new_domain_loss, paired_bootstrap_ci  # noqa: E402

ARMS = ["none", "cms", "lr_matched"]
runs = [json.loads(f.read_text()) for f in sorted((ROOT / "results" / "pilot_rev").glob("*.json"))]
A = [r for r in runs if r["cfg"].get("order") == "A"]
B = [r for r in runs if r["cfg"].get("order") == "B"]
print(f"order A runs: {len(A)}/27, order B runs: {len(B)}/9")

# pt[arm][seed] -> sorted list of (new_loss, forgetting, lr)
pt = {a: {} for a in ARMS}
for r in A:
    R = np.array(r["R"])
    pt[r["mode"]].setdefault(r["seed"], []).append((new_domain_loss(R), forgetting(R), r["cfg"]["lr"]))

seeds = sorted(set.intersection(*[{s for s, v in pt[a].items() if len(v) == 3} for a in ARMS]))
print("seeds with all 3 lrs for all arms:", seeds)
if len(seeds) < 2:
    sys.exit("not enough complete seeds yet")

print("\nraw per-lr means (new-domain loss, forgetting):")
lrs = sorted({p[2] for a in ARMS for s in seeds for p in pt[a][s]})
for lr in lrs:
    row = []
    for a in ARMS:
        v = np.array([[p[0], p[1]] for s in seeds for p in pt[a][s] if p[2] == lr])
        row.append(f"{a}: {v[:,0].mean():.3f}, {v[:,1].mean():.3f}")
    print(f"  lr {lr:g}: " + " | ".join(row))

target = float(np.median([p[0] for s in seeds for p in pt["none"][s]]))
ok = all(min(p[0] for p in pt[a][s]) <= target <= max(p[0] for p in pt[a][s]) for a in ARMS for s in seeds)
print(f"\ntarget new-domain loss (median baseline) = {target:.3f}; inside every arm's range for every seed: {ok}")

if ok:
    F = {}
    for a in ARMS:
        F[a] = np.array([np.interp(target, *zip(*sorted((p[0], p[1]) for p in pt[a][s]))) for s in seeds])
    for a in ARMS:
        print(f"  {a:11s} forgetting at matched plasticity: {F[a].mean():.3f} ± {F[a].std(ddof=1):.3f}")
    res = {}
    for x, y in (("cms", "none"), ("cms", "lr_matched"), ("lr_matched", "none")):
        m, lo, hi = paired_bootstrap_ci(F[x], F[y])
        res[(x, y)] = (m, lo, hi, m / F[y].mean())
        print(f"  {x} vs {y}: {m:+.3f} [{lo:+.3f}, {hi:+.3f}] ({m / F[y].mean():+.1%})")
    a_pass = res[("cms", "none")][1:3][1] < 0 and res[("cms", "none")][3] <= -0.10
    b_pass = res[("cms", "lr_matched")][1:3][1] < 0 and res[("cms", "lr_matched")][3] <= -0.10
    print(f"\n(a) CMS reduces forgetting vs baseline by >=10% with CI<0: {a_pass}")
    print(f"(b) CMS beats LR-matched control by >=10% with CI<0:       {b_pass}")
    print("GATE:", "full plan" if b_pass else "short version (pre-registered)")
else:
    print("NO OVERLAP: report raw per-lr differences only; gate decision is 'short version' by default "
          "since the declared test could not be evaluated.")

if B:
    print("\norder B (lr 6e-4) forgetting by arm:")
    for a in ARMS:
        v = np.array([forgetting(np.array(r["R"])) for r in B if r["mode"] == a])
        if len(v):
            print(f"  {a:11s} {v.mean():.3f} ± {(v.std(ddof=1) if len(v) > 1 else 0):.3f} (n={len(v)})")
