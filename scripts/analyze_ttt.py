"""Stage 1b readout, as declared in docs/PREREGISTRATION.md ("Stage 1b: test-time probe").

Per arm and seed: forgetting (held-out TinyStories loss at the end minus at step 0) is linearly interpolated
as a function of final held-out Python loss across the 3 test-time learning rates, evaluated at the median
ttt_single final Python loss, only if inside every arm's range for every seed (else "no overlap").
H5: ttt_cms vs ttt_single; H6: ttt_cms vs ttt_lrm. Pass = >=10% relative reduction with seed-paired bootstrap CI < 0.
Sanity check (read strictly, decided before seeing results): every ttt arm must beat `frozen` on final
Python loss at EVERY learning rate; failures are listed.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from catadye.metrics import paired_bootstrap_ci  # noqa: E402

ARMS = ["ttt_single", "ttt_cms", "ttt_lrm"]
runs = [json.loads(f.read_text()) for f in sorted((ROOT / "results" / "ttt").glob("*.json"))]
print(f"runs: {len(runs)}/30")


def end(r):
    c = r["curve"]
    return c[-1]["new"], c[-1]["old"] - c[0]["old"]


frozen = {r["seed"]: end(r) for r in runs if r["arm"] == "frozen"}
pt = {a: {} for a in ARMS}
for r in runs:
    if r["arm"] in ARMS:
        n, f = end(r)
        pt[r["arm"]].setdefault(r["seed"], []).append((n, f, r["lr"]))
seeds = sorted(set.intersection(*[{s for s, v in pt[a].items() if len(v) == 3} for a in ARMS]) & set(frozen))
print("complete seeds:", seeds)
if len(seeds) < 2:
    sys.exit("not enough complete seeds yet")

print("\nfrozen: final Python loss / forgetting (mean):",
      f"{np.mean([frozen[s][0] for s in seeds]):.3f} / {np.mean([frozen[s][1] for s in seeds]):+.3f}")
print("\nper-lr means (final Python loss, forgetting):")
lrs = sorted({p[2] for a in ARMS for s in seeds for p in pt[a][s]})
fails = []
for lr in lrs:
    row = []
    for a in ARMS:
        v = np.array([[p[0], p[1]] for s in seeds for p in pt[a][s] if p[2] == lr])
        row.append(f"{a}: {v[:, 0].mean():.3f}, {v[:, 1].mean():+.3f}")
        if v[:, 0].mean() >= np.mean([frozen[s][0] for s in seeds]):
            fails.append((a, lr))
    print(f"  lr {lr:g}: " + " | ".join(row))
print("sanity (every arm beats frozen at every lr):", "PASS" if not fails else f"FAIL at {fails}")

target = float(np.median([p[0] for s in seeds for p in pt["ttt_single"][s]]))
ok = all(min(p[0] for p in pt[a][s]) <= target <= max(p[0] for p in pt[a][s]) for a in ARMS for s in seeds)
print(f"\ntarget final Python loss (median ttt_single) = {target:.3f}; inside every arm's range for every seed: {ok}")
if ok:
    F = {a: np.array([np.interp(target, *zip(*sorted((p[0], p[1]) for p in pt[a][s]))) for s in seeds]) for a in ARMS}
    for a in ARMS:
        print(f"  {a:11s} forgetting at matched plasticity: {F[a].mean():+.4f} ± {F[a].std(ddof=1):.4f}")
    out = {}
    for name, (x, y) in {"H5 cms vs single": ("ttt_cms", "ttt_single"), "H6 cms vs lrm": ("ttt_cms", "ttt_lrm"),
                         "lrm vs single": ("ttt_lrm", "ttt_single")}.items():
        m, lo, hi = paired_bootstrap_ci(F[x], F[y])
        rel = m / F[y].mean() if F[y].mean() != 0 else float("nan")
        out[name] = (m, lo, hi, rel)
        print(f"  {name}: {m:+.4f} [{lo:+.4f}, {hi:+.4f}] ({rel:+.1%})")
    for k in ("H5 cms vs single", "H6 cms vs lrm"):
        m, lo, hi, rel = out[k]
        print(f"{k}: PASS" if (hi < 0 and rel <= -0.10) else f"{k}: FAIL (needs >=10% reduction with CI < 0)")
else:
    print("NO OVERLAP: report raw per-lr numbers only; H5/H6 cannot be evaluated as declared.")
