"""Stage 0 readout. Applies the pre-registered kill criteria; does not tune anything.

Criteria (README):
 (a) CMS must reduce domain-1 loss increase vs baseline at matched domain-2 loss.
 (b) CMS must differ from the LR-matched control beyond seed noise.
Matched plasticity: new-domain loss within 2% of the comparison arm; otherwise flagged "NOT MATCHED".
Minimum effect of interest: 10% relative reduction in forgetting, CI excluding 0.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from catadye.metrics import forgetting, new_domain_loss, paired_bootstrap_ci  # noqa: E402

runs = defaultdict(dict)  # mode -> seed -> result
for f in sorted((ROOT / "results" / "pilot").glob("*.json")):
    r = json.loads(f.read_text())
    runs[r["mode"]][r["seed"]] = r

seeds = sorted(set.intersection(*[set(v) for v in runs.values()])) if runs else []
print(f"modes={list(runs)} seeds={seeds}")
if len(seeds) < 2 or not {"none", "cms", "lr_matched"} <= set(runs):
    sys.exit("need none, cms, lr_matched with >=2 common seeds")


def vec(mode, fn):
    return np.array([fn(np.array(runs[mode][s]["R"])) for s in seeds])


F = {m: vec(m, forgetting) for m in runs}
N = {m: vec(m, new_domain_loss) for m in runs}
print(f"\n{'arm':12s} {'forgetting':>18s} {'new-dom loss':>18s} {'tok/s':>8s} {'VRAM MB':>8s} {'upd frac':>8s}")
for m in ("none", "cms", "lr_matched"):
    t = np.mean([runs[m][s]["tokens_per_s"] for s in seeds])
    v = np.mean([runs[m][s]["peak_vram_mb"] for s in seeds])
    u = runs[m][seeds[0]]["frac_params_updated_per_step"]
    print(f"{m:12s} {F[m].mean():8.3f} ± {F[m].std(ddof=1):5.3f}  {N[m].mean():8.3f} ± {N[m].std(ddof=1):5.3f} "
          f"{t:8.0f} {v:8.0f} {u:8.3f}")


def compare(a, b):
    m, lo, hi = paired_bootstrap_ci(F[a], F[b])
    rel = m / F[b].mean()
    plast = (N[a].mean() - N[b].mean()) / N[b].mean()
    matched = abs(plast) <= 0.02
    print(f"\n{a} vs {b}: forgetting diff {m:+.3f} [{lo:+.3f}, {hi:+.3f}] ({rel:+.1%} of {b}); "
          f"new-domain loss diff {plast:+.1%} -> {'matched' if matched else 'NOT MATCHED'}")
    return m, lo, hi, rel, matched


m1, lo1, hi1, rel1, mt1 = compare("cms", "none")
m2, lo2, hi2, rel2, mt2 = compare("cms", "lr_matched")
m3, lo3, hi3, rel3, mt3 = compare("lr_matched", "none")

print("\n--- pre-registered readout ---")
a_pass = hi1 < 0 and rel1 <= -0.10
b_pass = hi2 < 0 and rel2 <= -0.10
print(f"(a) CMS reduces forgetting vs baseline by >=10% with CI<0: {a_pass}  (plasticity matched: {mt1})")
print(f"(b) CMS beats LR-matched control by >=10% with CI<0:       {b_pass}  (plasticity matched: {mt2})")
if not (mt1 and mt2):
    print("NOTE: plasticity not matched in at least one comparison; verdict is provisional until an lr sweep "
          "is run (log as a deviation in README).")
print(f"n seeds = {len(seeds)}; with 3 seeds the bootstrap has 27 distinct resamples, so CIs are coarse.")
