# Results log

Moved verbatim from the README on 2026-11-12. Summary and figures are in the [README](../README.md);
the pre-registration is in [PREREGISTRATION.md](PREREGISTRATION.md). Figures are regenerated from
`results/` by `python scripts/make_figures.py`.

![Pilot 1 results](../figures/pilot1_results.png)

![Revised pilot: stability-plasticity frontier and order check](../figures/pilot_rev_frontier.png)


### Stage 0 pilot (2026-11-12) — raw runs in `results/pilot/`, readout via `scripts/analyze_pilot.py`
16.1M-param GPT, TinyStories -> Python, 2000 steps/domain, 3 seeds/arm. Forgetting = rise in
TinyStories eval loss after training on code (nats); lower is better.

| arm | forgetting | new-domain loss | tokens/s | updated fraction/step |
|---|---|---|---|---|
| baseline | 3.549 ± 0.189 | 2.806 ± 0.009 | 35.7k | 1.000 |
| CMS 1/8/64 | 3.726 ± 0.100 | 2.795 ± 0.009 | 36.9k | 0.919 |
| LR-matched | 3.622 ± 0.043 | 2.786 ± 0.013 | 36.7k | 1.000 |

Plasticity was matched (new-domain loss within 1%) in all comparisons.
- CMS vs baseline: forgetting +0.178 nats [+0.025, +0.334], i.e. CMS forgot ~5% **more**.
- CMS vs LR-matched: +0.104 [+0.026, +0.179], ~3% more.
- Pre-registered criteria (a) and (b) both fail: no reduction at all, let alone the 10% minimum.
  Under the kill criteria the "frequency per se" hypothesis is falsified at this scale.

Caveats: 3 seeds (bootstrap has 27 distinct resamples, so CIs are coarse); one learning rate and
one domain order; the staggered sub-blocks are only ~13% of the parameters (embeddings are ~80%),
so the intervention touches little of the model; forgetting is extreme (>3 nats) in this regime.
Per the decision gate, the short version of the plan applies unless the pilot is revised.

### Pilot revision 1 (2026-11-12) — raw runs in `results/pilot_rev/`, readout via `scripts/analyze_pilot_rev.py`
Per-learning-rate means over 3 seeds (new-domain loss, forgetting in nats), order A:

| lr | baseline | CMS 1/8/64 | LR-matched |
|---|---|---|---|
| 3e-4 | 3.002, 3.172 | 2.978, 3.285 | 2.980, 3.232 |
| 6e-4 | 2.802, 3.568 | 2.743, 3.780 | 2.754, 3.632 |
| 1.2e-3 | 2.659, 4.007 | 2.588, 4.260 | 2.597, 4.069 |

The declared interpolation target (median baseline new-domain loss, 2.790) lay inside every arm's
range for every seed, so the matched-plasticity test was valid. Outcome: criteria (a) and (b) fail;
gate = short version. Per the declaration, no further pilot revisions will be made.
Caveats: 3 seeds; one period set (1/8/64); small scale (24.0M params); from-scratch model; no replay.

### Post-hoc diagnostics (2026-11-12): where does the forgetting live? — EXPLORATORY, not pre-registered
Models: revised-pilot recipe, order A, lr 6e-4; baseline seeds 0-2, CMS and LR-matched seed 0 only
(n=1 each). Checkpoints after each domain; script `scripts/diagnose.py`, raw output
`results/diagnostics.json`, figure `figures/drift_diagnostic.png`.

**Drift** = ||theta_after - theta_before|| / ||theta_before|| over domain-2 training, per parameter group:

| group | baseline (mean of 3) | CMS | LR-matched |
|---|---|---|---|
| embeddings | 0.77 | 0.75 | 0.77 |
| attention | 0.57 | 0.56 | 0.55 |
| layernorm | 0.06 | 0.05 | 0.05 |
| MLP sub-block 1 (period 1) | 0.63 | 0.65 | 0.62 |
| MLP sub-block 2 (period 8) | 0.64 | 0.22 | 0.10 |
| MLP sub-block 3 (period 64) | 0.62 | 0.13 | 0.01 |
| forgetting (nats) | 3.58 | 3.71 | 3.60 |

Reading: the staggering does what it is meant to do (slow blocks move 3-50x less), yet forgetting is
the same as the baseline. The LR-matched sub-block 3 is almost frozen (drift 0.01) and forgetting does
not change. So the forgetting is not carried by the slow blocks; the parts that update every step
(embeddings, attention, MLP sub-block 1) move 55-77% in every arm and plausibly carry it. This is
observational: I did not run the counterfactual (freeze those parts and retrain), and a block can
simply be unimportant for the old domain.

**Revert test** (take the model after domain 2, set one group back to its domain-1 weights, re-evaluate):
not interpretable as a localisation. Reverting a single group mixes weights from different training
states, so most reverts make old-domain loss *worse* (recovery -10% to -60%). Only reverting embeddings
recovers old-domain loss (+32% to +41% of the forgetting), at the price of losing the new domain (+4.5
nats). Reported for completeness, not as evidence.

**Reproducibility limit found along the way:** re-running the same seed and config does not reproduce
the loss matrix. After domain 1 the runs agree to 0.002 nats, but forgetting after domain 2 differs by
-0.09 to +0.11 nats between same-seed runs (5 runs; GPU kernels are not deterministic and domain 2
rewrites most of the model). Effects below ~0.1 nats per run are therefore not resolvable; the
pre-registered 10% bar (~0.36 nats) is well above this, so the gate decisions stand, and the CIs above
already include this noise because every run is an independent draw.

### Stage 1b: test-time probe (2026-11-12) — raw runs in `results/ttt/`, readout via `scripts/analyze_ttt.py`
Declared in `docs/PREREGISTRATION.md` (tag `prereg-stage1b`) before any run. Base: 24.0M-param GPT trained on
TinyStories only (3 seeds). Stream: ~1M tokens of Python, 1000 steps of 4x256 tokens; only MLP sub-blocks update
(AdamW, constant lr, no weight decay). Frozen baseline (no updates): held-out Python loss 8.60, no forgetting.
Forgetting = held-out TinyStories loss at the end minus at step 0 (base: 3.12). 30 runs, 3 seeds.

Per-learning-rate means over 3 seeds (final held-out Python loss, forgetting in nats):

| lr | ttt_single | ttt_cms 1/8/64 | ttt_lrm |
|---|---|---|---|
| 1e-4 | 5.796, +2.660 | 5.795, +1.844 | 5.810, +1.704 |
| 3e-4 | 5.875, +3.557 | 5.786, +3.156 | 5.798, +2.854 |
| 1e-3 | 5.982, +4.601 | 5.921, +4.222 | 5.931, +3.887 |

**Declared readout (matched plasticity by interpolation across the 3 lrs; target 5.867, inside every arm's range):**
- Sanity check (every arm beats frozen at every lr): PASS.
- Forgetting at matched plasticity: single +3.43 ± 0.08, CMS +3.26 ± 0.92, LR-matched +2.70 ± 0.81.
- **H5 (CMS vs single): -0.17 [-0.74, +0.81] (-5.0%): FAIL**, inconclusive (CI spans zero widely).
- **H6 (CMS vs LR-matched): +0.56 [+0.40, +0.70] (+20.8%): FAIL**, CMS forgets *more*, CI excludes zero.

**Why the declared construction is unreliable here (limitation):** final Python loss barely changes with
test-time lr (5.79-5.98) and is not monotone (higher lr gives a worse Python loss for `ttt_single`), so the
forgetting-vs-plasticity curves are nearly vertical and interpolation at a common Python loss amplifies noise
(CMS sd 0.92). Even the lowest lr (1e-4) forgets 2.7 nats for single-rate updates, so the declared lr grid does
not reach a mild-forgetting regime.

**Post-hoc (not pre-registered) per-lr paired comparison, same lr, 3 seeds** (seed-paired bootstrap; with 3 seeds
the bootstrap has 27 distinct resamples, so CIs are coarse). Python-loss difference in the last column:

| lr | CMS vs single | LR-matched vs single | CMS vs LR-matched | Python-loss diff, CMS vs single |
|---|---|---|---|---|
| 1e-4 | -0.82 [-0.86, -0.76] (-31%) | -0.96 (-36%) | +0.14 [+0.13, +0.15] (+8%) | -0.001 |
| 3e-4 | -0.40 [-0.83, -0.16] (-11%) | -0.70 (-20%) | +0.30 [+0.25, +0.35] (+11%) | -0.088 |
| 1e-3 | -0.38 [-0.62, -0.15] (-8%) | -0.71 (-16%) | +0.34 [+0.09, +0.62] (+9%) | -0.060 |

Reading: at the same learning rate, staggered test-time updates forget 8-31% less than single-rate updates
with equal-or-better Python loss, which is the direction the paper claims. But the LR-matched control (same
blocks, every-step updates at lr/1, lr/8, lr/64) forgets less than CMS at every lr, by 8-11%. So the
benefit over single-rate updates is explained by giving deeper blocks a smaller effective step, not by the
update schedule. Caveats: 3 seeds; one period set; base model not co-trained with staggered blocks; no nested
consolidation; the lr grid is too high to reach small-forgetting regimes; per-lr comparisons are post-hoc.
