# catadye: does update-frequency staggering reduce catastrophic forgetting?

Controlled study of one component of Google's Nested Learning / Hope architecture
(arXiv 2512.24695): CMS-style MLP blocks that update at different frequencies, tested in a plain
small transformer. Negative results will be reported as readily as positive ones.

**Scope caveat.** In the paper, CMS frequencies count tokens in the context stream and blocks update
at test time on Llama3 backbones after 15B tokens of extra pre-training. The main study here
staggers updates per *optimizer step* during training ("CMS-inspired"). It neither confirms nor
refutes the paper's claims. A toy-scale paper-style probe (Stage 1b) is exploratory and reported
separately.

## Pre-registration (written before any experiment is run)

Status: PRE-REGISTERED. Changes after the first run go in the Deviations log at the bottom, with
a date and reason; the original text is never edited.

### Hypotheses
- **H1** CMS (periods 1/8/64) reduces forgetting versus the baseline at matched plasticity.
- **H2** CMS reduces forgetting versus the LR-matched control (blocks at lr, lr/8, lr/64 updated
  every step). This separates "update frequency" from "smaller effective step".
- **H3** Slow-momentum AdamW reduces forgetting versus plain AdamW.
- **H4** CMS and slow momentum interact (2x2 interaction term differs from zero).

### Primary metric
Forgetting at matched plasticity. For each arm, sweep the shared learning rate scale on domain 1
only; compare arms at equal new-domain (final-domain) eval loss, via the stability-plasticity
frontier (mean loss increase on earlier domains versus new-domain loss). Secondary: loss matrix
R[i,j] (domain j after training on i), backward transfer, forward transfer, final average loss.
Cost: wall-clock, peak VRAM, tokens/s.

### Minimum effect of interest
A 10% relative reduction in mean earlier-domain loss increase, with the 95% CI excluding 0 and
new-domain loss no more than 2% worse. Smaller differences are reported as "no practical effect".

### Statistics
Seed-paired differences, bootstrap 95% CIs (10,000 resamples), Holm correction across the planned
comparisons H1-H4. A difference whose CI includes 0 is called "within noise".

### Matching
Every arm has identical parameter count and forward FLOPs (each MLP is a 3-block chain in every
arm). Update overhead is reported honestly. Learning rates come from a pre-declared sweep on domain
1 only; nothing is tuned on forgetting metrics.

### Stage 0 pilot and kill criteria
~10M-param GPT, TinyStories then Python code, 3 arms (baseline, CMS 1/8/64, LR-matched), 3 seeds.
The "frequency per se" hypothesis is falsified at pilot scale if (a) CMS does not reduce
domain-1 loss increase versus baseline at matched domain-2 loss, or (b) CMS is within seed noise of
the LR-matched control. The study continues either way, framed around whichever outcome holds.

**Decision gate:** if CMS differs from the LR-matched control beyond seed noise, run the full plan
(Stage 1 grid, Model B check, Stage 1b probe). Otherwise run the short version (arms 1-5 plus
replay, Stage 1b, a negative-result write-up) and skip Stage 2.

### Arms (Stage 1)
1 baseline; 2 CMS 1/8/64; 3 slow-momentum AdamW; 4 CMS + slow momentum; 5 LR-matched control;
6 replay 10%; 7 EWC; 8 joint training (upper bound); 9 in-context-learning baseline (eval only).

## Layout
`src/catadye` library, `configs/` per-arm YAML, `scripts/` runners and analysis, `tests/`,
`results/` raw per-run JSONL (not edited by hand), `report/` write-up.

## Deviations log
(empty)
