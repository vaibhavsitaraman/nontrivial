Python lives in `src/` (`nets`, `envs`, `checks`, `runs`). Checkpoints, plots, New Jersey data, and this note are in `checkpoints/`, `plots/`, `data/`, and `docs/`. See `README.md` for the short map. Names below are the older filenames.

# Project Handoff: Multivariate Bellman Flows — Flow Matching for Hospital Surge-Capacity Allocation

This document is a complete context transfer for continuing this research project in a new
conversation. It covers the full history, current state, all validated results, all known
open issues, and a manifest of every code/document file, with enough detail that a fresh
Claude instance (or a human) can pick up the project without re-deriving anything already
established here.

---

## 1. The project, in one paragraph

We are testing whether **multivariate conditional flow matching** — a modern generative
modeling technique — can serve as the representation for **distributional reinforcement
learning** in a setting with genuinely joint, correlated risk: hospital surge-capacity
planning across multiple regions and patient demographics (general inpatient, adult ICU,
pediatric ICU). The central hypothesis is that modeling the **joint** return distribution
(not just each risk dimension's marginal) matters specifically because hospital networks
rely on inter-facility mutual aid as their release valve for local overload, and mutual aid
fails precisely when multiple facilities are overloaded **simultaneously** — a joint tail-risk
event that cannot, in general, be predicted from marginal (per-facility, per-demographic) risk
estimates alone. The project builds a synthetic-but-mechanistically-grounded environment,
trains a CVaR-based mutual-aid planning policy via flow-matching distributional TD learning,
and compares it against four baselines specifically designed to isolate different candidate
explanations for any observed advantage. The next planned phase (not yet started) is
calibrating the same methodology against real HHS hospital utilization data.

---

## 2. Full narrative of what has been built and validated, in order

Each numbered step below was gated: a new mechanism was only combined with existing code
after being validated in isolation against an independent ground truth (closed-form math,
Monte Carlo simulation, or exact tabular value iteration). This discipline caught several real
bugs, listed inline where they occurred — **do not skip this validation discipline when
extending the project**, it has repeatedly caught things that would otherwise have silently
corrupted results.

### Step 1 — Pure flow matching sanity check (no RL)
Verified the core flow-matching machinery (velocity network, conditional flow-matching loss,
Euler ODE sampler) against a known 1D Gaussian-mixture target, with no bootstrapping. Passed
(Wasserstein-1 = 0.13 at 50 ODE steps). Also characterized the ODE-steps-vs-sample-fidelity
tradeoff (diminishing returns past ~20 steps) — this curve became relevant much later.
**File:** `step1_flow_matching_sanity_check.py`

### Step 2 — Univariate distributional TD (no control)
A 3-state chain MRP with a **closed-form** mean/variance solution (derived via linear algebra
on the Bellman equations, verified against long Monte Carlo rollouts). Trained a
state-conditioned flow model via TD bootstrapping (target network + noise coupling between
current/successor paths). **Bug found and fixed:** using too few ODE integration steps
(`n_ode_steps_train=8`) when generating the **training-time bootstrap target** (not just at
sampling/eval time) caused a systematic, compounding ~15% **underestimate of variance**.
Raising to 20 steps mostly resolved it. **This finding recurred later at the hospital-control
scale and is a key methodological lesson of the whole project — see Step 5a/6 below.**
**Files:** `mrp_env.py`, `flow_model.py`, `distributional_td.py`, `evaluate_step2.py`

### Step 3 — Multivariate extension, no correlation (d=2)
Generalized to 2-dimensional rewards with the closed-form solver extended to mean **vectors**
and **covariance matrices**. **Bug found and fixed:** the first attempt at a "no correlation"
environment actually had strong (~-0.75) *induced* correlation, purely from both reward
dimensions sharing the same underlying state trajectory (even though per-step shocks were
independent). Had to search for reward-mean parameters that genuinely give near-zero induced
correlation (~0.01-0.015). **Files:** `mrp_env_multivariate.py`, `evaluate_step3.py`

### Step 4 — Correlation-strength sweep: does joint modeling actually help?
Injected deliberate within-step correlation (rho swept 0 to 0.99) and compared the **joint**
flow model against an **independent per-dimension** baseline (two separate univariate models,
combined only by stacking independently-drawn samples). **Core positive result:** joint model
tracked true correlation closely at every rho; independent baseline stayed pinned near zero by
construction; downstream CVaR error for independent baseline grew to ~2.9x the joint model's
at high correlation. **Two real findings along the way:** (a) injected within-step correlation
doesn't translate 1:1 into return correlation — diluted by differing per-state reward-std
ratios, capping around 0.33-0.41 even at rho=1 (found empirically before picking sweep
values); (b) background processes don't survive across tool calls in the sandbox, requiring
per-rho-checkpointed execution. **Files:** `mrp_env_correlated.py`, `metrics.py`,
`step4_run_rho.py`, `step4_finalize.py`, `evaluate_step4.py`

### Step 5a — Control mechanism, validated on a toy MDP first
Extended to **action-conditioned** flow matching with a genuine Bellman **optimality**
operator (CVaR-greedy action selection, `ConditionalVelocityNetSA`). Validated on a small
hand-designed 2-state/2-action MDP where the **mean-optimal** policy (Aggressive/Aggressive)
and the **CVaR-optimal** policy (Conservative/Conservative) deliberately disagree — found by
parameter search, confirmed via brute-force enumeration of all 4 stationary policies.
**Methodological note caught here:** CVaR is not time-consistent, so "best stationary policy
by total CVaR" is not necessarily the fixed point of a per-step greedy-CVaR Bellman operator —
built a separate exact **tabular** (particle-based, no neural network) value-iteration
reference computing the correct fixed point, used as ground truth ever since. Result: learned
greedy policy matched the tabular optimum in both states. **A container reset wiped one day's
unsaved work** mid-step; rebuilt from the conversation record — lesson: back up/present files
early and often, don't wait for a step to be "fully done." **Files:**
`mrp_env_toy_control.py`, `tabular_greedy_cvar_reference.py`, `distributional_td_control.py`
(now superseded for multivariate use — see Step 6's bug), `flow_model.py` (added
`ConditionalVelocityNetSA`/`ode_sample_sa`)

### Step 5b — Building the hospital environment (spatiotemporal + scenario-dependent)
This is where the project's "compelling framework" argument was developed, in response to a
request to ground the multivariate-modeling claim in genuine spatiotemporal structure rather
than an abstract correlation knob.

- **`hospital_network_regimes.py`**: a two-region (Hub/Spoke) regime chain (4 regimes each:
  Baseline/Rising/Peak/Waning). The Spoke's transition probabilities depend on the Hub's
  **current** regime — a genuine one-step-lagged leading-indicator mechanism. Validated via
  exact linear algebra (16-state joint chain solved by power iteration) cross-checked against
  500K-step simulation; confirmed the `spillover_strength` parameter produces the intended
  monotonic leading-indicator correlation (0.00 at spillover=0 -> 0.79 at spillover=1).
- **`hospital_reward_copula.py`**: a **scenario-dependent one-factor shock model** for the
  6-dimensional (2 regions x 3 bed types) shortfall vector. The *loading vector* — not a fixed
  covariance matrix — encodes whether a scenario is correlated at all: `isolated_pediatric`
  (near-zero loading, independent per-region shocks — the negative control), `regional_flu`
  (moderate, adult-skewed), `pandemic` (uniform loading across all 6 dims). Condition A/B
  (Gaussian vs Student-t shock) tests tail dependence specifically, matched in mean/variance.
  **Two real bugs found and fixed:** (1) at moderate thresholds (80th percentile), Student-t
  showed *lower* joint-overload probability than Gaussian — the *opposite* of the intended
  effect — because Student-t is more peaked (leptokurtic), redistributing mass away from
  moderate deviations specifically; the intended direction only appears past ~95-99th
  percentile (confirmed via explicit percentile sweep). (2) Using one fixed absolute threshold
  across scenarios made the low-variance `isolated_pediatric` scenario show exactly zero
  joint-overload events regardless of true correlation — fixed via **per-scenario calibrated
  thresholds** (`calibrate_overload_thresholds`).
- **`hospital_env_combined.py`**: wires the regime chain to the copula reward layer.
  Documents an important conceptual distinction: **mean co-movement** (regimes moving together
  -> means moving together, fully recoverable by any state-conditioned model, including
  independent ones) vs **residual shared-shock correlation** (the copula's common factor,
  invisible to independent models regardless of conditioning — this is the actual thing being
  tested). Validated via closed-form mean (16 states x 6 dims) vs Monte Carlo, and confirmed
  spillover strength measurably increases full-trajectory joint-overload rate (2.7x from
  spillover=0 to 0.8), showing the spatiotemporal mechanism genuinely contributes to the
  downstream risk metric, not just the copula alone.
- **`hospital_env_control.py`**: adds the **control** layer — a 6-action space along two
  axes: **direction** {none, H->S, S->H} and **intensity** {conservative, aggressive}. A
  transfer's benefit to the destination is scaled by the *source's* own spare capacity
  (regime-dependent) — a genuine joint/spatial dependency. Aggressive intensity helps the
  destination more but adds risk back to the source. **Bug found and fixed:** the
  destination floor-clip was applying a spurious mean-shift even when the transfer's effect
  was mathematically zero (source at Peak = no spare capacity) — fixed by skipping the clip
  when `effect~0`.
- **`hospital_tabular_reference_control.py`**: generalizes the tabular greedy-CVaR reference
  to vector-valued particles with a fixed clinical-priority scalarization weight. Result:
  8/16 states stable (consistent, sensible mutual-aid direction favoring whichever region is
  more severely affected), 8/16 oscillating (exactly the near-symmetric states where the
  decision is a genuine near-tie — not a bug).

### Step 5c — Combining control with the hospital environment (the joint model)
`distributional_td_control_multivariate.py` — reuses the toy-MDP control algorithm, applied to
the full 16-state/6-action/6-dim environment. **This is the main "joint model" referenced
throughout the rest of the project.** Given the sandbox's execution-time limits, training was
handed off to **local execution** via `run_hospital_control_training.py`.

**Local run results (user-executed):**
- Loss converged cleanly (~5.2 -> ~0.65-0.8) over 15,000 steps.
- Match on **unambiguous (stable) tabular-reference states: 9/9 (100%)**.
- **A systematic CVaR "optimism" bias was found and diagnosed via a controlled A/B test:**
  the joint model's CVaR was consistently *less negative* (looking safer) than the true
  tabular-optimal CVaR by ~1.26 units average at `n_ode_steps_train=12`. Rerunning with
  `n_ode_steps_train=25` (everything else held fixed) roughly **halved** the gap to ~0.65 —
  directly confirming the Step 2 finding (insufficient training-time ODE integration steps
  compound into underestimated variance) at this much larger scale. **`n_ode_steps_train=25`
  is now the standard setting for all hospital-control runs.**

### Step 6 — Baselines
Built to isolate specific candidate explanations for any joint-model advantage, each
validated independently before hospital-scale integration, each handed off to local execution
via a dedicated run script:

1. **Independent per-dimension** (`distributional_td_control_independent.py` +
   `run_hospital_independent_baseline_training.py`): 6 separate univariate flow models,
   structurally blind to cross-dimension correlation (own gradient, own uncoupled noise per
   dimension), combined only at decision time via stacking + scalarization. **Bug caught by
   proactive code audit before running anything:** the original 5a-era CVaR computation
   assumed 1D samples; fed 6D hospital rewards directly, `np.quantile` with no axis argument
   would have silently *flattened across all dimensions* rather than scalarizing first — no
   crash, just a meaningless CVaR. Fixed via `metrics.weighted_cvar` reuse in the new file.
2. **floq** (`floq_baseline.py`): same architecture/ODE mechanics as the joint model, but the
   bootstrap successor sample is generated from a **fixed** starting point rather than random
   noise — collapsing the "distribution" to a point mass. Decision rule reduces to
   mean-chasing (isolates: does flow matching's benefit come from genuine distributional
   modeling, or just extra iterative-ODE compute?). **Best-effort reconstruction — no
   official floq code/paper text was available, only search-confirmed high-level description.**
3. **Sliced Cramer distance** (`sliced_cramer_baseline.py` + `sliced_cramer_control.py`): a
   genuinely different architecture — a particle-output network (no ODE at all, single
   forward pass), trained via projecting particles onto random directions and matching sorted
   order statistics (the standard 1D Cramer distance estimator). Validated in isolation first
   against a known correlated 2D Gaussian mixture (recovered true correlation almost exactly:
   -0.7385 true vs -0.7357 learned) — **caught and fixed a real bug while writing this
   validation** (target particles were originally broadcast from one sample instead of drawn
   independently, which would have collapsed the target to a degenerate point set). No
   official code was found for this method either — implemented from the paper's described
   mechanism (confirmed via search: "Multivariate Distributional RL Using Sliced Divergences,"
   ICML 2026).

**Local run results (user-executed, all four methods + tabular, `run_hospital_floq_and_sliced_cramer_training.py`):**

| Method | Overall match (of 16) | Match on 9 unambiguous states | Mean abs. CVaR gap from tabular | Mean relative gap |
|---|---|---|---|---|
| Joint (flow matching) | 12-13 | **9/9** | **0.65** | **3.2%** |
| Independent | 13 | 9/9 | 4.45 | 22.0% |
| Sliced Cramer | 12 | 9/9 | 7.70 | 38.1% |
| floq | 13 | 9/9 | 8.98 | 44.5% |

**Key interpretation:** all methods agree on the correct *decision* in every unambiguous
state — the advantage of joint modeling shows up specifically in the **quantitative risk
estimate** (how much reserve capacity a real planner would hold), not the coarser
which-action-to-pick question. The independent-baseline comparison is the cleanest evidence
for the core hypothesis (same architecture/training, only cross-dimension coupling differs).
floq's huge gap is *expected* given it has no tail-risk representation at all — not
necessarily evidence it's a poor mean estimator. **The sliced-Cramer underperformance
(worse than the correlation-blind independent baseline, despite being independently validated
to represent correlation correctly) is flagged as an open, unresolved finding** — see Section 3.

### Step 7 — Written report (LaTeX)
A full ~5,660-word report was written (`report.tex`/`report.pdf`), covering: problem
motivation (mutual aid mechanism, why mean-based planning fails), related work (Bellemare
et al. distributional RL, Wiltzer et al. multivariate-DRL projection failure, sliced
divergences, flow matching, floq), environment/methodology description, the four-baseline
ablation design, the results above, and a **substantial limitations section** documenting
every bug in this document plus the open sliced-Cramer question. The "scaling to real HHS
data" section is **intentionally left as a placeholder** pending the next project phase.
**AI usage disclosure note:** the fellowship organizers were shown the actual extent of AI
involvement in this project and approved disclosure language stating AI was used "to assist
in code generation and as an intellectual sparring partner" — this exact language is already
in the report; do not need to revisit this unless asked.

### Step 8 — In-progress: maximization-bias diagnosis and fix (NOT YET RE-RUN)
The user observed that floq's and sliced-Cramer's training loss curves both show an initial
trough followed by a rise that plateaus **higher** than the trough minimum (unlike the joint/
independent models' smooth monotonic convergence). Asked to check for bugs before assuming
"overfitting."

**Code review found no logic bugs**, but identified a real structural asymmetry: both
baselines' periodic greedy-action tables were computed from **far fewer effective samples**
than the joint/independent models' (floq: n=1 point estimate per cell vs joint's 800 samples;
sliced-Cramer: N=64 particles, ~6-7 in the CVaR tail, vs joint's ~80). This is the textbook
setup for **maximization bias** (Thrun & Schwartz 1993; van Hasselt 2010's Double-Q
motivation): `argmax` over noisy/imperfectly-approximated value estimates systematically
selects whichever estimate got noise-inflated, and this compounds through repeated Bellman
backups — distinct from classical overfitting, and mechanistically predicts exactly the
observed pattern (worse for the few-sample baselines, absent for the many-sample ones).

**Fixes implemented and smoke-tested (200 steps, runs cleanly, loss decreasing) but NOT yet
run at full scale:**
- `floq_baseline.py`: `compute_greedy_actions_floq` and the training loop's bootstrap-target
  computation now **average over multiple random x0 draws** (`n_point_samples_greedy=20`,
  `n_point_samples_target=8` by default) instead of a single fixed `x0=0`.
- `sliced_cramer_control.py`: default `n_particles` increased 64->192 (so the CVaR tail at
  alpha=0.1 uses ~19 particles instead of ~6-7), `n_directions` increased 8->16 (less noisy
  per-step training gradient). Measured smoke-test cost: ~6.4x slower per step than before
  (45s/200 steps vs ~7s/200 steps previously) — expect roughly 50-60 minutes for 15,000 steps
  locally, up from the original ~204s. **This cost increase should be flagged to the user
  before they commit to a full rerun.**

**This is the immediate next action item — see Section 4.**

---

## 3. Open questions / things that don't yet have a clean answer

1. **Does the maximization-bias fix actually explain and fix the loss curve pattern?** Not
   yet confirmed — the fix is implemented and smoke-tested for correctness (no crashes,
   loss still decreases) but has NOT been run at full scale to check whether it changes the
   *shape* of the loss curve (removes the rise-and-plateau) or improves the CVaR-gap numbers
   in Section 2's table. This needs a full local rerun.
2. **Why did sliced-Cramer underperform the independent baseline**, despite being
   independently validated to represent correlation correctly on a toy 2D problem? Three
   candidate explanations were identified and none has been definitively ruled in or out:
   (a) under-tuning of a from-scratch, no-official-code reimplementation (plausible — the
   loss curve was visibly noisier and higher than floq's); (b) a genuine limitation of
   one-sample-bootstrapped sliced-Cramer training in this specific correlated, high-dimensional
   setting; (c) a real, currently unexplained weakness of the method relative to flow matching
   in this kind of control setting. The Step 8 particle-count fix may partially address (a)
   but was motivated by the maximization-bias diagnosis, not this question specifically —
   worth re-examining once the reruns are in.
3. **Has the maximization-bias issue also been (silently) affecting the joint and
   independent models**, just to a smaller degree given their larger sample counts? Their
   loss curves looked clean, but this hasn't been explicitly ruled out via the same kind of
   controlled ablation used to diagnose the ODE-step-count bias. Worth a deliberate check if
   time allows, rather than assuming they're unaffected just because their curves look smooth.
4. **All results so far are on a synthetic, self-designed testbed.** The correlation
   structure, regime dynamics, and reward parameters were chosen to make the hypothesized
   effect visible and testable, not calibrated to real hospital data. Real HHS data calibration
   (Section 4) is the direct next step to address this, but has not yet started.

---

## 4. What's left to do (in rough priority order)

1. **Re-run floq and sliced-Cramer locally with the Step 8 fixes** (files already updated;
   the existing `run_hospital_floq_and_sliced_cramer_training.py` script works unchanged since
   the new parameters have defaults — just re-run it). Check: (a) does the loss curve shape
   change (no more rise-and-plateau)? (b) do the CVaR-gap numbers improve, especially for
   sliced-Cramer relative to the independent baseline? **Flag to the user: sliced-Cramer will
   now take roughly 50-60 min instead of ~3.5 min, given the particle-count increase.**
2. **Investigate the sliced-Cramer open question further** depending on what the rerun shows
   — possibly a dedicated particle-count/direction-count sweep if compute allows.
3. **Real HHS data calibration** — the explicit next phase flagged in the report. The HHS
   COVID-19 hospital utilization dataset (Socrata resource ID `sgxm-t72h`, fields confirmed to
   include `inpatient_beds`, `inpatient_beds_used`, adult/pediatric ICU fields, daily,
   state-level, Jan 2020-present) is reachable via `web_fetch` using Socrata's SoQL-filtered
   query pattern (`/resource/{id}.csv?$select=...&$where=...`), NOT via the bash sandbox
   directly (healthdata.gov is not in the network allowlist) and NOT via the full bulk
   download (too large, times out). Use this to calibrate: regime severity levels, per-region
   reward means/stds, and ideally the copula's correlation magnitude/tail behavior against
   real documented surge-wave co-occurrence (2020 initial wave, Delta, Omicron).
4. **Scale to the fuller action space** — a 3-axis design (transfer direction, response
   intensity, elective-deferral posture, external staffing request) was discussed early on and
   deliberately deferred in favor of validating the smaller 2-axis/6-action space first. Now
   that the smaller version is confirmed working, extending it is a natural next step, but will
   require re-validating the tabular reference and control mechanism at the larger scale
   (expect real compute-cost implications, following the same pattern as before).
5. **Remaining baselines not yet built**: MD3QN (MMD-based multivariate DRL) and
   SinkhornDistRL — working code repositories were found for both during literature search
   but implementation was deliberately deprioritized given the four baselines already isolate
   the two most important confounds (correlation-awareness, genuine distributional modeling).
   Revisit only if there's a specific reason the four aren't sufficient.
6. **Extend to 3+ regions** — the current environment uses 2 regions (Hub/Spoke) as the
   minimal case with genuine spatiotemporal spillover; a richer regional network was discussed
   as a natural but deliberately-deferred extension.
7. **Update the LaTeX report** with the Section 3/4 findings above once the reruns are in,
   and fill in the currently-placeholder "Scaling to Real Hospital Utilization Data" section
   once HHS calibration is complete.
8. **Verify the two paper citations flagged as uncertain** in the report's references (the
   sliced-Cramer/SDRL paper and the floq paper) — exact author lists/venues were not fully
   confirmed during search; the report currently uses "Anonymous"/best-effort attribution.

---

## 5. Practical / infrastructure notes for whoever continues this

- **The sandbox environment resets its filesystem and installed packages unpredictably
  between tool calls** (confirmed once, mid-project). `torch`/`numpy`/`matplotlib`/`scipy`
  need `pip install ... --break-system-packages` if missing. **Back up validated files to
  `/mnt/user-data/outputs/` (via `present_files`) frequently, not just at the end of a step** —
  this is how the one filesystem reset was recovered from without losing substantive work.
- **Tool-call execution has a wall-clock time limit** shorter than full training runs need
  (observed failures around 200-300s for some calls). For anything beyond a quick smoke test,
  either (a) split into checkpointed sub-calls that save intermediate state to disk (see
  `step4_run_rho.py`/`step4_finalize.py` for the pattern), or (b) hand off to **local
  execution** via a self-contained run script (see `run_hospital_control_training.py` and its
  successors for the pattern: train, evaluate, save plots + a text summary, print explicit
  instructions on what to send back). Background/detached processes do **not** survive across
  tool calls in this sandbox — confirmed via direct test.
- **Always smoke-test a full run script end-to-end at drastically reduced scale** (few hundred
  steps, few tabular iterations) before handing it to the user for a full run — this has
  caught real bugs in previously-untested evaluation/plotting code paths more than once.
- **The dependency closure for each run script was computed programmatically** (via AST
  parsing of local imports), not just by eye — recommended to keep doing this when packaging
  new run scripts, to avoid missing a file.
- Every environment file's `__main__` block contains a self-check — running any of them
  directly (e.g. `python3 hospital_reward_copula.py`) reproduces the validation described
  above and is a fast way to confirm the current codebase state matches this document.

---

## 6. Complete file manifest

All files below are attached to this handoff in their current (latest) state.

**Core flow-matching machinery:**
- `flow_model.py` — velocity networks (`ConditionalVelocityNet` for state-only,
  `ConditionalVelocityNetSA` for state-action) and Euler ODE samplers. Dimension-generic
  throughout.
- `metrics.py` — shared evaluation utilities: `energy_distance`, `cvar`, `weighted_cvar`.

**Steps 1-4 (toy environments, evaluation-only and correlation-sweep):**
- `step1_flow_matching_sanity_check.py`
- `mrp_env.py`, `distributional_td.py`, `evaluate_step2.py`
- `mrp_env_multivariate.py`, `evaluate_step3.py`
- `mrp_env_correlated.py`, `step4_run_rho.py`, `step4_finalize.py`, `evaluate_step4.py`

**Step 5a (toy control validation):**
- `mrp_env_toy_control.py`, `tabular_greedy_cvar_reference.py`,
  `distributional_td_control.py` (superseded for multivariate use by
  `distributional_td_control_multivariate.py`, but still correct for the toy MDP),
  `step5a_compute_reference.py`, `step5a_train.py`, `step5a_evaluate.py`

**Step 5b (hospital environment construction):**
- `hospital_network_regimes.py`, `hospital_reward_copula.py`, `hospital_env_combined.py`,
  `hospital_env_control.py`, `hospital_tabular_reference_control.py`

**Step 5c / joint model (hand off to local execution):**
- `distributional_td_control_multivariate.py`, `run_hospital_control_training.py`

**Step 6 (baselines, hand off to local execution):**
- `distributional_td_control_independent.py`, `run_hospital_independent_baseline_training.py`
- `floq_baseline.py` **(latest version, includes Step 8 fix)**
- `sliced_cramer_baseline.py` (core mechanism + fixed-distribution validation),
  `sliced_cramer_control.py` **(latest version, includes Step 8 fix)**
- `run_hospital_floq_and_sliced_cramer_training.py` (works unchanged with the Step 8 fixes,
  since new parameters default sensibly — **this is the script to re-run next**)

**Step 7 (report):**
- `report.tex`, `report.pdf`, `loss_curves.png`, `multiway_comparison.png` (the two figures
  embedded in the report, from the pre-Step-8-fix floq/sliced-Cramer run)

**This document:**
- `PROJECT_HANDOFF.md` (this file)

---

## 7. Suggested first message to send in the new chat

Something like: *"Continuing a research project — please read the attached
PROJECT_HANDOFF.md in full before doing anything else, then [pick up at item 1 of Section 4 /
whatever you want to do next]."*
