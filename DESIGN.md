# DESIGN.md — Methodology

This document answers the five design questions SPEC.md requires, and
records the statistical/engineering decisions added at the Week 5 review
point, several of which come directly from the two assigned papers
(FrugalGPT — Chen, Zaharia & Zou 2023; "Adding Error Bars to Evals" —
Miller 2024).

---

## 1. Prompt fairness

**Primary condition: one shared prompt for every model.** `build_prompt()`
in `Runner.py` constructs the exact same instruction + input + options
string regardless of which model receives it — no per-model formatting,
no model-specific few-shot examples, no model-specific system prompt.

This follows directly from SPEC.md's own framing: this project holds the
prompt fixed and varies the model, so prompt sensitivity is a confound to
control, not a thing to study. A per-model tuned prompt would let a
model's score reflect the prompt engineer's skill on that specific model
as much as the model's own capability, and that effect is inseparable from
capability after the fact — there's no way to look at a score and know how
much of it was "the model" versus "how well someone happened to prompt it."

**Secondary condition (not run yet): per-model tuned prompts.** Reserved
as a future ablation, not part of any headline number. If it's ever run,
it must be a clearly separate table (or a `prompt_variant` field the
report can facet on) — never blended into the same accuracy column as the
shared-prompt run. Mixing them would silently let a well-tuned prompt on
one model make it look more capable than a model that only ever saw the
generic instruction.

**What the report claims:** every accuracy number in this report — the
Week 3 table, the Pareto frontier, the quality-floor evaluation, the
paired comparison — is entirely from the shared-prompt condition. No model
in the current results has received any prompt help the others didn't get.

---

## 2. What "good enough" means

Two floor types are implemented (`QualityFloor.kind`): **absolute**
(clear a fixed accuracy) and **relative** (within some margin of the best
model). **Relative, margin 0.03, is the project default** — an absolute
bar doesn't adapt as the model ladder changes, and "within 3 points of the
best available model" is the framing a team actually reasons in when
deciding whether a cheaper model is an acceptable substitute.

The harder half of the question: must a model clear the floor by its point
estimate, or by the lower bound of its confidence interval?

**This project's headline rule is the CI lower bound**
(`QualityFloor.require_ci_lower_bound=True`). Defense: the point-estimate
rule can pass a model whose true accuracy — for all this data can tell you
— might be well below the floor; it's just reporting the middle of a wide
interval as if it were exact. On a 30-item toy set (15 items per split),
Wilson intervals are wide enough that this isn't a hypothetical. A real
run against this project's own toy set surfaced exactly the disagreement
SPEC.md predicts: a model with a perfect 15/15 holdout score passed the
point-estimate rule outright, but failed the CI-lower-bound rule, because a
perfect score on 15 items still has a Wilson interval that reaches down to
roughly 0.80 — not distinguishable from "within 3 points of itself" once
you're honest about the uncertainty. The point-estimate rule would have
confidently recommended a model on evidence that doesn't actually support
that confidence.

Someone spending real money on this recommendation is better served by the
stricter rule: it means fewer models "pass" on a small toy set, and it will
sometimes fail to bless the model with the best point estimate — but it
never recommends a model whose advantage might be entirely noise. As the
real (non-toy) task set grows and intervals narrow, the two rules converge;
until then, the gap between them is a measurement-quality signal in its own
right, not just an inconvenience, so `Report.py` prints both rules side by
side rather than only the headline one.

---

## 3. Escalation signal

**Signal: self-reported confidence**, from the primary model's own
per-token log-probabilities (`Runner.get_confidence`, computed from the
`logprobs` the streaming call already requests). `RoutingDecision`/
`RoutingPolicy` reserve room for three other signals — sample disagreement,
a cheap verifier model, output heuristics — none implemented yet.

**Why this one first, and what it costs:** self-reported confidence is the
only signal on the list with zero marginal cost. It rides along on tokens
the primary model was already generating — no extra API call, so nothing
is subtracted from the theoretical savings a cascade produces.
`RoutingPolicy.escalation_signal_cost_per_request_usd` exists specifically
to account for signals that aren't free — a verifier-model signal would
charge that cost on *every* item, escalated or not, which directly eats
into savings before the cascade even decides anything. Self-reported
confidence's honest cost entry is $0.00.

**Known limitation, not yet resolved:** self-reported confidence derived
from log-probabilities is a proxy for calibrated correctness, not a
calibrated probability itself — models are well-documented to be over- or
under-confident in ways that vary by family and task type. It was chosen
for Week 5 because it's free and doesn't require running a second model;
if cascade evaluation in Week 6 shows it's miscalibrated enough to route
badly (e.g. via a low MPI number — see Section 6 below — for the pairs
it's supposed to be escalating), a verifier-model signal is the documented
fallback, at the cost of no longer being free.

---

## 4. Honest savings

Thresholds get tuned on `SplitName.CALIBRATION`; every number in a report
gets computed on `SplitName.HOLDOUT`. This isn't a convention that relies
on remembering to follow it — it's enforced structurally:

- `simulate_policy()` requires `split` as an explicit, non-optional
  argument. There is no default, so there's no accidental "just run it and
  it'll pick something reasonable" path that quietly lands on calibration.
- `Report.py`'s Pareto frontier, quality-floor evaluation, and paired
  comparison are hardcoded to `SplitName.HOLDOUT` regardless of what
  `--split` the caller passed for the headline table's display filter —
  the display can show calibration numbers for reference, but nothing that
  feeds a recommendation is computed from them.

**The calibration/holdout gap itself:** not yet measured. Cascade
threshold-tuning and routing simulation against the live task set are
Week 6 work per SPEC.md's own timeline (`simulate_policy` and
`choose_call` exist and are unit-tested, but haven't been run against a
real calibration-tuned threshold yet). When that run happens, both numbers
get reported side by side, per SPEC.md's instruction — the gap is a
finding, not something to average away or omit if it's larger than hoped.

---

## 5. Aggregating across task types

**The real deliverable is the per-task-type routing table** (Pareto
frontier per task type, both cost and latency axes —
`compute_all_pareto_frontiers`, printed by `Report.print_pareto_frontiers`),
not a single collapsed headline model. The pooled comparison
(`print_headline_comparison`: best accuracy vs. cheapest model clearing the
floor) is printed too, but as a convenience summary sitting alongside the
disaggregated view, not in place of it.

**What the collapse hides, concretely, on this project's task set:** five
task types (classification, mcq_qa, short_answer, extraction,
summarization) with different scorers and different difficulty profiles.
A model that's Pareto-dominant on classification and MCQ (cheap, easy
exact-match tasks) can lose badly on summarization or short-answer, which
are scored on TOKEN_F1 and reward a stronger model's ability to produce a
well-formed open-ended answer. A single pooled accuracy number blends both
regimes into one figure and can recommend a model that's actually a poor
fit for half the workload — exactly SPEC.md's "wins on classification,
loses badly on extraction" scenario.

**Defense of not collapsing:** the pooled number stays in the report
because it's the fastest thing to read and not every consumer of this
report wants five separate tables. But it's explicitly framed as a
convenience, and the per-task-type frontier is what a routing policy
should actually be built from — a model doesn't need to win everywhere to
be worth having in the ladder, it needs to win on the task types it's
routed to. Collapsing to one number is a deliberate simplification made
visible, not a hidden one.

---

## 6. Statistical additions from the assigned papers

Both were assigned reading; these are the concrete pipeline changes that
came out of them, not just citations.

**Wilson intervals** (`scorer.wilson_interval`) for every `accuracy` field
on `ModelProfile` — verified against `sample-data.json`'s worked example to
4 decimal places for all profiled models and both policy simulations.

**Paired bootstrap** (`scorer.paired_bootstrap`, Miller 2024 Section 4):
model-vs-model comparisons in this project are now paired, not the naive
`sqrt(SE_a² + SE_b²)` unpaired formula — resampling whole `(score_a,
score_b)` pairs per item rather than each model's scores independently,
which is strictly tighter whenever the two models' scores are positively
correlated across the same items (the normal case here, since every model
sees identical inputs). This is what `Report.py` uses for the "does the
cheaper model actually give up accuracy, or is that gap noise" comparison
the headline recommendation hinges on, instead of just comparing two point
estimates.

**Maximum Performance Improvement / MPI** (`scorer.
maximum_performance_improvement`, FrugalGPT): among items where two models
disagree, what fraction does each get right that the other gets wrong. This
is the number that actually justifies routing to a "worse on average" model
for some items instead of just deploying the Pareto-optimal single model —
Section 6's MPI print in `Report.py` is meant to be read as "this is what a
cascade to the cheaper model would be trying to preserve," directly ahead
of Week 6's cascade work.

**Not yet added:** Miller's clustered-standard-error correction (Section
2.2) — doesn't apply to the current toy set, since no items share a
passage or source document; worth revisiting if the real task set ever
groups multiple items under one source. Also not added: a CLT-based
(rather than Wilson/Bernoulli) interval specifically for the two
continuous-valued scorers (`TOKEN_F1`, `LLM_JUDGE`) — Miller notes the
Bernoulli approximation runs conservative for fractional scores; current
`ModelProfile.ci_method` is `"wilson"` for every scorer, which is the
correct interval for the four binary scorers and a deliberately
conservative (not incorrect) one for the two continuous ones.

---

## 7. Runner-level fixes made in this pass

Not methodology changes, but worth recording since they'd have silently
corrupted results otherwise:

- **Dead-model detection was comparing incompatible types** (`ModelSpec in
  list[SDK Model]`), which is never true regardless of whether the model
  actually exists — every model was being marked dead before a single call
  was made. Fixed to compare on `model_id` directly.
- **Missing price-table entries were silently priced at $0.00** rather than
  raising. `model-set.json` and `price-table.json` had already drifted
  (`google/gemma-4-31b-it` vs. the price table's stale
  `google/gemma-2-2b-it`) — exactly the kind of mismatch this would have
  hidden. `setup()` now validates every target model has a price entry
  before any API calls are made, and `compute_cost()` raises instead of
  defaulting to free.
- **TTFT was hardcoded to `None`.** The non-streaming call had no way to
  observe it. `run_model` now streams (`stream=True`) and records the time
  of the first content chunk separately from total latency.
  **Caveat: this project's sandbox couldn't reach NVIDIA Build / OpenRouter
  to verify live**, so `stream_options={"include_usage": True}` and
  streamed `logprobs` are implemented per the standard OpenAI-compatible
  convention but not confirmed against these specific providers. Smoke-test
  on one item before trusting a full run's TTFT/usage numbers; if a usage
  chunk never arrives, `run_model` prints a loud warning per-call rather
  than silently recording zero cost.
- **`toy-set.json` didn't satisfy the `TaskSet` schema** (`id`/`name` were
  missing; the file used `dataset_name`/`version` instead). Fixed by adding
  the required fields alongside the originals.

---

## 8. Week 5 scope-cut

Per SPEC.md's own description of this milestone: fix the model list, task
set, and item counts based on what Weeks 3-4 taught, and decide now what
shrinks if it has to.

**Staying in scope:** 5-model target ladder, 30-item toy set (15/15
calibration/holdout, all 5 task types represented in calibration), the
statistical additions in Section 6, the per-task-type routing table as the
real deliverable over a single collapsed headline.

**Explicitly deferred to Week 6+, not attempted here:**
- Cascade routing actually simulated against tuned calibration thresholds
  (`simulate_policy`/`choose_call` exist and are unit-tested against
  synthetic data, but haven't run against this project's real results yet)
- `VERIFIER_MODEL`/`SAMPLE_DISAGREEMENT`/`OUTPUT_HEURISTIC` escalation
  signals (only `SELF_REPORTED_CONFIDENCE` is implemented — see Section 3)
- Matplotlib Pareto plots and CSV export (report is text-table only so far)
- CLI restructuring into `modelpicker profile` / `modelpicker route`
  subcommands (checklist item, but a Week 7-8 concern per the milestone
  schedule) — current `modelpicker.py` is one invocation that runs the
  pipeline then reports, which is sufficient through Week 5
- Sourced, non-illustrative price table (`price-table.json` is still
  entirely `is_estimate: true` placeholder values, as flagged in its own
  `notes` field)

If any of these have to shrink further by Week 6 — most likely candidate:
scoping the cascade down to a single escalation signal permanently instead
of eventually implementing all four — that decision gets made and recorded
here when it happens, not silently.