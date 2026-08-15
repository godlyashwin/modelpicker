# Project Spec — Model Selection & Cost/Quality/Latency Profiler

**Intern:** Ashwin Gupta
**Duration:** 8 weeks
**Stack:** Python 3.11+, Pydantic 2, HuggingFace `datasets`, pandas, matplotlib, NVIDIA Build (OpenAI-compatible LLM API)

---

## The Problem

Every team shipping an LLM feature makes the same decision the same way: they read a leaderboard, pick the biggest model on it, and ship. Then they get the bill.

The leaderboard answered a question nobody asked. "86.4% on MMLU" is one model's average across an academic benchmark. What a team actually needs to know is: *for my task, at my quality bar, what is the cheapest and fastest model that's good enough?* Frontier models cost 30-100x what small ones do and are several times slower. A large fraction of real production traffic — classification, extraction, routing, short factual lookups, reformatting — is nowhere near frontier-hard. Teams pay frontier prices for it anyway, because nobody ever measured the alternative on their own data.

The evidence that they're overpaying already exists in the literature: FrugalGPT showed cascades matching GPT-4 quality at a fraction of the cost, RouteLLM showed learned routing does the same. But those are papers about the authors' task sets. There is no tool you can point at *your* task set and *your* candidate models that comes back with an honest answer.

The academic move is another leaderboard. The startup move is a profiler: give it a task set and a ladder of models, get back a quality-vs-cost Pareto frontier, a defensible "cheapest model that clears your bar," and a routing policy you can actually deploy. That tool is your project.

## What You're Building

A standalone Python package with a CLI — call it whatever you like (`modelpicker` is used as a placeholder below) — that:

1. **Ingests a task set** — items with a prompt, a gold answer, and a declared scoring method — and runs every item against every model in a configured ladder, holding the prompt and decoding parameters fixed.
2. **Scores with pluggable scorers** — exact match, normalized match, MCQ letter parsing, numeric tolerance, token-overlap F1, and an LLM judge for free-form answers. One `Scorer` protocol; the runner never knows which one it's using.
3. **Accounts for cost and latency honestly** — real token counts from the API's `usage` field, a versioned price table, and latency measured under stated concurrency with p50/p95, not a single stopwatch reading.
4. **Computes the Pareto frontier** — for each task type, which models are not dominated on (quality, cost) and (quality, latency)? Which are dominated, and by what? Every accuracy number ships with a confidence interval, because a 2-point gap on 100 items is noise.
5. **Recommends and simulates a routing policy** — a static policy (per task type, the cheapest model clearing your quality floor) and a cascade (small model first, escalate on a confidence signal), each simulated on a held-out split against two baselines: always-small and always-large.
6. **Emits a report** — Markdown + CSV + Pareto plots, with a headline every engineer understands: "$X per 1,000 requests at Y% of frontier quality."

End artifacts:
- Public GitHub repo (MIT license)
- Pip-installable package with a CLI anyone can point at their own task set and any OpenAI-compatible endpoint
- A published profile report comparing 5 models across 2+ task types, with a routing recommendation and measured savings
- A 5-slide final presentation

**Territory notes.** Arihaan's suite holds the model fixed and varies the prompt; yours holds the prompt fixed and varies the model — for you, prompt sensitivity is a *confound to control*, not a thing to study, which is why you use one shared prompt as the primary condition. Mohith's playground is an interactive UI for a human comparing outputs side by side; yours is a batch harness that produces frontiers and policies, and has no web UI at all. Rohan judges whether answers are grounded in a source document; you reuse ordinary task scorers and never build claim decomposition. Nothing here touches safety or bias (Mandy) or multi-step agent traces (Jayden) — you are strictly single-turn.

## Pipeline Architecture (suggested — refine in Week 2)

```
┌───────────┐    ┌──────────────┐    ┌───────────┐    ┌──────────────┐    ┌───────────┐
│ Task set  │ -> │ Runner       │ -> │ Scorers   │ -> │ Profiles +   │ -> │ Policy    │
│ + model   │    │ (interleaved,│    │ (pluggable│    │ Pareto       │    │ sim +     │
│ ladder    │    │  concurrency)│    │  protocol)│    │ frontier     │    │ report    │
└───────────┘    └──────────────┘    └───────────┘    └──────────────┘    └───────────┘
                         │                                    │
                    usage tokens                         price table
                    + latency samples                   (versioned, dated)
```

Each stage reads and writes typed Pydantic objects as JSONL, so a run is auditable and every metric can be recomputed offline without re-querying a single model. That matters more here than in most eval projects: you will change the price table long after the runs are done, and re-deriving cost must never mean paying for the calls again.

## The Cost & Latency Accounting Problem

This is the part that doesn't exist in any tool you'll find, and it's the part where this project is easiest to get quietly, expensively wrong. Four traps, all of which have burned real teams:

**1. Cost is modeled, not measured.** NVIDIA Build is free, so there is no bill to read. Your entire cost axis rests on a price table you maintain: `$/1M input tokens` and `$/1M output tokens` per model, each entry carrying a source URL and an `as_of` date. Cite public list prices from the providers who host these models; where you can't find one, mark the entry `is_estimate: true` and say so in the report. Every cost claim you publish is conditional on that table — the report must state it, and the table must be committed to the repo alongside the results.

**2. Cost per token is the wrong unit; cost per *request* is the right one.** A model priced at half the rate but emitting four times the tokens is more expensive. Output length is a lever the *model* pulls, not you, so treat mean completion tokens per model as a first-class measured number — not a footnote. Then take it one step further: **cost per correct answer** (cost per request ÷ accuracy) is usually the number that actually decides the argument, because a cheap model that's wrong half the time isn't cheap.

**3. Latency is a property of the environment, not the model.** You're measuring a shared free endpoint whose load varies by hour. If you run model A for an hour and then model B for an hour, you measured time of day. **Interleave**: round-robin requests across models so every model sees the same load conditions. Report p50 and p95, not the mean — tail latency is what users feel. State your concurrency level in the report, because latency at concurrency 1 and at concurrency 16 are different numbers about different things. Separate time-to-first-token from total time; for streaming UIs they are not the same product decision.

**4. "Good enough" is a statistical claim.** The single most likely way this project produces a wrong answer is declaring a small model adequate on n=50. Every accuracy gets a confidence interval (Wilson for proportions; paired bootstrap when you compare two models on the same items, which you always can, since every model sees identical inputs). Then decide explicitly whether your quality floor must be cleared by the *point estimate* or by the *lower bound* of the interval — those two rules recommend different models, and the sample data is built so you can see exactly that happen.

## Staged Build Path

The project is designed to ramp. Each stage is a working system, not a fraction of one:

- **W1** — Scope doc. Run the Day-1 API snippet from `KICKOFF-TeamD-SummerII.md`. Pick your two task sets and write down, before measuring anything, what quality bar you'd personally accept.
- **W2** — Research & requirements. Read FrugalGPT and *Adding Error Bars to Evals* properly (skim the rest). Build the price table with sources. Hand-build a 30-item toy set with gold answers and declared scoring methods, split into calibration and holdout.
- **W3** — **Prototype**: 2 models × toy set × exact-match scorer, end-to-end CLI producing one table — accuracy, mean tokens in/out, latency, modeled cost. No frontier, no routing, no plots. Crude but complete beats sophisticated but half-finished.
- **W4** — Scale to the full 5-model ladder. Add the interleaved runner, latency percentiles, Wilson intervals, the Pareto frontier, and the first plot. This is where the profiler becomes real.
- **W5** — Mid-project review. Scope-cut point: fix your model list, task sets, and item counts based on what W3–W4 taught you. If the routing work has to shrink to a static policy only, decide that here and say so.
- **W6** — Routing: static policy plus a cascade with one escalation signal, threshold tuned on the calibration split and evaluated on holdout, reported against both baselines.
- **W7** — Scale to 300+ items across 2–3 task types, final report generation, analysis notebook, 5-slide deck (due 48h before the Week 8 meeting).
- **W8** — Pass-off.

## Data Model

See `DataStructures.py`. Key Pydantic types:

- `TaskType` / `ScoringMethod` / `EscalationSignal` / `PolicyKind` / `SplitName` — the enums the pipeline is organized around
- `TaskItem` — one item: prompt, input, options, gold answer, and the scoring method it declares
- `TaskSet` — a named collection of items with a calibration/holdout split
- `ModelSpec` — a model in the ladder: id, family, parameter scale, architecture notes
- `PriceEntry` / `PriceTable` — the versioned, dated, sourced cost model everything downstream depends on
- `TokenUsage` — prompt/completion/total tokens, taken from the API response, never estimated
- `LatencySample` — TTFT, total, and the concurrency the request was issued under
- `ModelCall` — one call: raw output, parsed answer, score, usage, latency, derived cost, retries, errors
- `ScoreResult` — a score in [0, 1] plus the method and (for judged answers) the judge and its rationale
- `ModelProfile` — per (model × task set): accuracy with CI, latency percentiles, mean tokens, cost per 1k requests, cost per correct answer, parse-failure rate
- `ParetoPoint` / `ParetoFrontier` — who's on the frontier, who's dominated, and by whom
- `QualityFloor` — the "good enough" rule, including whether the CI lower bound must clear it
- `RoutingRule` / `RoutingPolicy` — static rules or a cascade with an escalation signal and threshold
- `PolicySimulation` — blended accuracy and cost on a held-out split, escalation rate, and comparison against always-small / always-large
- `ProfileRun` — run-level metadata, config hash, and an embedded snapshot of the price table used
- `Scorer` (protocol) — the pluggable scoring contract

## Sample Data

`sample-data.json` contains a complete worked miniature of the whole pipeline:

- A **price table** of 6 models with `as_of` dates and `is_estimate` flags (illustrative values — replacing them with sourced ones is a Week 2 task)
- A **model ladder** of 5 targets spanning 1B to 122B across four families, plus a separate judge/verifier model
- **12 task items** across five task types, each declaring its scoring method, split 4 calibration / 8 holdout (the calibration items deliberately span four different task types)
- **Example model calls** showing the full record shape — including a case where the 1B model gets it right and the 70B doesn't, so your report generator has to handle it
- **Worked profiles, a worked Pareto frontier, and a worked cascade simulation** with every number arithmetically consistent
- A **`worked_example_check`** block: the derived quantities (costs per 1k, frontier membership, savings percentage) stated explicitly so your Week 3 code has something to reproduce. If your implementation doesn't reproduce these numbers exactly, one of you is wrong — find out which.
- **`examples_of_bad_analysis`** — four ways to produce a confident, wrong recommendation, each with the fix

The sample data is built around a deliberate narrative: the small model is far too weak to route to statically, the second-best model's point estimate clears the quality floor while its confidence interval doesn't, and a cascade beats both baselines — but by noticeably less on holdout than the calibration split promised. All three lessons are load-bearing.

## LLM Access

NVIDIA Build, same as the rest of Team D — free, OpenAI-compatible.

Recommended **target ladder** (spread of size and provenance is the point — you're measuring the shape of the size/quality curve, so a ladder of five 70B models teaches you nothing):

- `meta/llama-3.2-1b-instruct` — the bottom rung; the whole question is how often this is enough
- `google/gemma-2-2b-it` — small, different training pipeline
- `meta/llama-3.3-70b-instruct` — the strong open default most teams reach for
- `mistralai/mixtral-8x22b-instruct` — MoE architecture; sparse models price differently than their parameter count suggests
- `qwen/qwen3-5-122b-a10b` — largest rung, different training distribution

Recommended **judge / cascade verifier**: `deepseek-ai/deepseek-v4-pro` — and keep it **out of the target ladder**. A model that judges answers it's also being scored on is a methodological hole, and if it's the escalation target in your cascade it's marking its own homework twice.

## Design Questions to Answer

Document your reasoning in `DESIGN.md`:

1. **Prompt fairness.** Small models often need more explicit format instructions to score well. One shared prompt for all models is fair but flatters the big ones; a per-model tuned prompt is realistic but introduces a confound you can't separate from model capability. Which is your primary condition, what's the secondary, and what does the report claim about each?
2. **What "good enough" means.** An absolute floor (≥85%) and a relative floor (within 3 points of the best model) recommend different models as the ladder changes. Pick one as the headline. Then answer the harder half: must the floor be cleared by the point estimate or by the CI's lower bound? Your sample data contains a model where those two rules disagree — say which rule you'd defend to someone spending real money on your recommendation.
3. **Escalation signals.** A cascade needs a cheap signal for "the small model probably got this wrong." Self-reported confidence, disagreement across k samples, output length or hedging heuristics, a cheap verifier model — each has a different cost, and the signal's cost counts against your savings. Which did you pick, what does it cost, and how much of the theoretical savings does it eat?
4. **Honest savings.** Tuning an escalation threshold on the same data you report savings on is how cascade papers overstate results. How do you split, and what's the gap between your calibration and holdout numbers? (Report both. The gap is a finding, not an embarrassment.)
5. **Aggregating across task types.** A model can win on classification and lose badly on extraction. Does your headline recommendation collapse to one model, or is the real deliverable a per-task-type routing table? Defend the choice — and if you collapse, say what the collapse hides.

## Deliverable Checklist

By Week 8 you should have:

- [ ] Public GitHub repo (MIT)
- [ ] Python package installable via `pip install -e .`
- [ ] CLI: `modelpicker profile --tasks data/tasks.jsonl --models meta/llama-3.2-1b-instruct,meta/llama-3.3-70b-instruct,qwen/qwen3-5-122b-a10b --prices prices/2026-08.json --concurrency 8 --out results/`
- [ ] CLI: `modelpicker route --results results/ --floor relative:0.03 --policy cascade --out policy/`
- [ ] At least 5 models profiled over 300+ items spanning 2+ task types
- [ ] At least 4 scorers implemented behind the `Scorer` protocol, including one LLM judge
- [ ] Versioned price table committed to the repo, with sources and `as_of` dates
- [ ] Interleaved runner with recorded concurrency; p50/p95 latency and TTFT reported separately
- [ ] Confidence intervals on every accuracy number; paired comparison between the two models your recommendation hinges on
- [ ] Pareto frontier per task type (quality × cost and quality × latency), with dominated models named and their dominators identified
- [ ] Routing policy — static plus cascade — simulated on a held-out split against always-small and always-large baselines
- [ ] Profile report (Markdown + CSV) with a plain-English headline recommendation and its cost/quality consequence
- [ ] `README.md` with reproduction instructions (clone → install → set `NVIDIA_API_KEY` → one command → headline numbers)
- [ ] `DESIGN.md` with methodology decisions
- [ ] Jupyter notebook: where does the size/quality curve flatten? Which item types force escalation?
- [ ] 5-slide final presentation (PDF, due 48h before Week 8 meeting)

## Out of Scope

- **Training or fine-tuning anything** — no distillation, no learned router weights, no fine-tuned classifiers. Off-the-shelf hosted models only. (RouteLLM-style *learned* routing is a stretch goal, not the deliverable.)
- **Prompt optimization** — you control for prompt variation, you don't search over it. Prompt sensitivity is Arihaan's project.
- **Multi-turn conversations, agents, and tool use** — strictly single-turn. Agent traces are Jayden's.
- **Safety, bias, and toxicity scoring** — Mandy's. Your quality axis is task correctness only.
- **A web UI or dashboard** — CLI in, Markdown/CSV/PNG out. Mohith's playground is the interactive tool; yours is the batch one.
- **Self-hosting or local inference** — no vLLM, no GPU provisioning, no measuring your own serving stack. Hosted API endpoints only.

## Stretch Goals

- **Learned routing.** Train a lightweight classifier (embeddings + logistic regression) that predicts, per item, whether the small model will get it right — then route on the prediction instead of a fixed threshold. Compare against your rule-based cascade on the same holdout. This is RouteLLM's idea at intern scale, and it's a genuinely strong portfolio piece.
- **Repeated-sampling economics.** Is *k* samples from the small model with majority voting cheaper than one call to the large model at equal quality? *Large Language Monkeys* says sometimes yes. Find your task set's crossover point — this is the kind of finding people quote.
- **Cost simulator.** Given a traffic mix (60% classification, 30% extraction, 10% free-form) and a monthly request volume, project the monthly bill under each policy. Turns your report into something a founder reads instead of skims.
- **Price-table sensitivity.** Prices move. Re-run your recommendation across a range of price ratios and report at what small/large price ratio your conclusion flips. A recommendation that survives a 3x price shift is worth much more than one that doesn't.
- **Public leaderboard page.** A static HTML page from your profiles, showing the Pareto frontier per task type. Instantly demo-able.

## Resources

- [NVIDIA Build (LLM endpoints)](https://build.nvidia.com)
- [FrugalGPT (Chen, Zaharia & Zou, 2023)](https://arxiv.org/abs/2305.05176) — LLM cascades and cost reduction; the closest prior art to your routing stage. Read this one properly.
- [Adding Error Bars to Evals (Miller, 2024)](https://arxiv.org/abs/2411.00640) — the statistics of LLM evaluation: CIs, paired comparisons, power. Read this one properly too; it's the difference between a recommendation and a guess.
- [RouteLLM (Ong et al., 2024)](https://arxiv.org/abs/2406.18665) — learned routing between a strong and weak model; the stretch goal, made rigorous
- [FrugalML (Chen, Zaharia & Zou, 2020)](https://arxiv.org/abs/2006.07512) — the pre-LLM version of the same idea, on prediction APIs; useful for how they formalize the cost/accuracy optimization
- [Large Language Monkeys (Brown et al., 2024)](https://arxiv.org/abs/2407.21787) — repeated sampling as an inference-compute lever; the repeated-sampling stretch goal
- [Efficiently Scaling Transformer Inference (Pope et al., 2022)](https://arxiv.org/abs/2211.05102) — why latency and throughput trade off the way they do; background for interpreting your latency numbers
- [Chatbot Arena (Chiang et al., 2024)](https://arxiv.org/abs/2403.04132) — how the field currently ranks models, and what that ranking leaves out
- [HELM](https://crfm.stanford.edu/helm/) — Stanford's holistic eval framework; one of the few that reports efficiency alongside accuracy. Study its reporting methodology.
- [Wilson score interval](https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Wilson_score_interval) — the right interval for accuracy proportions; `statsmodels.stats.proportion.proportion_confint(method="wilson")` gives it to you

## Notes for Ashwin specifically

- **This is the most directly commercial project on the team.** Every other eval in this cohort answers "is the model good?" Yours answers "which one should I pay for?" — a question with a dollar figure attached that every engineering team is currently guessing at. You said you want to build tools that have real impact: a profiler that tells a team they can cut inference spend 70% without losing quality *is* that, and it's a portfolio piece that explains itself in one sentence in an interview.
- **Follow the staged build path literally — it's your safety net.** Get the Day-1 API snippet working before anything else, then make W3 your first win: two models, one scorer, one table printed to the terminal. Every week after that upgrades one stage of something that already runs end to end. You never have to hold the whole system in your head at once.
- **The measurement discipline is the skill here, not the plumbing.** Running five models over a task set is a weekend of work. Doing it so the numbers survive someone pushing back — interleaved so latency isn't time-of-day, CIs so a 2-point gap isn't a recommendation, held-out so the cascade savings aren't fiction — is the actual eight weeks, and it's the part that transfers to every job you'll ever have. When you're tempted to skip an error bar, that's the moment the project is happening.
- **When stuck, shrink the problem.** One item, two models, printed to the terminal with the token counts — then scale back up. And ask early: five minutes with your manager or another intern beats an afternoon of silent stalling. Asking early reads as strength on this team, not weakness.
- **You wrote that there's always another challenge to tackle and another skill to develop — that's true here, and it's also the trap.** This project has an infinite tail: more models, more task types, learned routers, sampling economics. In Week 2, read two papers deeply (FrugalGPT + *Adding Error Bars*) rather than eight shallowly, and let the W5 review be where you cut scope without guilt. A five-model profile with defensible error bars beats a twelve-model one with shaky ones, every time.