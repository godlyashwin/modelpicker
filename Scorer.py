from pathlib import Path
import json
import os
import random
import re
import string
from collections import defaultdict, Counter
from statistics import mean, mode
from math import inf, sqrt, floor, ceil
from typing import Optional

from nltk.tokenize import word_tokenize
from openai import OpenAI
from dotenv import load_dotenv

from DataStructures import (
    TaskItem,
    TaskSet,
    TaskType,
    ModelSpec,
    PriceTable,
    PriceEntry,
    TokenUsage,
    LatencySample,
    ScoreResult,
    ModelCall,
    ScoringMethod,
    ModelProfile,
    ParetoFrontier,
    ParetoPoint,
    PolicySimulation,
    RoutingDecision,
    RoutingPolicy,
    PolicyKind,
    EscalationSignal,
    SplitName,
    QualityFloor,
    Scorer,
)

load_dotenv()


# ---------------------------------------------------------------------------
# Scorers — one class per ScoringMethod, conforming to the `Scorer` protocol
# in DataStructures.py. This is the piece the pipeline diagram calls out on
# its own: "pluggable scorers ... one Scorer protocol; the runner never knows
# which one it's using." Runner.py currently has this logic inline in an
# async score_answer() function — that's the thing being fixed here. Wire
# runner.py to call `score(item, raw_output)` below instead of duplicating
# this logic once you're happy with it.
#
# Every score() implementation here is deterministic and synchronous per the
# protocol's contract, EXCEPT LLMJudgeScorer, which calls out to a judge
# model. It still exposes a plain synchronous `score()` — scoring happens as
# its own pipeline stage after the runner has finished collecting raw
# outputs, so there's no concurrency to preserve here the way there is in
# the runner's model-calling loop.
# ---------------------------------------------------------------------------

_ARTICLES = {"a", "an", "the"}


def normalize(text: str) -> str:
    """Lowercase, strip punctuation/quotes/articles, collapse whitespace."""
    if text is None:
        return ""

    text = text.lower()

    text = (
        text.replace("\u201c", '"')
            .replace("\u201d", '"')
            .replace("\u2019", "'")
            .replace("\u2018", "'")
    )

    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()

    tokens = [t for t in text.split() if t not in _ARTICLES]
    return " ".join(tokens)


class ExactMatchScorer:
    name = "exact_match"
    method = ScoringMethod.EXACT_MATCH

    def score(self, item: TaskItem, raw_output: str) -> ScoreResult:
        pred = raw_output.strip()
        correct = pred == item.gold_answer.strip()
        return ScoreResult(
            value=1.0 if correct else 0.0,
            method=self.method,
            parsed_answer=pred,
            parse_failed=False,
        )


class NormalizedMatchScorer:
    name = "normalized_match"
    method = ScoringMethod.NORMALIZED_MATCH

    def score(self, item: TaskItem, raw_output: str) -> ScoreResult:
        # BUGFIX vs. runner.py: normalize() already returns a joined string.
        # The original code did `" ".join(normalize(raw_output))`, which
        # re-splits that string into individual characters. Fixed here.
        pred = normalize(raw_output)
        gold = normalize(item.gold_answer)
        correct = gold in pred
        return ScoreResult(
            value=1.0 if correct else 0.0,
            method=self.method,
            parsed_answer=raw_output.strip(),
            parse_failed=False,
        )


class NumericToleranceScorer:
    name = "numeric_tolerance"
    method = ScoringMethod.NUMERIC_TOLERANCE

    def score(self, item: TaskItem, raw_output: str) -> ScoreResult:
        try:
            pred = float(raw_output.strip())
            gold = float(item.gold_answer)
            tol = item.scoring_params.get("tolerance", 0.0)
            correct = abs(pred - gold) <= tol
            return ScoreResult(
                value=1.0 if correct else 0.0,
                method=self.method,
                parsed_answer=str(pred),
                parse_failed=False,
            )
        except (TypeError, ValueError):
            return ScoreResult(
                value=0.0,
                method=self.method,
                parsed_answer=None,
                parse_failed=True,
            )


class MCQLetterScorer:
    name = "mcq_letter"
    method = ScoringMethod.MCQ_LETTER

    def score(self, item: TaskItem, raw_output: str) -> ScoreResult:
        text = raw_output.strip().upper()
        match = re.search(r"\b([ABCD])\b", text)

        if not match:
            return ScoreResult(
                value=0.0,
                method=self.method,
                parsed_answer=None,
                parse_failed=True,
            )

        idx = ord(match.group(1)) - ord("A")

        if not item.options or idx >= len(item.options):
            return ScoreResult(
                value=0.0,
                method=self.method,
                parsed_answer=None,
                parse_failed=True,
            )

        # Gold answers for MCQ are stored as option TEXT, never letters —
        # see TaskItem docstring — so we map the parsed letter through
        # item.options before comparing.
        pred = item.options[idx]
        correct = normalize(pred) == normalize(item.gold_answer)
        return ScoreResult(
            value=1.0 if correct else 0.0,
            method=self.method,
            parsed_answer=pred,
            parse_failed=False,
        )


class TokenF1Scorer:
    name = "token_f1"
    method = ScoringMethod.TOKEN_F1

    def score(self, item: TaskItem, raw_output: str) -> ScoreResult:
        pred_tokens = [
            t for t in word_tokenize(raw_output.lower())
            if t not in string.punctuation
        ]
        gold_tokens = [
            t for t in word_tokenize(item.gold_answer.lower())
            if t not in string.punctuation
        ]

        if len(pred_tokens) == 0 or len(gold_tokens) == 0:
            value = 1.0 if len(pred_tokens) == len(gold_tokens) == 0 else 0.0
            return ScoreResult(
                value=value,
                method=self.method,
                parsed_answer=raw_output.strip(),
                parse_failed=False,
            )

        pred_counter = Counter(pred_tokens)
        gold_counter = Counter(gold_tokens)
        num_same = sum((pred_counter & gold_counter).values())

        if num_same == 0:
            return ScoreResult(
                value=0.0,
                method=self.method,
                parsed_answer=raw_output.strip(),
                parse_failed=False,
            )

        precision = num_same / len(pred_tokens)
        recall = num_same / len(gold_tokens)
        f1 = (2 * precision * recall) / (precision + recall)

        return ScoreResult(
            value=f1,
            method=self.method,
            parsed_answer=raw_output.strip(),
            parse_failed=False,
        )


class LLMJudgeScorer:
    """Asks a judge model to rate the candidate answer against the gold
    answer under a rubric. Per the Scorer protocol docstring: must run at
    temperature 0, and must record its model id on the ScoreResult so a
    scored run can be traced back to who scored it.

    IMPORTANT: judge_model must stay out of the target ladder (SPEC.md
    "Recommended judge / cascade verifier"). Default here is a placeholder —
    point it at your actual judge (e.g. deepseek-ai/deepseek-v4-pro) via the
    constructor, not by editing this class.
    """

    name = "llm_judge"
    method = ScoringMethod.LLM_JUDGE

    def __init__(self, judge_model: str = "openai/gpt-oss-120b", client: OpenAI | None = None):
        self.judge_model = judge_model
        self._client = client  # built lazily — see _get_client()

    def _get_client(self) -> OpenAI:
        # Built on first real use, not at construction time, so importing
        # this module (or registering LLMJudgeScorer in SCORERS) doesn't
        # blow up for people who never run a summarization task and never
        # set API_KEY.
        if self._client is None:
            self._client = OpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=os.getenv("API_KEY"),
            )
        return self._client

    def score(self, item: TaskItem, raw_output: str) -> ScoreResult:
        rubric = item.scoring_params.get("rubric", "Judge semantic correctness.")

        judge_prompt = f"""You are an impartial evaluation judge.

Evaluate the candidate answer using ONLY the provided rubric.

Rubric:
{rubric}

Reference Answer:
{item.gold_answer}

Candidate Answer:
{raw_output.strip()}

Return ONLY valid JSON.

{{
    "score": <number between 0.0 and 1.0>,
    "reason": "<one short sentence>"
}}
"""

        try:
            response = self._get_client().chat.completions.create(
                model=self.judge_model,
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=0,
                top_p=1,
                max_tokens=200,
                stream=False,
            )

            text = response.choices[0].message.content.strip()
            result = json.loads(text)
            value = max(0.0, min(1.0, float(result["score"])))
            reason = result.get("reason", "")

            return ScoreResult(
                value=value,
                method=self.method,
                parsed_answer=raw_output.strip(),
                parse_failed=False,
                judge_model=self.judge_model,
                judge_rationale=reason,
            )

        except Exception as e:
            return ScoreResult(
                value=0.0,
                method=self.method,
                parsed_answer=raw_output.strip(),
                parse_failed=True,
                judge_model=self.judge_model,
                judge_rationale=f"Judge failed: {e}",
            )


# Registry: the runner (or anything else) dispatches through this and never
# needs to know which concrete Scorer it's using.
SCORERS: dict[ScoringMethod, Scorer] = {
    ScoringMethod.EXACT_MATCH: ExactMatchScorer(),
    ScoringMethod.NORMALIZED_MATCH: NormalizedMatchScorer(),
    ScoringMethod.NUMERIC_TOLERANCE: NumericToleranceScorer(),
    ScoringMethod.MCQ_LETTER: MCQLetterScorer(),
    ScoringMethod.TOKEN_F1: TokenF1Scorer(),
    ScoringMethod.LLM_JUDGE: LLMJudgeScorer(),
}


def score_output(item: TaskItem, raw_output: str) -> ScoreResult:
    """Look up the right Scorer for item.scoring_method and run it.

    This is the one function runner.py should call instead of carrying its
    own inline scoring branches.
    """
    scorer = SCORERS.get(item.scoring_method)
    if scorer is None:
        raise ValueError(f"No Scorer registered for {item.scoring_method}")
    return scorer.score(item, raw_output)


def load_results(path: str | Path) -> list[ModelCall]:
    """
    Load every ModelCall stored in a JSONL file.

    Each line of the file should contain one serialized ModelCall.
    """

    results: list[ModelCall] = []

    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            data = json.loads(line)

            results.append(ModelCall.model_validate(data))

    return results


def load_task_set(path: str | Path) -> TaskSet:
    """
    Load a TaskSet either from a raw TaskSet JSON file, or from a
    sample-data.json-style file that nests it under a "task_set" key.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "task_set" in data:
        data = data["task_set"]

    return TaskSet(**data)


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def wilson_interval(successes: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """
    Wilson score interval for a binomial proportion (SPEC.md's recommended
    method — see the Wilson score interval link in Resources).

    `successes` doesn't need to be an integer: for TOKEN_F1's partial-credit
    scores, passing sum(scores) as `successes` treats each item's [0,1] score
    as a fractional "success" out of n trials, which is the standard way to
    extend a proportion interval to a continuous score. It collapses to the
    ordinary binomial case for binary scorers.

    Verified against sample-data.json's worked_example: wilson_interval(0.78*200, 200)
    reproduces (0.7176, 0.8318) to 4 decimal places, and likewise for the other
    three profiled models and both policy_simulations entries.
    """
    if n <= 0:
        return 0.0, 0.0

    p_hat = successes / n
    denom = 1 + (z ** 2) / n
    center = p_hat + (z ** 2) / (2 * n)
    margin = z * sqrt(p_hat * (1 - p_hat) / n + (z ** 2) / (4 * n ** 2))

    low = (center - margin) / denom
    high = (center + margin) / denom

    return max(0.0, low), min(1.0, high)


def _percentile(values: list[float], pct: float) -> float:
    """Linear-interpolation percentile (numpy's default 'linear' method),
    implemented without a numpy dependency."""
    if not values:
        return 0.0

    s = sorted(values)
    if len(s) == 1:
        return s[0]

    k = (len(s) - 1) * (pct / 100)
    f = floor(k)
    c = ceil(k)

    if f == c:
        return s[int(k)]

    return s[f] * (c - k) + s[c] * (k - f)


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------

def build_model_profiles(
    calls: list[ModelCall],
    task_set: TaskSet,
    price_table_id: str,
    include_pooled: bool = True,
    include_per_task_type: bool = True,
) -> list[ModelProfile]:
    """
    Compute one ModelProfile per (model_id, task_type, split) group found in
    `calls`, using `task_set` to look up each item's task_type and split
    (ModelCall itself only carries item_id — the pairing key — not those,
    so a TaskSet is required here rather than optional).

    task_type=None rows are pooled across all task types for that split, per
    ModelProfile's own docstring ("None = pooled across types"). Per-task-type
    rows are emitted too by default, since the Pareto frontier and routing
    checklist items in SPEC.md both need quality broken out by task type, not
    just pooled — "A model can win on classification and lose badly on
    extraction" is exactly the thing pooling would hide.

    Calls whose item_id isn't found in `task_set` are skipped rather than
    guessed at.
    """
    items_by_id = {item.id: item for item in task_set.items}

    groups: dict[tuple[str, Optional[TaskType], SplitName], list[ModelCall]] = defaultdict(list)

    for call in calls:
        item = items_by_id.get(call.item_id)
        if item is None:
            continue

        if include_pooled:
            groups[(call.model_id, None, item.split)].append(call)
        if include_per_task_type:
            groups[(call.model_id, item.task_type, item.split)].append(call)

    profiles: list[ModelProfile] = []

    for (model_id, task_type, split), group_calls in groups.items():
        profiles.append(
            _profile_for_group(
                model_id=model_id,
                task_set_id=task_set.id,
                task_type=task_type,
                split=split,
                calls=group_calls,
                price_table_id=price_table_id,
            )
        )

    return profiles


def _profile_for_group(
    model_id: str,
    task_set_id: str,
    task_type: Optional[TaskType],
    split: SplitName,
    calls: list[ModelCall],
    price_table_id: str,
) -> ModelProfile:
    n = len(calls)

    # Accuracy counts every call, including errors — ModelCall's own docstring
    # is explicit: "error: ... score is 0 and the item counts". Excluding
    # failures from the denominator would quietly reward a flaky model by
    # only grading it on the requests it managed to complete.
    scores = [c.score.value for c in calls]
    accuracy = mean(scores) if n else 0.0
    ci_low, ci_high = wilson_interval(sum(scores), n)

    # Tokens/latency/cost, by contrast, are computed over calls that actually
    # completed. A failed call's zero token count isn't "0 tokens for this
    # request" — it's the absence of a request — and averaging zeros in would
    # understate cost/latency rather than penalize the failure. error_rate
    # below is what carries that penalty instead.
    successful = [c for c in calls if c.error is None]

    mean_prompt_tokens = mean(c.usage.prompt_tokens for c in successful) if successful else 0.0
    mean_completion_tokens = mean(c.usage.completion_tokens for c in successful) if successful else 0.0

    latencies = [c.latency.total_ms for c in successful]
    latency_p50 = _percentile(latencies, 50)
    latency_p95 = _percentile(latencies, 95)

    ttfts = [c.latency.ttft_ms for c in successful if c.latency.ttft_ms is not None]
    latency_ttft_p50 = _percentile(ttfts, 50) if ttfts else None

    # Concurrency is a property of how the run was executed, not something
    # the caller should have to repeat as a separate argument — derive it
    # from the calls themselves (the most common value, in case of any
    # irregular retries issued at a different concurrency).
    concurrency = mode(c.latency.concurrency for c in successful) if successful else 0

    costs = [c.cost_usd for c in successful if c.cost_usd is not None]
    cost_per_request = mean(costs) if costs else 0.0
    cost_per_1k = cost_per_request * 1000
    cost_per_correct = (cost_per_request / accuracy) if accuracy > 0 else inf

    parse_failure_rate = mean(1.0 if c.score.parse_failed else 0.0 for c in calls) if n else 0.0
    error_rate = mean(1.0 if c.error is not None else 0.0 for c in calls) if n else 0.0

    return ModelProfile(
        model_id=model_id,
        task_set_id=task_set_id,
        task_type=task_type,
        split=split,
        n_items=n,
        accuracy=accuracy,
        ci_low=ci_low,
        ci_high=ci_high,
        ci_method="wilson",
        mean_prompt_tokens=mean_prompt_tokens,
        mean_completion_tokens=mean_completion_tokens,
        latency_p50_ms=latency_p50,
        latency_p95_ms=latency_p95,
        latency_ttft_p50_ms=latency_ttft_p50,
        concurrency=concurrency,
        cost_per_request_usd=cost_per_request,
        cost_per_1k_requests_usd=cost_per_1k,
        cost_per_correct_answer_usd=cost_per_correct,
        parse_failure_rate=parse_failure_rate,
        error_rate=error_rate,
        price_table_id=price_table_id,
    )


# ---------------------------------------------------------------------------
# Pareto frontier
# ---------------------------------------------------------------------------

def _axis_value(profile: ModelProfile, axis: str) -> float:
    if axis == "cost":
        return profile.cost_per_1k_requests_usd
    if axis == "latency":
        return profile.latency_p95_ms
    raise ValueError(f"Unknown axis: {axis!r} (expected 'cost' or 'latency')")


def dominates(a: ModelProfile, b: ModelProfile, axis: str) -> bool:
    """
    Returns True if model 'a' dominates model 'b' on (accuracy, axis).

    Higher accuracy is better. Lower cost/latency is better.
    """
    a_axis = _axis_value(a, axis)
    b_axis = _axis_value(b, axis)

    not_worse = a.accuracy >= b.accuracy and a_axis <= b_axis
    strictly_better = a.accuracy > b.accuracy or a_axis < b_axis

    return not_worse and strictly_better


def compute_pareto_frontier(profiles: list[ModelProfile], axis: str) -> ParetoFrontier:
    """
    Compute the Pareto frontier for one axis ('cost' or 'latency') over a set
    of ModelProfiles that all share the same task_set_id, task_type, and
    split — pass one group at a time (e.g. one entry per model, for a single
    task type on the holdout split). Mixing groups would compare accuracy
    numbers that aren't measuring the same thing.

    Matches sample-data.json's worked_example.pareto_frontier shape exactly:
    a ParetoPoint per model with on_frontier / dominated_by, plus the
    frontier_model_ids summary list.
    """
    if not profiles:
        raise ValueError("compute_pareto_frontier requires at least one profile")

    task_set_id = profiles[0].task_set_id
    task_type = profiles[0].task_type
    split = profiles[0].split

    for p in profiles:
        if p.task_set_id != task_set_id or p.task_type != task_type or p.split != split:
            raise ValueError(
                "All profiles passed to compute_pareto_frontier must share the same "
                "(task_set_id, task_type, split) — pass one group at a time, e.g. via "
                "group_profiles_for_frontier()."
            )

    points: list[ParetoPoint] = []

    for candidate in profiles:
        dominators = [
            other.model_id
            for other in profiles
            if other.model_id != candidate.model_id and dominates(other, candidate, axis)
        ]
        points.append(
            ParetoPoint(
                model_id=candidate.model_id,
                quality=candidate.accuracy,
                cost_axis_value=_axis_value(candidate, axis),
                on_frontier=len(dominators) == 0,
                dominated_by=dominators,
            )
        )

    points.sort(key=lambda p: (-p.quality, p.cost_axis_value))

    return ParetoFrontier(
        task_set_id=task_set_id,
        task_type=task_type,
        axis=axis,
        points=points,
        frontier_model_ids=[p.model_id for p in points if p.on_frontier],
    )


def group_profiles_for_frontier(
    profiles: list[ModelProfile],
) -> dict[tuple[str, Optional[TaskType], SplitName], list[ModelProfile]]:
    """Bucket a flat profile list into the (task_set_id, task_type, split)
    groups compute_pareto_frontier expects one at a time."""
    groups: dict[tuple[str, Optional[TaskType], SplitName], list[ModelProfile]] = defaultdict(list)
    for p in profiles:
        groups[(p.task_set_id, p.task_type, p.split)].append(p)
    return groups


def compute_all_pareto_frontiers(profiles: list[ModelProfile]) -> list[ParetoFrontier]:
    """
    Convenience wrapper: group `profiles` by (task_set_id, task_type, split)
    and compute both the cost and latency frontier for each group — this is
    the "quality × cost and quality × latency ... per task type" deliverable
    from SPEC.md's checklist in one call.
    """
    frontiers: list[ParetoFrontier] = []
    for group in group_profiles_for_frontier(profiles).values():
        if len(group) < 2:
            continue  # a "frontier" of one model isn't a comparison
        frontiers.append(compute_pareto_frontier(group, axis="cost"))
        frontiers.append(compute_pareto_frontier(group, axis="latency"))
    return frontiers


# ---------------------------------------------------------------------------
# Routing policies
# ---------------------------------------------------------------------------
#
# RoutingPolicy (DataStructures.py) is a plain Pydantic data model — it
# describes a policy, it doesn't execute one. The original scorer.py had
# execution logic (`.choose()`) living on a `@dataclass` subclass of a
# Pydantic BaseModel, which doesn't work: mixing `@dataclass` onto a
# pydantic-metaclass base bypasses pydantic's own __init__/validation, so
# `threshold` was never a real, validated field, and the class couldn't be
# constructed the way every other RoutingPolicy is (RoutingPolicy(...)),
# round-tripped through JSON, or type-checked against the schema everything
# else in this file uses.
#
# Fix: keep RoutingPolicy as a plain, serializable data model (as
# DataStructures.py defines it), and put execution logic in a free function
# that dispatches on `policy.kind`. This also means the sample cascade policy
# in sample-data.json ("pol-cascade-conf-070" — primary/escalation model,
# self_reported_confidence signal, threshold 0.7) can be loaded straight from
# JSON and run without a bespoke subclass for every policy shape.

def choose_call(
    policy: RoutingPolicy,
    candidates: list[ModelCall],
    item: Optional[TaskItem] = None,
) -> ModelCall:
    """
    Pick one ModelCall from `candidates` (all candidates are for the same
    item, one per model) according to `policy`. `item` is required for
    STATIC/BASELINE policies whose rules reference item properties (e.g.
    task_type); CASCADE policies don't need it.
    """
    successful = [c for c in candidates if c.error is None]
    if not successful:
        raise ValueError("No successful model calls to route between.")

    if policy.kind in (PolicyKind.STATIC, PolicyKind.BASELINE):
        return _choose_by_rules(policy, successful, item)

    if policy.kind == PolicyKind.CASCADE:
        return _choose_cascade(policy, successful)

    raise ValueError(f"Unknown policy kind: {policy.kind!r}")


def _rule_matches(condition: str, item: Optional[TaskItem]) -> bool:
    condition = condition.strip()

    if condition == "always":
        return True

    match = re.match(r"task_type\s*==\s*['\"]([\w_]+)['\"]", condition)
    if match:
        if item is None:
            raise ValueError(f"Rule {condition!r} needs `item` but none was passed to choose_call().")
        return item.task_type.value == match.group(1)

    # Deliberately narrow: this is an internal DSL, not user input, and a
    # silent no-match on an unrecognized condition is worse than an error —
    # it would make a policy quietly fall through to the next rule.
    raise NotImplementedError(
        f"RoutingRule condition {condition!r} isn't one of the supported forms "
        "('always' or \"task_type == '...'\"). Extend _rule_matches to add more."
    )


def _choose_by_rules(
    policy: RoutingPolicy,
    candidates: list[ModelCall],
    item: Optional[TaskItem],
) -> ModelCall:
    for rule in policy.rules:
        if _rule_matches(rule.condition, item):
            match = next((c for c in candidates if c.model_id == rule.model_id), None)
            if match is not None:
                return match

    raise ValueError(
        f"No rule in policy {policy.id!r} matched "
        f"item {item.id if item else '<unknown>'} against the available candidates."
    )


def _choose_cascade(policy: RoutingPolicy, candidates: list[ModelCall]) -> ModelCall:
    if policy.primary_model_id is None or policy.escalation_model_id is None:
        raise ValueError(f"Cascade policy {policy.id!r} needs primary_model_id and escalation_model_id set.")

    primary = next((c for c in candidates if c.model_id == policy.primary_model_id), None)

    if primary is None:
        # Primary unavailable for this item — fall back per policy, or
        # straight to the escalation model if no fallback is configured.
        fallback_id = policy.fallback_model_id or policy.escalation_model_id
        fallback = next((c for c in candidates if c.model_id == fallback_id), None)
        if fallback is None:
            raise ValueError(f"Neither primary nor fallback model available for cascade {policy.id!r}.")
        return fallback

    if policy.escalation_signal != EscalationSignal.SELF_REPORTED_CONFIDENCE:
        # SAMPLE_DISAGREEMENT / VERIFIER_MODEL / OUTPUT_HEURISTIC each need
        # their own branch (and their own cost accounting into
        # escalation_signal_cost_per_request_usd) — not implemented yet.
        raise NotImplementedError(
            f"choose_call only implements the "
            f"{EscalationSignal.SELF_REPORTED_CONFIDENCE.value!r} escalation signal so far; "
            f"{policy.escalation_signal!r} needs its own branch in _choose_cascade()."
        )

    confidence = primary.self_reported_confidence if primary.self_reported_confidence is not None else 0.0
    threshold = policy.escalation_threshold if policy.escalation_threshold is not None else 0.75

    if confidence >= threshold:
        return primary

    escalation = next((c for c in candidates if c.model_id == policy.escalation_model_id), None)
    return escalation if escalation is not None else primary


def _select_model(grouped: dict[str, list[ModelCall]], model_id: str) -> list[ModelCall]:
    """Pick each item's call from one specific model — used to compute the
    always-small / always-large baselines PolicySimulation always reports
    alongside a policy's blended numbers."""
    selected = []
    for item_id, candidates in grouped.items():
        match = next((c for c in candidates if c.model_id == model_id), None)
        if match is None:
            raise ValueError(f"No call from {model_id!r} found for item {item_id!r}.")
        selected.append(match)
    return selected


def simulate_policy(
    calls: list[ModelCall],
    task_set: TaskSet,
    policy: RoutingPolicy,
    split: SplitName,
    baseline_small_model_id: str,
    baseline_large_model_id: str,
) -> tuple[PolicySimulation, list[RoutingDecision]]:
    """
    Replay a routing policy over previously collected ModelCalls, restricted
    to items in `split`.

    SPEC.md is explicit that thresholds get tuned on CALIBRATION and reported
    on HOLDOUT, never the reverse — `split` is a required argument rather
    than an optional one so that choice has to be made explicitly at every
    call site instead of silently defaulting.

    Returns (PolicySimulation, routing_decisions): PolicySimulation matches
    the DataStructures.py schema exactly (no routing_decisions field there —
    it's an aggregate). routing_decisions is the per-item audit trail, meant
    to be persisted separately (e.g. its own JSONL) for anyone who wants to
    see which items escalated.
    """
    items_by_id = {item.id: item for item in task_set.items}

    grouped: dict[str, list[ModelCall]] = defaultdict(list)
    for call in calls:
        item = items_by_id.get(call.item_id)
        if item is None or item.split != split:
            continue
        grouped[call.item_id].append(call)

    if not grouped:
        raise ValueError(f"No calls found on split={split.value!r} for task_set {task_set.id!r}.")

    routing_decisions: list[RoutingDecision] = []
    chosen_calls: list[ModelCall] = []

    for item_id, candidates in grouped.items():
        item = items_by_id[item_id]
        chosen = choose_call(policy, candidates, item)
        chosen_calls.append(chosen)

        escalated = (
            policy.kind == PolicyKind.CASCADE
            and policy.primary_model_id is not None
            and chosen.model_id != policy.primary_model_id
        )

        routing_decisions.append(
            RoutingDecision(
                item_id=item_id,
                selected_model=chosen.model_id,
                score=chosen.score.value,
                latency_ms=chosen.latency.total_ms,
                cost_usd=chosen.cost_usd or 0.0,
                confidence=chosen.self_reported_confidence,
                escalated=escalated,
            )
        )

    n = len(chosen_calls)
    scores = [c.score.value for c in chosen_calls]

    blended_accuracy = mean(scores)
    blended_ci_low, blended_ci_high = wilson_interval(sum(scores), n)
    blended_cost_per_1k = mean((c.cost_usd or 0.0) for c in chosen_calls) * 1000
    blended_latency_p95 = _percentile([c.latency.total_ms for c in chosen_calls], 95)

    escalation_rate = (
        mean(1.0 if d.escalated else 0.0 for d in routing_decisions)
        if policy.kind == PolicyKind.CASCADE
        else None
    )

    small_calls = _select_model(grouped, baseline_small_model_id)
    large_calls = _select_model(grouped, baseline_large_model_id)

    baseline_small_accuracy = mean(c.score.value for c in small_calls)
    baseline_small_cost_per_1k = mean((c.cost_usd or 0.0) for c in small_calls) * 1000
    baseline_large_accuracy = mean(c.score.value for c in large_calls)
    baseline_large_cost_per_1k = mean((c.cost_usd or 0.0) for c in large_calls) * 1000

    cost_savings_vs_large = (
        1 - (blended_cost_per_1k / baseline_large_cost_per_1k)
        if baseline_large_cost_per_1k > 0 else 0.0
    )
    quality_retained_vs_large = (
        blended_accuracy / baseline_large_accuracy
        if baseline_large_accuracy > 0 else 0.0
    )

    simulation = PolicySimulation(
        policy_id=policy.id,
        task_set_id=task_set.id,
        split=split,
        n_items=n,
        blended_accuracy=blended_accuracy,
        blended_ci_low=blended_ci_low,
        blended_ci_high=blended_ci_high,
        blended_cost_per_1k_requests_usd=blended_cost_per_1k,
        blended_latency_p95_ms=blended_latency_p95,
        escalation_rate=escalation_rate,
        baseline_small_accuracy=baseline_small_accuracy,
        baseline_small_cost_per_1k_usd=baseline_small_cost_per_1k,
        baseline_large_accuracy=baseline_large_accuracy,
        baseline_large_cost_per_1k_usd=baseline_large_cost_per_1k,
        cost_savings_vs_large=cost_savings_vs_large,
        quality_retained_vs_large=quality_retained_vs_large,
        notes="",
    )

    return simulation, routing_decisions

# ---------------------------------------------------------------------------
# Model-vs-model comparison (Miller 2024, "Adding Error Bars to Evals")
# and cascade justification (Chen/Zaharia/Zou 2023, "FrugalGPT")
#
# Both papers were assigned reading in SPEC.md's Week 2; the pieces below are
# what the pipeline was missing to actually use them, not just cite them.
# ---------------------------------------------------------------------------

def paired_bootstrap(
    calls_a: list[ModelCall],
    calls_b: list[ModelCall],
    n_resamples: int = 10000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """
    Paired-bootstrap CI on the accuracy difference (a - b), joined on
    item_id — Miller (2024) Section 4. Paired beats the naive
    sqrt(SE_a^2 + SE_b^2) unpaired formula whenever the two models' scores
    are positively correlated across the same questions (both models tend
    to do well on easy items, worse on hard ones), which is the normal case
    here since every model in this pipeline sees identical inputs. The
    correlation is what SPEC.md's line 63 means by "paired bootstrap when
    you compare two models on the same items, which you always can."

    Returns (mean_diff, ci_low, ci_high) for (a - b). A CI that excludes 0
    is evidence the two models are genuinely different on this task set,
    not noise — a CI that straddles 0 means don't bet the recommendation on
    that ordering yet, however large the point-estimate gap looks.

    Resamples whole item-pairs together (not each model's scores
    independently) — that's what "paired" means here: every bootstrap
    resample re-draws matched (score_a_i, score_b_i) pairs, preserving
    whatever correlation exists between them.
    """
    by_item_a = {c.item_id: c.score.value for c in calls_a}
    by_item_b = {c.item_id: c.score.value for c in calls_b}

    shared_ids = sorted(set(by_item_a) & set(by_item_b))
    if not shared_ids:
        raise ValueError(
            "No shared item_ids between the two call sets — a paired comparison "
            "requires both models to have been run on identical inputs."
        )

    diffs = [by_item_a[i] - by_item_b[i] for i in shared_ids]
    n = len(diffs)
    mean_diff = mean(diffs)

    rng = random.Random(seed)
    resampled_means = []
    for _ in range(n_resamples):
        resample = [diffs[rng.randrange(n)] for _ in range(n)]
        resampled_means.append(mean(resample))

    resampled_means.sort()
    alpha = 1 - ci
    lo_idx = max(0, int((alpha / 2) * n_resamples))
    hi_idx = min(n_resamples - 1, int((1 - alpha / 2) * n_resamples) - 1)

    return mean_diff, resampled_means[lo_idx], resampled_means[hi_idx]


def maximum_performance_improvement(
    calls_a: list[ModelCall],
    calls_b: list[ModelCall],
) -> dict[str, float]:
    """
    MPI (Chen, Zaharia & Zou 2023 — FrugalGPT, Section on cascade motivation):
    among shared items, what fraction does A get right that B gets wrong,
    and vice versa? This is the number that justifies a cascade instead of
    just deploying the Pareto-optimal single model — a model that's worse
    *on average* can still be worth escalating to if it corrects the primary
    model's mistakes on a meaningful slice of items. FrugalGPT's own numbers:
    up to 6% (HEADLINES) and 13% (COQA) of items were only right in the
    cheaper model.

    "Right" here is score >= 1.0, which is exact for the binary scorers
    (EXACT_MATCH, NORMALIZED_MATCH, MCQ_LETTER, NUMERIC_TOLERANCE) but a
    poor fit for TOKEN_F1/LLM_JUDGE's continuous scores, where landing on
    precisely 1.0 is rare — treat MPI on those task types as approximate,
    or threshold "right" at something below 1.0 for that scorer if you need
    it to mean something there.
    """
    by_item_a = {c.item_id: c.score.value for c in calls_a}
    by_item_b = {c.item_id: c.score.value for c in calls_b}
    shared_ids = sorted(set(by_item_a) & set(by_item_b))

    n = len(shared_ids)
    if n == 0:
        raise ValueError("No shared item_ids between the two call sets.")

    a_right_b_wrong = sum(1 for i in shared_ids if by_item_a[i] >= 1.0 and by_item_b[i] < 1.0)
    b_right_a_wrong = sum(1 for i in shared_ids if by_item_b[i] >= 1.0 and by_item_a[i] < 1.0)

    return {
        "n_items": n,
        "mpi_a_over_b": a_right_b_wrong / n,
        "mpi_b_over_a": b_right_a_wrong / n,
    }


def evaluate_quality_floor(
    profiles: list[ModelProfile],
    floor: QualityFloor,
) -> dict[str, bool]:
    """
    Which models in `profiles` clear `floor` — SPEC.md's Design Question 2.
    `profiles` should be one group (same task_set_id/task_type/split), same
    caveat as compute_pareto_frontier: mixing groups compares accuracy
    numbers that aren't measuring the same thing.

    If floor.require_ci_lower_bound is False, a model clears the floor on
    its point estimate. If True, it must clear even at its worst plausible
    accuracy (ci_low) — the stricter rule, and the one this project's
    DESIGN.md defends as the headline (see DESIGN.md Question 2). The two
    rules can disagree, by design — SPEC.md's sample data is built so you
    can see exactly that happen, and evaluate_quality_floor is what lets a
    report show both.
    """
    if not profiles:
        return {}

    if floor.kind == "absolute":
        threshold = floor.value
    elif floor.kind == "relative":
        best = max(p.accuracy for p in profiles)
        threshold = best - floor.value
    else:
        raise ValueError(f"Unknown QualityFloor kind: {floor.kind!r} (expected 'absolute' or 'relative')")

    return {
        p.model_id: (p.ci_low if floor.require_ci_lower_bound else p.accuracy) >= threshold
        for p in profiles
    }