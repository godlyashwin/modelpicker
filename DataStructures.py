"""
Model selection & cost/quality/latency profiler — data structures.

These Pydantic v2 models define the contract between pipeline stages:
task set -> runner -> scorers -> profiles/Pareto -> routing policy -> report.

Two invariants shape everything below:

  1. Every model sees identical inputs, so any two models can be compared
     paired-by-item. Keep `item_id` on every call and never lose it.
  2. Cost is *derived*, never measured. Token counts come from the API's
     usage field; dollars come from a versioned PriceTable applied afterward.
     Store the tokens, snapshot the price table into the run, and you can
     re-price a finished run without paying for a single call again.

Serialize to JSONL for storage.
"""
from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any, Optional, Protocol, runtime_checkable

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------

class TaskType(str, Enum):
    MCQ_QA = "mcq_qa"                      # multiple choice, one correct option
    CLASSIFICATION = "classification"      # pick a label from a fixed set
    EXTRACTION = "extraction"              # pull a span/value out of a passage
    SHORT_ANSWER = "short_answer"          # free-form, but with a checkable answer
    SUMMARIZATION = "summarization"        # free-form, judge-scored


class ScoringMethod(str, Enum):
    EXACT_MATCH = "exact_match"                # string equality after strip
    NORMALIZED_MATCH = "normalized_match"      # lowercase, punctuation/article-stripped
    MCQ_LETTER = "mcq_letter"                  # parse a letter, compare to gold option TEXT
    NUMERIC_TOLERANCE = "numeric_tolerance"    # parse a number, compare within tolerance
    TOKEN_F1 = "token_f1"                      # SQuAD-style token overlap; partial credit
    LLM_JUDGE = "llm_judge"                    # judge model rules on equivalence


class EscalationSignal(str, Enum):
    """Cheap evidence that the small model probably got this one wrong.

    Every signal has a cost, and that cost counts against the cascade's
    savings — SAMPLE_DISAGREEMENT means k calls to the small model, and
    VERIFIER_MODEL means an extra call to a third model on every item.
    """
    SELF_REPORTED_CONFIDENCE = "self_reported_confidence"
    SAMPLE_DISAGREEMENT = "sample_disagreement"
    VERIFIER_MODEL = "verifier_model"
    OUTPUT_HEURISTIC = "output_heuristic"      # hedging phrases, length outliers, parse failure
    ALWAYS = "always"                          # degenerate: escalate everything (= always-large baseline)


class PolicyKind(str, Enum):
    STATIC = "static"        # route on item properties only (task type, length)
    CASCADE = "cascade"      # small model first, escalate on a signal
    BASELINE = "baseline"    # always-small / always-large, for comparison


class SplitName(str, Enum):
    CALIBRATION = "calibration"   # tune thresholds here
    HOLDOUT = "holdout"           # report numbers here. Never the other way around.


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

class TaskItem(BaseModel):
    """One task item. The prompt is fixed across models — that's the whole design."""
    id: str
    task_type: TaskType
    scoring_method: ScoringMethod
    instruction: str = Field(description="Task instruction, identical for every model in the ladder")
    input_text: str = Field(description="The question / passage / text to act on")
    options: Optional[list[str]] = Field(default=None, description="MCQ/classification option texts")
    # Gold answers for MCQ are stored as option TEXT, never letters — letters are a
    # rendering detail and comparing them breaks the moment option order changes.
    gold_answer: str
    scoring_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Method-specific knobs, e.g. {'tolerance': 0.01} for NUMERIC_TOLERANCE",
    )
    split: SplitName = SplitName.HOLDOUT
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskSet(BaseModel):
    """A named collection of items with a calibration/holdout split."""
    id: str
    name: str
    description: str = ""
    items: list[TaskItem]
    source: str = Field(default="", description="HuggingFace dataset id, file path, or 'hand-built'")


# ---------------------------------------------------------------------------
# Models & pricing
# ---------------------------------------------------------------------------

class ModelSpec(BaseModel):
    """One rung on the ladder. The ladder should span sizes and families —
    five 70B models tell you nothing about the size/quality curve."""
    model_id: str = Field(description="NVIDIA Build model ID")
    family: str = Field(description="llama / mixtral / gemma / qwen / deepseek ...")
    params_b: Optional[float] = Field(default=None, description="Total parameters in billions")
    active_params_b: Optional[float] = Field(
        default=None,
        description="Active params per token for MoE models — this, not total, drives serving cost",
    )
    is_moe: bool = False
    context_window: Optional[int] = None
    role: str = Field(default="target", description="'target' | 'judge' | 'verifier' — judges stay out of the target ladder")
    notes: str = ""


class PriceEntry(BaseModel):
    """One model's modeled price. Every field here is load-bearing for honesty:
    a cost claim with no source and no date is not a result."""
    model_id: str
    usd_per_1m_input: float
    usd_per_1m_output: float
    source_url: str = Field(default="", description="Public list price this came from")
    as_of: date
    is_estimate: bool = Field(
        default=False,
        description="True when no public list price exists and this is your best guess. Must surface in the report.",
    )
    notes: str = ""


class PriceTable(BaseModel):
    """Versioned price model. Snapshot this into every ProfileRun — re-pricing an
    old run later must not require re-querying the models."""
    id: str
    as_of: date
    entries: list[PriceEntry]
    notes: str = ""

    def entry_for(self, model_id: str) -> Optional[PriceEntry]:
        return next((e for e in self.entries if e.model_id == model_id), None)


# ---------------------------------------------------------------------------
# Calls: usage, latency, scoring
# ---------------------------------------------------------------------------

class TokenUsage(BaseModel):
    """Straight from the API response's `usage` field. Never estimated, never
    recomputed with a local tokenizer — providers count their own way and that's
    the count you'll be billed on."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class LatencySample(BaseModel):
    """Latency is a property of the environment as much as the model. A sample
    without its concurrency is uninterpretable, so it's required here."""
    ttft_ms: Optional[float] = Field(default=None, description="Time to first token; None if not streaming")
    total_ms: float
    concurrency: int = Field(description="In-flight requests when this one was issued")
    attempt: int = Field(default=1, description="Retries have their own latency — don't silently average them in")
    timestamp: datetime


class ScoreResult(BaseModel):
    """A score in [0, 1]. Binary methods emit 0.0 or 1.0; TOKEN_F1 emits partial credit."""
    value: float = Field(ge=0.0, le=1.0)
    method: ScoringMethod
    parsed_answer: Optional[str] = Field(default=None, description="What the scorer extracted; None if unparseable")
    parse_failed: bool = Field(
        default=False,
        description="Model produced something the scorer couldn't read. Track separately from 'wrong' — "
                    "small models fail to follow output format far more often, and that IS a real cost.",
    )
    judge_model: Optional[str] = Field(default=None, description="LLM_JUDGE only; must not be in the target ladder")
    judge_rationale: Optional[str] = None

class ModelCall(BaseModel):
    """One model's attempt at one item — the atomic record of the whole project.

    `cost_usd` is derived from `usage` + the run's PriceTable. It is stored for
    convenience, but the tokens are the source of truth: change the price table
    and every cost recomputes without a single new API call.
    """
    id: str
    item_id: str = Field(description="TaskItem.id — the pairing key across models")
    model_id: str
    raw_output: str = Field(description="Full completion text, kept for audit and failure examples")
    score: ScoreResult
    usage: TokenUsage
    latency: LatencySample
    cost_usd: Optional[float] = Field(default=None, description="Derived: usage x PriceTable. None until priced.")
    temperature: float = 0.0
    max_tokens: int = 512
    retries: int = 0
    error: Optional[str] = Field(default=None, description="Set when the call failed after retries; score is 0 and the item counts")
    self_reported_confidence: Optional[float] = Field(
        default=None, description="If you asked the model for it — the cheapest escalation signal, and the least reliable"
    )
    timestamp: datetime

# ---------------------------------------------------------------------------
# Profiles & the Pareto frontier
# ---------------------------------------------------------------------------

class ModelProfile(BaseModel):
    """Per (model x task set) rollup — one row of the profile table."""
    model_id: str
    task_set_id: str
    task_type: Optional[TaskType] = Field(default=None, description="None = pooled across types")
    split: SplitName
    n_items: int

    accuracy: float = Field(ge=0.0, le=1.0, description="Mean score. A bare accuracy is not a result — ship the CI.")
    ci_low: float
    ci_high: float
    ci_method: str = Field(default="wilson", description="wilson | bootstrap")

    mean_prompt_tokens: float
    mean_completion_tokens: float = Field(
        description="First-class number: output length is a cost lever the MODEL pulls, not you"
    )

    latency_p50_ms: float
    latency_p95_ms: float = Field(description="Tail latency is what users feel; the mean hides it")
    latency_ttft_p50_ms: Optional[float] = None
    concurrency: int = Field(description="The load these latencies were measured under")

    cost_per_request_usd: float
    cost_per_1k_requests_usd: float
    cost_per_correct_answer_usd: float = Field(
        description="cost_per_request / accuracy — usually the number that settles the argument"
    )

    parse_failure_rate: float = Field(ge=0.0, le=1.0)
    error_rate: float = Field(ge=0.0, le=1.0, description="Calls that failed after retries")
    price_table_id: str


class ParetoPoint(BaseModel):
    """One model's position on a (quality, cost) or (quality, latency) plane."""
    model_id: str
    quality: float
    cost_axis_value: float = Field(description="cost_per_1k_requests_usd or latency_p95_ms — see frontier.axis")
    on_frontier: bool
    dominated_by: list[str] = Field(
        default_factory=list,
        description="Models with >= quality AND <= cost. A dominated model has no defensible use case.",
    )


class ParetoFrontier(BaseModel):
    """Frontier for one task type on one plane."""
    task_set_id: str
    task_type: Optional[TaskType] = None
    axis: str = Field(description="'cost' | 'latency'")
    points: list[ParetoPoint]
    frontier_model_ids: list[str]


# ---------------------------------------------------------------------------
# Quality floors & routing
# ---------------------------------------------------------------------------

class QualityFloor(BaseModel):
    """The 'good enough' rule. `require_ci_lower_bound` is the design decision
    with teeth: a point estimate that clears the floor while its interval doesn't
    is exactly the case where a recommendation quietly becomes a guess."""
    kind: str = Field(description="'absolute' (accuracy >= value) | 'relative' (within value of the best model)")
    value: float
    require_ci_lower_bound: bool = Field(
        default=False, description="If True, ci_low must clear the floor, not just the point estimate"
    )
    rationale: str = ""


class RoutingRule(BaseModel):
    """One branch of a policy: when this condition holds, use this model."""
    condition: str = Field(description="Human-readable predicate, e.g. \"task_type == 'classification'\"")
    model_id: str
    note: str = ""


class RoutingDecision(BaseModel):
    """One item's outcome under a simulated policy — which model got picked
    and what it cost. Not embedded in PolicySimulation (that's the aggregate);
    this is the per-item audit trail behind it, useful to persist alongside
    PolicySimulation for anyone who wants to see *which* items escalated."""
    item_id: str
    selected_model: str
    score: float = Field(ge=0.0, le=1.0)
    latency_ms: float
    cost_usd: float
    confidence: Optional[float] = None
    escalated: bool = Field(
        default=False,
        description="True if a CASCADE policy routed this item to escalation_model_id instead of primary_model_id",
    )


class RoutingPolicy(BaseModel):
    """A deployable recommendation. The two baselines are policies too — always
    encode them explicitly so the comparison is apples to apples."""
    id: str
    name: str
    kind: PolicyKind
    rules: list[RoutingRule] = Field(default_factory=list, description="STATIC policies")

    # CASCADE policies
    primary_model_id: Optional[str] = None
    escalation_model_id: Optional[str] = None
    escalation_signal: Optional[EscalationSignal] = None
    escalation_threshold: Optional[float] = Field(
        default=None, description="Tuned on CALIBRATION only. Tuning on holdout is how cascade savings become fiction."
    )
    escalation_signal_cost_per_request_usd: float = Field(
        default=0.0, description="k-sampling and verifier calls aren't free — this comes out of the savings"
    )

    fallback_model_id: Optional[str] = Field(default=None, description="Used when the primary call errors out")
    quality_floor: Optional[QualityFloor] = None


class PolicySimulation(BaseModel):
    """What a policy actually delivers, replayed over stored ModelCalls on a split.

    Because every model ran every item, simulation is a replay over cached calls —
    no new API spend, and any policy can be evaluated after the fact.
    """
    policy_id: str
    task_set_id: str
    split: SplitName = Field(
        default=SplitName.HOLDOUT, description="Report on HOLDOUT. Calibration numbers are for tuning and for showing the gap."
    )
    n_items: int

    blended_accuracy: float
    blended_ci_low: float
    blended_ci_high: float
    blended_cost_per_1k_requests_usd: float
    blended_latency_p95_ms: Optional[float] = Field(
        default=None, description="A cascade's escalated items pay both models' latency — don't forget the tail"
    )
    escalation_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    # Both baselines, always. A policy that beats neither is not a result.
    baseline_small_accuracy: float
    baseline_small_cost_per_1k_usd: float
    baseline_large_accuracy: float
    baseline_large_cost_per_1k_usd: float

    cost_savings_vs_large: float = Field(description="1 - (blended_cost / baseline_large_cost)")
    quality_retained_vs_large: float = Field(description="blended_accuracy / baseline_large_accuracy")
    notes: str = ""


# ---------------------------------------------------------------------------
# Run metadata
# ---------------------------------------------------------------------------

class ProfileRun(BaseModel):
    """One end-to-end profiling run (audit + reproducibility)."""
    run_id: str
    started_at: datetime
    ended_at: Optional[datetime] = None
    task_set_id: str
    models: list[ModelSpec]
    judge_model: Optional[str] = None

    temperature: float = Field(default=0.0, description="Identical across models — that's the point of the comparison")
    max_tokens: int = Field(default=512, description="Identical cap across models, or the cost comparison is rigged")
    concurrency: int
    interleaved: bool = Field(
        default=True,
        description="Round-robin across models. Running A for an hour then B measures time of day, not the models.",
    )
    n_repeats: int = Field(default=1, description="Repeated calls per (item, model) for latency stability")
    rng_seed: int

    price_table: PriceTable = Field(description="Snapshotted, not referenced — prices change and old runs must stay reproducible")
    n_items: int
    n_calls: int
    config_hash: str
    notes: str = ""


# ---------------------------------------------------------------------------
# Scorer protocol (pluggable; concrete scorers live in your package)
# ---------------------------------------------------------------------------

@runtime_checkable
class Scorer(Protocol):
    """
    Pluggable scoring contract. One implementation per method, e.g.:

      - ExactMatchScorer        — strip and compare
      - NormalizedMatchScorer   — lowercase, strip punctuation/articles, compare
      - MCQLetterScorer         — parse a letter, map through options, compare to gold TEXT
      - NumericToleranceScorer  — parse a number, compare within tolerance
      - TokenF1Scorer           — SQuAD-style overlap, partial credit
      - LLMJudgeScorer          — asks a judge model; the judge stays out of the target ladder

    Implementations must be deterministic given (item, raw_output) — except the
    LLM judge, which must run at temperature 0 and record its model id in the
    ScoreResult so a scored run can be traced back to who scored it.

    A scorer that can't parse the output returns value=0.0 with parse_failed=True.
    That distinction matters: format-following failure is a different weakness
    from being wrong, and small models fail that way disproportionately.
    """

    name: str
    method: ScoringMethod

    def score(self, item: TaskItem, raw_output: str) -> ScoreResult:
        """Score one model output against one item's gold answer."""
        ...