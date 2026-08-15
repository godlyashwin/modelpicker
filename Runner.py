import json
import os
import time
import asyncio
import hashlib
import uuid
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv
from openai import AsyncOpenAI
import math
import random
from collections import Counter
from DataStructures import (
    TaskItem,
    ModelSpec,
    PriceTable,
    PriceEntry,
    TokenUsage,
    LatencySample,
    ScoreResult,
    ModelCall,
    ScoringMethod,
    ProfileRun,
)
from Scorer import score_output


# --------------------------------------------------
# Setup
# --------------------------------------------------

load_dotenv()

dead_models = set()

def model_fail(
    task: TaskItem,
    model: ModelSpec,
    error: str,
    timestamp: datetime,
    total_ms: float = 0.0,
    max_tokens: int = 256,
    concurrency: int = 4,
    raw_output: str = "",
    cost_usd: float = 0.0,
    self_reported_confidence: float | None = None,
    retries: int = 0,
) -> ModelCall:

    return ModelCall(
        id=f"{task.id}::{model.model_id}",
        item_id=task.id,
        model_id=model.model_id,
        raw_output=raw_output,
        score=ScoreResult(
            value=0.0,
            method=task.scoring_method,
            parsed_answer=None,
            parse_failed=True,
        ),
        usage=TokenUsage(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
        ),
        latency=LatencySample(
            ttft_ms=0.0,
            total_ms=total_ms,
            concurrency=concurrency,
            attempt=retries,
            timestamp=timestamp,
        ),
        cost_usd=cost_usd,
        temperature=0.0,
        max_tokens=max_tokens,
        retries=retries,
        error=error,
        self_reported_confidence=self_reported_confidence,
        timestamp=timestamp,
    )

def setup(task_file: str, model_file: str, price_file: str):
    with open(task_file, "r", encoding="utf-8") as f:
        task_data = json.load(f)

    with open(model_file, "r", encoding="utf-8") as f:
            model_data = json.load(f)

    with open(price_file, "r", encoding="utf-8") as f:
            price_data = json.load(f)

    task_set_id = task_data.get("id", "unknown")
    tasks = [TaskItem(**t) for t in task_data["items"]]

    models = [
        ModelSpec(**m)
        for m in model_data["model_ladder"]
        if m["role"] == "target"   # skip judge model
    ]
    judge_specs = [m for m in model_data["model_ladder"] if m["role"] != "target"]
    judge_model_id = judge_specs[0]["model_id"] if judge_specs else None

    price_table = PriceTable(
        id=price_data["price_table"]["id"],
        as_of=price_data["price_table"]["as_of"],
        notes=price_data["price_table"]["notes"],
        entries=[PriceEntry(**e) for e in price_data["price_table"]["entries"]],
    )

    # Fail before spending any API budget, not partway through a run: every
    # target model needs a price entry, or the cost axis silently goes wrong
    # for whichever model is missing one.
    missing_prices = [m.model_id for m in models if price_table.entry_for(m.model_id) is None]
    if missing_prices:
        raise ValueError(
            f"price_table {price_table.id!r} is missing entries for: {missing_prices}. "
            "Add a PriceEntry for each (mark is_estimate=True if uncertain) before running."
        )

    return tasks, models, price_table, task_set_id, judge_model_id

def build_prompt(task: TaskItem) -> str:
    prompt = task.instruction.strip() + "\n\n"
    prompt += task.input_text.strip()
    if task.options:
        prompt += "\n\nOptions:\n"
        for i, option in enumerate(task.options):
            letter = chr(ord("A") + i)
            prompt += f"{letter}. {option}\n"

    return prompt

def compute_cost(model_id: str, usage: TokenUsage, price_table) -> float:
    entry = price_table.entry_for(model_id)
    if entry is None:
        # SPEC.md's own warning: "cost is modeled, not measured" — a model
        # missing from the price table silently priced at $0.00 is the worst
        # version of that trap, because nothing downstream flags it as an
        # estimate; it just looks free. Fail loudly instead so a mismatched
        # model-set.json / price-table.json (like gemma-4-31b-it vs
        # gemma-2-2b-it) gets caught before a run, not discovered in a report.
        raise ValueError(
            f"No price_table entry for {model_id!r} (price_table id={price_table.id!r}). "
            "Every target model needs a PriceEntry — add one (mark is_estimate=True if you "
            "don't have a sourced list price) rather than letting cost default to $0."
        )

    input_cost = (
        usage.prompt_tokens / 1_000_000
    ) * entry.usd_per_1m_input

    output_cost = (
        usage.completion_tokens / 1_000_000
    ) * entry.usd_per_1m_output

    return input_cost + output_cost

def get_confidence(logprob_content):
    probabilities = []
    for token_data in logprob_content:
        logprob = token_data.logprob
        prob = math.exp(logprob)
        probabilities.append(prob)

    if not probabilities:
        return 0.0
        
    total_logprob = sum(math.log(p) for p in probabilities)
    avg_logprob = total_logprob / len(probabilities)
    overall_confidence = math.exp(avg_logprob) * 100
    
    return overall_confidence


async def _stream_completion(client, model_id: str, prompt: str, max_tokens: int, start: float):
    """
    Issue a streaming chat completion and consume it, returning
    (ttft_ms, raw_text, logprob_events, usage).

    `start` is the perf_counter() timestamp the caller took immediately
    before awaiting this, so TTFT is measured from when the request was
    actually issued, not from inside this function.

    This is what makes the interleaved runner's TTFT/total_ms distinction
    real instead of always-None (SPEC.md's checklist: "Separate
    time-to-first-token from total time; for streaming UIs they are not the
    same product decision.") — the previous non-streaming version had no way
    to observe TTFT at all.

    CAVEAT: `stream_options={"include_usage": True}` is the standard
    OpenAI-compatible convention for getting a final usage-only chunk at the
    end of a stream, and per-chunk `logprobs` is the standard shape for
    per-token confidence under streaming. Neither has been verified live
    against NVIDIA Build / OpenRouter in this session (no network access to
    those hosts from here) — smoke-test on one item before trusting a full
    run. If `usage` comes back None, that provider isn't honoring
    stream_options and usage will need a different approach.
    """
    ttft_ms = None
    raw_parts: list[str] = []
    logprob_events = []
    usage = None

    stream = await client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=max_tokens,
        logprobs=True,
        stream=True,
        stream_options={"include_usage": True},
    )

    async for chunk in stream:
        if chunk.choices:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                if ttft_ms is None:
                    ttft_ms = (time.perf_counter() - start) * 1000
                raw_parts.append(delta.content)

            choice_logprobs = getattr(chunk.choices[0], "logprobs", None)
            if choice_logprobs and choice_logprobs.content:
                logprob_events.extend(choice_logprobs.content)

        chunk_usage = getattr(chunk, "usage", None)
        if chunk_usage is not None:
            usage = chunk_usage

    return ttft_ms, "".join(raw_parts), logprob_events, usage


async def run_model(client, task: TaskItem, model: ModelSpec, price_table):
    if model.model_id in dead_models:
        return model_fail(
            task=task,
            model=model,
            error=f"Model is unavailable.",
            retries=0,
            timestamp=datetime.now(timezone.utc),
            total_ms=0.0,
            max_tokens=0,
            raw_output=""
        )

    prompt = build_prompt(task)

    MAX_RETRIES = 3

    # Initialize variables so they always exist
    stream_completed = False
    raw = "No response."
    confidence = None
    ttft_ms = None
    stream_usage = None
    total_ms = 0.0
    retries = 0
    error = None

    max_tokens = (
        64
        if task.scoring_method in {
            ScoringMethod.EXACT_MATCH,
            ScoringMethod.NORMALIZED_MATCH,
            ScoringMethod.NUMERIC_TOLERANCE,
            ScoringMethod.MCQ_LETTER,
        }
        else 256
    )

    for retry in range(MAX_RETRIES):
        try:
            start = time.perf_counter()
            ttft_ms, raw, logprob_events, stream_usage = await asyncio.wait_for(
                _stream_completion(client, model.model_id, prompt, max_tokens, start),
                timeout=120,
            )
            total_ms = (time.perf_counter() - start) * 1000

            if not raw:
                return model_fail(
                    task=task,
                    model=model,
                    error="Invalid Model Output: empty stream",
                    retries=retry + 1,
                    timestamp=datetime.now(timezone.utc),
                    total_ms=total_ms,
                    max_tokens=max_tokens,
                    raw_output=""
                )

            confidence = get_confidence(logprob_events) if logprob_events else None
            stream_completed = True
            retries = retry
            error = None
            break
        except asyncio.TimeoutError:
            return model_fail(
                task=task,
                model=model,
                error="Model took too long to respond",
                retries=retry + 1,
                timestamp=datetime.now(timezone.utc),
                total_ms=total_ms,
                max_tokens=max_tokens,
                raw_output=""
            )
        except Exception as e:
            error = str(e)
            if "404" in error or "not found" in error.lower():
                dead_models.add(model.model_id)
                return model_fail(
                    task=task,
                    model=model,
                    error="Model does not exist or work anymore",
                    retries=retry + 1,
                    timestamp=datetime.now(timezone.utc),
                    total_ms=total_ms,
                    max_tokens=max_tokens,
                    raw_output=""
                )
            elif "504" in error:
                return model_fail(
                    task=task,
                    model=model,
                    error="Model did not respond back in time",
                    retries=retry + 1,
                    timestamp=datetime.now(timezone.utc),
                    total_ms=total_ms,
                    max_tokens=max_tokens,
                    raw_output=""
                )

            
            retries = retry + 1 

            if retry == MAX_RETRIES - 1:
                return model_fail(
                    task=task,
                    model=model,
                    error="Model unable to work or doesn't exist",
                    retries=retry + 1,
                    timestamp=datetime.now(timezone.utc),
                    total_ms=total_ms,
                    max_tokens=max_tokens,
                    raw_output=""
                )

            await asyncio.sleep((2 ** retry) + random.random())

    # Every failure branch above returns early; reaching here means the
    # stream completed. The explicit check is defensive, not dead code by
    # accident — if that ever stops being true, fail loudly instead of
    # silently treating a real response as empty.
    if not stream_completed:
        usage = TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
    elif stream_usage is not None:
        usage = TokenUsage(
            prompt_tokens=stream_usage.prompt_tokens,
            completion_tokens=stream_usage.completion_tokens,
            total_tokens=stream_usage.total_tokens,
        )
    else:
        # The provider didn't send a usage chunk despite stream_options —
        # flag it loudly rather than silently pricing this call at $0.
        print(
            f"⚠ {task.id} | {model.model_id} | stream completed but no usage "
            "chunk was returned — cost/tokens for this call will read as 0. "
            "Check whether this provider supports stream_options include_usage."
        )
        usage = TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)

    latency = LatencySample(
        ttft_ms=ttft_ms,
        total_ms=total_ms,
        concurrency=4,
        attempt=retries + 1,
        timestamp=datetime.now(timezone.utc),
    )

    if not stream_completed:
        score = ScoreResult(
            value=0.0,
            method=task.scoring_method,
            parsed_answer=None,
            parse_failed=True,
        )
    else:
        score = score_output(task, raw)

    cost = compute_cost(model.model_id, usage, price_table)

    return ModelCall(
        id=f"{task.id}::{model.model_id}",
        item_id=task.id,
        model_id=model.model_id,
        raw_output=str(raw),
        score=score,
        usage=usage,
        latency=latency,
        cost_usd=cost,
        temperature=0.0,
        max_tokens=max_tokens,
        retries=retries,
        error=error,
        self_reported_confidence=confidence,
        timestamp=datetime.now(timezone.utc),
    )


RNG_SEED = 42


async def main(results_file: str, task_file: str, model_file: str, price_table: str):
    random.seed(RNG_SEED)

    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("API_KEY"),
    )
    tasks, models, price_table, task_set_id, judge_model_id = setup(
        task_file=task_file, model_file=model_file, price_file=price_table
    )
    started_at = datetime.now(timezone.utc)
    print("Verifying models' existence...")
    openrouter_models = await client.models.list()
    # BUGFIX: `model in openrouter_models` compared a ModelSpec (this
    # project's pydantic type) against the SDK's own `Model` objects, which
    # are never equal regardless of id — every model was getting marked dead
    # before a single call was made. Compare on the actual id string instead.
    available_ids = {m.id for m in openrouter_models.data}
    for model in models:
        if model.model_id not in available_ids:
            print(f"    {model.model_id} does not exist")
            dead_models.add(model.model_id)
    CONCURRENCY = 4
    semaphore = asyncio.Semaphore(CONCURRENCY) # max 4 tasks running parallel
    results = []
    async def guarded(task, model):
        async with semaphore:
            try:
                call = await run_model(client, task, model, price_table)
                if call.error:
                    print(
                        f"\033[31m✗ {task.id} | {model.model_id} | "
                        f"{call.error}\033[0m"
                    )
                else:
                    print(
                        f"\033[32m✓ {task.id} | {model.model_id} | "
                        f"score={call.score.value} | "
                        f"cost=${call.cost_usd:.6f}\033[0m"
                    )

                return call

            except Exception as e:
                print(
                    f"\033[33m? {task.id} | {model.model_id} | {e}\033[0m"
                )
                
    jobs = [
        guarded(task, model)
        for task in tasks
        for model in models
    ]

    calls = await asyncio.gather(*jobs)

    results = [c for c in calls if c is not None]

    with open(results_file, "w", encoding="utf-8") as f:
        for call in results:
            f.write(call.model_dump_json() + "\n")

    print(f"\n\033[34mSaved {len(results)} model calls to {results_file}\033[0m")

    ended_at = datetime.now(timezone.utc)

    # ProfileRun: run-level audit metadata (DataStructures.py defines this
    # for exactly this purpose — reproducibility — but nothing was
    # constructing one). config_hash lets two runs be compared for "did
    # anything about the setup actually change" without diffing every field
    # by hand.
    config_payload = json.dumps(
        {
            "task_set_id": task_set_id,
            "price_table_id": price_table.id,
            "model_ids": sorted(m.model_id for m in models),
            "concurrency": CONCURRENCY,
            "temperature": 0.0,
            "rng_seed": RNG_SEED,
        },
        sort_keys=True,
    )
    config_hash = hashlib.sha256(config_payload.encode("utf-8")).hexdigest()[:16]

    profile_run = ProfileRun(
        run_id=str(uuid.uuid4()),
        started_at=started_at,
        ended_at=ended_at,
        task_set_id=task_set_id,
        models=models,
        judge_model=judge_model_id,
        temperature=0.0,
        max_tokens=256,
        concurrency=CONCURRENCY,
        interleaved=True,
        n_repeats=1,
        rng_seed=RNG_SEED,
        price_table=price_table,
        n_items=len(tasks),
        n_calls=len(results),
        config_hash=config_hash,
        notes=(
            "max_tokens is 64 for exact_match/normalized_match/mcq_letter/numeric_tolerance "
            "and 256 for open-ended scoring methods (token_f1) — identical across models for "
            "a given task, so the cost comparison isn't rigged by model. "
            f"{len(dead_models)} model(s) unavailable at run start: {sorted(dead_models)}."
        ),
    )

    run_path = Path(results_file).with_name(Path(results_file).stem + ".run.json")
    run_path.write_text(profile_run.model_dump_json(indent=2), encoding="utf-8")
    print(f"\033[34mSaved run metadata to {run_path}\033[0m")

    # Quick summary
    by_model = {}
    for c in results:
        by_model.setdefault(c.model_id, []).append(c)

    #print("\n=== SUMMARY ===")
    #for model_id, calls in by_model.items():
    #    acc = sum(x.score.value for x in calls) / len(calls)
    #    cost = sum(x.cost_usd for x in calls) / len(calls)
    #    lat = sum(x.latency.total_ms for x in calls) / len(calls)
    #
    #    print(
    #        f"{model_id}\n"
    #        f"  accuracy={acc:.3f} ({100*(acc):.3f}%)\n"
    #        f"  avg_cost=${cost:.6f}\n"
    #        f"  avg_latency={lat:.1f} ms ({0.001*(lat):.1f} sec)\n"
    #    )

if __name__ == "__main__":
    asyncio.run(main())

# KEY:
# cls: classification
# ext: extraction
# mcq: multiple choice question
# sa: short answer
# sum: summarization