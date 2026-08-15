#!/usr/bin/env python3
"""
Week 4-5 report.

Called by modelpicker.py as report(task_set, price_table_id, results, split),
or run standalone:

    python Report.py --results results.jsonl --task-set toy-set.json

Prints, in order:
  1. The Week 3 table (accuracy w/ 95% Wilson CI, tokens, p50 latency, cost),
     pooled across task types, both splits labeled.
  2. Pareto frontier per task type (quality x cost, quality x latency) on
     the holdout split — SPEC.md checklist: "A model can win on
     classification and lose badly on extraction."
  3. Quality-floor pass/fail under both rules from DESIGN.md Question 2
     (point estimate vs. CI lower bound) — shown side by side specifically
     because they can disagree.
  4. A paired-bootstrap comparison (Miller 2024) between the best model and
     the cheapest model that clears the floor, if they differ — this is the
     comparison the headline recommendation actually hinges on, so it's the
     one that gets the statistical treatment, not just a bar chart.
"""
import argparse
import json
from pathlib import Path

from Scorer import (
    load_results,
    load_task_set,
    build_model_profiles,
    compute_all_pareto_frontiers,
    evaluate_quality_floor,
    paired_bootstrap,
    maximum_performance_improvement,
)
from DataStructures import ModelProfile, ModelCall, TaskSet, SplitName, QualityFloor


# Project default floor, defended in DESIGN.md Question 2: relative to the
# best model on holdout, and — the stricter, headline rule — a model must
# clear it even at its worst plausible accuracy (ci_low), not just its point
# estimate. See DESIGN.md for why the CI-lower-bound rule is the one this
# project defends to someone spending real money on the recommendation.
DEFAULT_FLOOR = QualityFloor(kind="relative", value=0.03, require_ci_lower_bound=True)


def _price_table_id_from(task_set_path: Path) -> str:
    """Best-effort: pull price_table.id out of a sample-data.json-style file
    if one's alongside the TaskSet, so --price-table-id doesn't have to be
    typed by hand every run. Falls back to "unknown" for a bare TaskSet JSON
    with no price_table section."""
    try:
        data = json.loads(task_set_path.read_text(encoding="utf-8"))
        return data.get("price_table", {}).get("id", "unknown")
    except Exception:
        return "unknown"


def print_profile_table(profiles: list[ModelProfile]) -> None:
    """One row per model, pooled across task types (task_type=None) — the
    Week 3 table. Call build_model_profiles(..., include_per_task_type=False)
    upstream so `profiles` only contains pooled rows; this just renders them."""
    if not profiles:
        raise ValueError("No profiles to print — check that --results and --task-set actually overlap.")

    rows_sorted = sorted(profiles, key=lambda p: (p.split.value, -p.accuracy))

    headers = ["model", "split", "n", "accuracy", "95% CI", "tok in", "tok out", "p50 ms", "$/1k req"]
    rows = [
        [
            p.model_id,
            p.split.value,
            str(p.n_items),
            f"{p.accuracy:.3f}",
            f"[{p.ci_low:.3f}, {p.ci_high:.3f}]",
            f"{p.mean_prompt_tokens:.0f}",
            f"{p.mean_completion_tokens:.0f}",
            f"{p.latency_p50_ms:.0f}",
            f"{p.cost_per_1k_requests_usd:.4f}",
        ]
        for p in rows_sorted
    ]

    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]

    def fmt_row(cells: list[str]) -> str:
        return "  ".join(c.ljust(w) for c, w in zip(cells, widths))

    print("\033[34m"+fmt_row(headers)+"\033[0m")
    print("\033[34m"+"  ".join("-" * w for w in widths)+"\033[0m")
    for r in rows:
        print(fmt_row(r))


def print_pareto_frontiers(per_type_profiles: list[ModelProfile], split: SplitName) -> None:
    """Pareto frontier per task type, cost and latency axes, restricted to
    one split (holdout by default — never calibration; see DESIGN.md
    Question 4). SPEC.md checklist item: quality x cost and quality x
    latency, per task type, not just pooled."""
    split_profiles = [p for p in per_type_profiles if p.split == split]
    frontiers = compute_all_pareto_frontiers(split_profiles)

    if not frontiers:
        print(f"\033[31m(not enough models-per-task-type on split={split.value!r} to compute a frontier)\033[0m")
        return

    by_task_type: dict = {}
    for f in frontiers:
        by_task_type.setdefault(f.task_type, []).append(f)

    for task_type, fs in by_task_type.items():
        label = task_type.value if task_type is not None else "pooled"
        print(f"\n\033[34m  {label}:\033[0m")
        for f in fs:
            frontier_str = ", ".join(f.frontier_model_ids) if f.frontier_model_ids else "(none)"
            print(f"    {f.axis:8s} frontier: {frontier_str}")


def print_quality_floor(pooled_holdout_profiles: list[ModelProfile], floor: QualityFloor) -> dict[str, bool]:
    """Print pass/fail under both the point-estimate rule and the
    CI-lower-bound rule side by side — DESIGN.md Question 2 is explicit that
    these can disagree, and the report should show that disagreement rather
    than silently pick one."""
    point_rule = floor.model_copy(update={"require_ci_lower_bound": False})
    ci_rule = floor.model_copy(update={"require_ci_lower_bound": True})

    point_result = evaluate_quality_floor(pooled_holdout_profiles, point_rule)
    ci_result = evaluate_quality_floor(pooled_holdout_profiles, ci_rule)

    kind_label = f"{floor.kind} floor (value={floor.value})"
    print(f"\n\033[34mQuality floor — {kind_label}, evaluated on holdout:\033[0m")
    print(f"  {'model':40s} {'point est.':>10s} {'ci_low':>10s}")
    for model_id in sorted(point_result, key=lambda m: -point_result[m]):
        p = "PASS" if point_result[model_id] else "fail"
        c = "PASS" if ci_result[model_id] else "fail"
        marker = "  <-- disagreement" if point_result[model_id] != ci_result[model_id] else ""
        print(f"  {model_id:40s} {p:>10s} {c:>10s}{marker}")

    print(f"  Headline rule used below: {'ci_low' if floor.require_ci_lower_bound else 'point estimate'}.")
    return ci_result if floor.require_ci_lower_bound else point_result


def _calls_by_model_for_split(
    calls: list[ModelCall],
    task_set: TaskSet,
    split: SplitName,
) -> dict[str, list[ModelCall]]:
    item_ids_in_split = {item.id for item in task_set.items if item.split == split}
    by_model: dict[str, list[ModelCall]] = {}
    for c in calls:
        if c.item_id in item_ids_in_split:
            by_model.setdefault(c.model_id, []).append(c)
    return by_model


def print_headline_comparison(
    pooled_holdout_profiles: list[ModelProfile],
    floor_pass: dict[str, bool],
    calls_by_model: dict[str, list[ModelCall]],
) -> None:
    """The paired comparison the recommendation actually hinges on: best
    accuracy on holdout vs. cheapest model that clears the quality floor.
    If they're the same model, there's nothing to compare. If they differ,
    a paired bootstrap (not a bare point-estimate gap) is what tells you
    whether the cheaper pick is really giving up quality or whether that
    gap is noise on a 30-item toy set — see the paper notes on Minimum
    Detectable Effect for why that question matters more than the point
    estimate here."""
    if not pooled_holdout_profiles:
        return

    best = max(pooled_holdout_profiles, key=lambda p: p.accuracy)
    passing = [p for p in pooled_holdout_profiles if floor_pass.get(p.model_id, False)]

    print(f"\n\033[34mBest accuracy on holdout: {best.model_id} ({best.accuracy:.3f})\033[0m")

    if not passing:
        print("\033[31mNo model clears the quality floor on holdout under the headline rule — "
              "recommendation defaults to the best-accuracy model, at whatever it costs.\033[0m")
        return

    cheapest_passing = min(passing, key=lambda p: p.cost_per_1k_requests_usd)
    print(f"\033[34mCheapest model clearing the floor: {cheapest_passing.model_id} \033[0m"
          f"(accuracy={cheapest_passing.accuracy:.3f}, ${cheapest_passing.cost_per_1k_requests_usd:.4f}/1k req)")

    if cheapest_passing.model_id == best.model_id:
        print("Same model — no cost/quality tradeoff to weigh here.")
        return

    calls_best = calls_by_model.get(best.model_id, [])
    calls_cheap = calls_by_model.get(cheapest_passing.model_id, [])

    if not calls_best or not calls_cheap:
        print("(missing calls for one of these models on holdout — can't run the paired comparison)")
        return

    diff, ci_low, ci_high = paired_bootstrap(calls_best, calls_cheap)
    verdict = "genuinely different" if (ci_low > 0 or ci_high < 0) else "not distinguishable from noise on this holdout set"
    print(
        f"\nPaired bootstrap ({best.model_id} - {cheapest_passing.model_id}), holdout, "
        f"95% CI: diff={diff:+.3f}  ci=({ci_low:+.3f}, {ci_high:+.3f})  -> {verdict}"
    )

    mpi = maximum_performance_improvement(calls_best, calls_cheap)
    print(
        f"MPI: {best.model_id} right / {cheapest_passing.model_id} wrong on "
        f"{mpi['mpi_a_over_b']:.1%} of items; reverse on {mpi['mpi_b_over_a']:.1%} "
        f"(n={mpi['n_items']}) — the reverse number is what a cascade to "
        f"{cheapest_passing.model_id} would be trying to keep, not just discard."
    )

    savings = 1 - (cheapest_passing.cost_per_1k_requests_usd / best.cost_per_1k_requests_usd) \
        if best.cost_per_1k_requests_usd > 0 else 0.0
    gap_note = (
        "the paired comparison confirms the accuracy gap is real, not noise"
        if verdict == "genuinely different"
        else "the accuracy gap versus the best model is not statistically distinguishable "
             "on this holdout set — the cheaper model is the defensible pick"
    )
    print(
        f"\nRecommendation: {cheapest_passing.model_id} clears the quality floor at "
        f"{savings:.0%} lower cost than {best.model_id} — {gap_note}."
    )


def report(task_set, price_table_id, results, split) -> None:
    task_set_path = Path(task_set)
    price_table_id = price_table_id or _price_table_id_from(task_set_path)

    calls = load_results(results)
    loaded_task_set = load_task_set(task_set_path)

    pooled_profiles = build_model_profiles(
        calls, loaded_task_set, price_table_id=price_table_id,
        include_pooled=True, include_per_task_type=False,
    )
    per_type_profiles = build_model_profiles(
        calls, loaded_task_set, price_table_id=price_table_id,
        include_pooled=False, include_per_task_type=True,
    )

    display_profiles = pooled_profiles
    if split is not None:
        display_profiles = [p for p in display_profiles if p.split.value == split]

    print_profile_table(display_profiles)

    # Everything below is deliberately holdout-only, regardless of --split:
    # SPEC.md Design Question 4 — thresholds/decisions get tuned on
    # calibration, reported on holdout, never the reverse.
    print_pareto_frontiers(per_type_profiles, split=SplitName.HOLDOUT)

    pooled_holdout = [p for p in pooled_profiles if p.split == SplitName.HOLDOUT]
    floor_pass = print_quality_floor(pooled_holdout, DEFAULT_FLOOR)

    calls_by_model = _calls_by_model_for_split(calls, loaded_task_set, SplitName.HOLDOUT)
    print_headline_comparison(pooled_holdout, floor_pass, calls_by_model)


def main() -> None:
    parser = argparse.ArgumentParser(description="Print the Week 4-5 model comparison report.")
    parser.add_argument("--results", default="results.jsonl", help="Path to a ModelCall JSONL file (default: results.jsonl)")
    parser.add_argument("--task-set", required=True, help="Path to a TaskSet JSON, or a sample-data.json-style file with a 'task_set' key")
    parser.add_argument("--price-table-id", default=None, help="Override the price_table_id recorded on each profile")
    parser.add_argument("--split", choices=["calibration", "holdout"], default=None, help="Filter the headline table to one split (Pareto/floor/comparison are always holdout)")
    args = parser.parse_args()
    report(args.task_set, args.price_table_id, args.results, args.split)


if __name__ == "__main__":
    main()