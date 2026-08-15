import json
from pathlib import Path
import argparse
import asyncio

from Runner import main as run_pipeline
from Report import report


def _price_table_id_from_price_file(price_table_path: str) -> str:
    """The actual price table is now its own file (-p/--price-table), not
    nested inside the task-set file — read its id directly from there
    rather than relying on Report.py's task-set-nested fallback, which
    predates the toy-set/model-set/price-table split and would silently
    label every profile price_table_id="unknown" otherwise."""
    try:
        data = json.loads(Path(price_table_path).read_text(encoding="utf-8"))
        return data["price_table"]["id"]
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run models against a task set, then report accuracy/cost/latency and a recommendation.")

    parser.add_argument("-r", "--results", default="results.jsonl", help="Path to write/read the ModelCall JSONL file (default: results.jsonl)")
    parser.add_argument("-m", "--model-set", required=True, help="Path to a ModelSet JSON")
    parser.add_argument("-p", "--price-table", required=True, help="Path to a PriceTable JSON")
    parser.add_argument("-t", "--task-set", required=True, help="Path to a TaskSet JSON")
    parser.add_argument("--price-table-id", default=None, help="Override the price_table_id recorded on each profile (default: read from --price-table)")
    parser.add_argument("-s", "--split", choices=["calibration", "holdout"], default=None, help="Filter the headline table to one split (Pareto/floor/comparison are always holdout)")
    parser.add_argument("--skip-run", action="store_true", help="Skip calling the models and just report on an existing --results file")

    args = parser.parse_args()

    price_table_id = args.price_table_id or _price_table_id_from_price_file(args.price_table)

    if not args.skip_run:
        asyncio.run(run_pipeline(args.results, args.task_set, args.model_set, args.price_table))
        print("\n" + "=" * 70 + "\n")

    report(args.task_set, price_table_id, args.results, args.split)


if __name__ == "__main__":
    main()