"""Run the three-layer evaluation for the Insurellm RAG pipeline.

Usage:
    python scripts/evaluate.py              # all three layers (Ragas makes API calls)
    python scripts/evaluate.py --offline    # retrieval + guardrails only, no API calls
    python scripts/evaluate.py --no-ragas   # skip only the Ragas layer

The retrieval and guardrail layers run offline against the local Chroma store;
the Ragas layer calls the judge model. A Ragas failure never discards the
offline results.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag import RagPipeline  # noqa: E402
from rag.evaluation import run_full_evaluation  # noqa: E402

REPORT_PATH = Path(__file__).resolve().parent.parent / "ragas_report.csv"


def _print_retrieval(retrieval: dict) -> None:
    print("\n=== Layer 1: Retrieval (Hit@k / MRR) ===")
    for stage, metrics in retrieval.items():
        label = stage.replace("_", " ")
        pairs = "  ".join(f"{name}={value:.3f}" for name, value in metrics.items())
        print(f"  {label:<14} {pairs}")


def _print_guardrails(guardrails: dict) -> None:
    print("\n=== Layer 3: Guardrails (policy pass/fail) ===")
    print(
        f"  pass rate: {guardrails['pass_rate']:.0%} "
        f"({guardrails['passed']}/{guardrails['total']})"
    )
    for detail in guardrails["details"]:
        mark = "PASS" if detail["ok"] else "FAIL"
        print(
            f"  [{mark}] expected={detail['expected']:<9} "
            f"actual={detail['actual']:<9} {detail['query']}"
        )


def _print_and_save_ragas(ragas_result) -> None:
    print("\n=== Layer 2: Generation (Ragas) ===")
    if ragas_result is None:
        print("  (skipped or failed — see log above)")
        return

    df = ragas_result.to_pandas()

    metric_columns = [
        column
        for column in df.columns
        if df[column].dtype.kind == "f" and column not in {"user_input", "reference"}
    ]
    if metric_columns:
        print("  Mean scores:")
        for column in metric_columns:
            print(f"    {column:<40} {df[column].mean():.3f}")

    df.to_csv(REPORT_PATH, index=False)
    print(f"\n  Saved per-query Ragas report to {REPORT_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run only offline layers (no query expansion, no Ragas API calls).",
    )
    parser.add_argument(
        "--no-ragas",
        action="store_true",
        help="Skip the Ragas generation layer only.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    # The graph store is not needed to evaluate the text-retrieval path.
    pipeline = RagPipeline.build(enable_graph=False)

    results = run_full_evaluation(
        pipeline,
        use_query_expansion=not args.offline,
        run_ragas=not (args.offline or args.no_ragas),
    )

    _print_retrieval(results["retrieval"])
    _print_and_save_ragas(results["ragas"])
    _print_guardrails(results["guardrails"])


if __name__ == "__main__":
    main()
