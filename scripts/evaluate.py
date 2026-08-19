"""Run retrieval evaluation (Hit@k / MRR) over the labelled query set.

    python scripts/evaluate.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Make the project root importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag import RagPipeline  # noqa: E402
from rag.evaluation import evaluate_retrieval  # noqa: E402


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # Evaluation only exercises the text retrieval path, so skip the graph.
    pipeline = RagPipeline.build(enable_graph=False)
    metrics = evaluate_retrieval(pipeline)

    print("\nMetrics:")
    print(f"  Before reranking: {metrics['before_rerank']}")
    print(f"  After reranking:  {metrics['after_rerank']}")


if __name__ == "__main__":
    main()
