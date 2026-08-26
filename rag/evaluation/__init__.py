"""Three-layer evaluation for the Insurellm RAG pipeline.

The layers answer different questions and have different costs, so they are kept
separate and can be run independently:

1. **Retrieval** (:func:`run_retrieval_evaluation`) — Hit@k / MRR before and
   after reranking. Cheap and deterministic; runs offline with
   ``use_query_expansion=False``.
2. **Generation** (:func:`run_ragas_evaluation`) — end-to-end Ragas
   LLM-as-judge metrics on the real retrieval path. Makes API calls.
3. **Guardrails** (:func:`run_guardrail_evaluation`) — deterministic pass/fail
   check of the input policy. Free and offline.

:func:`run_full_evaluation` runs all three and is resilient: a failure in the
(networked, dependency-heavy) Ragas layer never discards the retrieval and
guardrail results.
"""

from __future__ import annotations

import logging

from ..pipeline import RagPipeline
from .dataset import EVALUATION_SET, GUARDRAIL_CASES
from .generation import run_ragas_evaluation
from .guardrails import run_guardrail_evaluation
from .retrieval import run_retrieval_evaluation

logger = logging.getLogger(__name__)

__all__ = [
    "EVALUATION_SET",
    "GUARDRAIL_CASES",
    "run_retrieval_evaluation",
    "run_ragas_evaluation",
    "run_guardrail_evaluation",
    "run_full_evaluation",
]


def run_full_evaluation(
    pipeline: RagPipeline,
    *,
    use_query_expansion: bool = True,
    run_ragas: bool = True,
) -> dict:
    """Run all three evaluation layers and return their results in one dict.

    Keys: ``retrieval``, ``guardrails``, ``ragas``. The ``ragas`` value is the
    Ragas ``EvaluationResult`` on success, or ``None`` if the layer was skipped
    (``run_ragas=False``) or raised — the offline layers always run regardless.
    """
    results: dict = {}

    logger.info("Layer 1/3: retrieval metrics (Hit@k, MRR)...")
    results["retrieval"] = run_retrieval_evaluation(
        pipeline, use_query_expansion=use_query_expansion
    )

    logger.info("Layer 2/3: guardrail policy checks...")
    results["guardrails"] = run_guardrail_evaluation()

    results["ragas"] = None
    if run_ragas:
        logger.info("Layer 3/3: end-to-end Ragas metrics...")
        try:
            results["ragas"] = run_ragas_evaluation(pipeline)
        except Exception:  # noqa: BLE001 - keep the offline layers' results.
            logger.exception(
                "Ragas evaluation failed; retrieval and guardrail results are "
                "still available."
            )
    else:
        logger.info("Layer 3/3: Ragas skipped (run_ragas=False).")

    return results
