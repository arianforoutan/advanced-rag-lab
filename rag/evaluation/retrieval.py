"""Retrieval-quality metrics: Hit@k and MRR, before and after reranking.

This layer measures the retrieval stack in isolation — it never calls a judge
model, so it is cheap, deterministic (aside from optional LLM query expansion),
and answers the question the end-to-end metrics cannot: *did the right document
get retrieved, and does reranking actually help?*

It uses the same production components (query expansion + hybrid vector/BM25
search fused with RRF, then cross-encoder reranking) so the numbers reflect what
the agent really retrieves.
"""

from __future__ import annotations

import logging
import os

from langchain_core.documents import Document

from .. import config
from ..pipeline import RagPipeline
from ..retrieval import generate_queries, hybrid_search
from .dataset import EVALUATION_SET

logger = logging.getLogger(__name__)


def _source_names(documents: list[Document]) -> list[str]:
    return [os.path.basename(doc.metadata.get("source", "")) for doc in documents]


def _metrics(results_by_query, k: int) -> dict:
    """Compute Hit@k and MRR from ``(example, results)`` pairs."""
    hits = 0
    reciprocal_ranks = []

    for example, results in results_by_query:
        names = _source_names(results[:k])
        first_match = next(
            (
                rank
                for rank, name in enumerate(names, start=1)
                if name in example["expected_sources"]
            ),
            None,
        )
        hits += first_match is not None
        reciprocal_ranks.append(1 / first_match if first_match else 0)
        logger.info(
            "%s\n  Sources: %s\n  Match rank: %s",
            example["query"],
            names,
            first_match,
        )

    total = len(results_by_query)
    return {
        f"Hit@{k}": hits / total if total else 0,
        "MRR": sum(reciprocal_ranks) / total if total else 0,
    }


def run_retrieval_evaluation(
    pipeline: RagPipeline,
    examples: list[dict] = EVALUATION_SET,
    *,
    use_query_expansion: bool = True,
) -> dict:
    """Return Hit@k / MRR for hybrid retrieval, before and after reranking.

    Set ``use_query_expansion=False`` to skip the LLM query-expansion step and
    run the retrieval metrics fully offline (no API calls).
    """
    before_rerank = []
    after_rerank = []

    for example in examples:
        query = example["query"]

        alternatives = generate_queries(pipeline.llm, query) if use_query_expansion else None
        hybrid_results = hybrid_search(
            pipeline.retriever, pipeline.bm25_index, query, alternatives
        )
        before_rerank.append((example, hybrid_results))

        candidates = hybrid_results[: config.HYBRID_CANDIDATE_K]
        reranked = pipeline.reranker.rerank(query, candidates, top_k=config.RERANK_TOP_K)
        after_rerank.append((example, reranked))

    logger.info("=== Hybrid retrieval: before reranking ===")
    before_metrics = _metrics(before_rerank, config.VECTOR_TOP_K)

    logger.info("=== Hybrid retrieval: after reranking ===")
    after_metrics = _metrics(after_rerank, config.RERANK_TOP_K)

    return {"before_rerank": before_metrics, "after_rerank": after_metrics}
