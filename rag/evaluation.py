"""Retrieval evaluation: Hit@k and MRR, before and after reranking.

The metrics use the same hybrid retrieval path as production (vector + BM25 +
RRF), so the numbers reflect what the agent actually retrieves.
"""

from __future__ import annotations

import logging
import os

from langchain_core.documents import Document

from . import config
from .pipeline import RagPipeline
from .retrieval import hybrid_search

logger = logging.getLogger(__name__)

# Labelled queries with the source filenames a good retriever should surface.
# Edit this set as the knowledge base and expected answers evolve.
EVALUATION_SET = [
    {
        "query": "What products does Insurellm offer?",
        "expected_sources": {"overview.md", "about.md"},
    },
    {
        "query": "What are the features of Rellm?",
        "expected_sources": {"Rellm.md"},
    },
    {
        "query": "What does Claimllm do?",
        "expected_sources": {"Claimllm.md", "overview.md"},
    },
    {
        "query": "Which Insurellm contracts use Homellm?",
        "expected_sources": {"Homellm.md", "overview.md"},
    },
]


def _source_names(documents: list[Document]) -> list[str]:
    return [os.path.basename(doc.metadata.get("source", "")) for doc in documents]


def _metrics(results_by_query, k: int) -> dict:
    """Compute Hit@k and MRR from (example, results) pairs."""
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


def evaluate_retrieval(pipeline: RagPipeline, examples=EVALUATION_SET) -> dict:
    """Return Hit@k / MRR for hybrid retrieval, before and after reranking."""
    before_rerank = []
    after_rerank = []

    for example in examples:
        query = example["query"]

        hybrid_results = hybrid_search(pipeline.retriever, pipeline.bm25_index, query)
        before_rerank.append((example, hybrid_results))

        candidates = hybrid_results[: config.HYBRID_CANDIDATE_K]
        reranked = pipeline.reranker.rerank(query, candidates, top_k=config.RERANK_TOP_K)
        after_rerank.append((example, reranked))

    logger.info("=== Hybrid retrieval: before reranking ===")
    before_metrics = _metrics(before_rerank, config.VECTOR_TOP_K)

    logger.info("=== Hybrid retrieval: after reranking ===")
    after_metrics = _metrics(after_rerank, config.RERANK_TOP_K)

    return {"before_rerank": before_metrics, "after_rerank": after_metrics}
