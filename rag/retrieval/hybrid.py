"""Hybrid search: run vector + BM25 retrieval per query and fuse with RRF."""

from __future__ import annotations

from langchain_core.documents import Document

from .bm25 import BM25Index
from .fusion import reciprocal_rank_fusion


def hybrid_search(
    retriever,
    bm25_index: BM25Index,
    query: str,
    alternative_queries: list[str] | None = None,
) -> list[Document]:
    """Retrieve for the query and each alternative via both paths, then fuse.

    For every query variant we collect one ranked list from the vector
    retriever and one from BM25; all lists are then merged with Reciprocal Rank
    Fusion into a single ordering.
    """
    ranked_results: list[list[Document]] = []
    for search_query in [query, *(alternative_queries or [])]:
        ranked_results.append(retriever.invoke(search_query))
        ranked_results.append(bm25_index.search(search_query))

    return reciprocal_rank_fusion(ranked_results)
