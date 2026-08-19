"""Reciprocal Rank Fusion for combining multiple ranked result lists."""

from __future__ import annotations

import os

from langchain_core.documents import Document

from .. import config


def reciprocal_rank_fusion(
    ranked_results: list[list[Document]],
    k: int = config.RRF_K,
) -> list[Document]:
    """Fuse several ranked lists into one ordering using RRF.

    Each document contributes ``1 / (k + rank)`` from every list it appears in.
    Documents are de-duplicated on ``(source filename, content)`` so the same
    chunk returned by both the vector and BM25 paths is fused into a single,
    score-boosted entry -- the filename basename is used so the merge is stable
    whether a ``source`` path is stored absolute or relative.
    """
    scores: dict = {}
    unique_documents: dict = {}

    for results in ranked_results:
        for rank, document in enumerate(results, start=1):
            key = (
                os.path.basename(document.metadata.get("source", "")),
                document.page_content,
            )
            scores[key] = scores.get(key, 0.0) + 1 / (k + rank)
            unique_documents[key] = document

    ranked_keys = sorted(scores, key=scores.get, reverse=True)
    return [unique_documents[key] for key in ranked_keys]
