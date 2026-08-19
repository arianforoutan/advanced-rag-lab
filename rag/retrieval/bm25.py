"""Lexical (keyword) retrieval over public chunks using Okapi BM25."""

from __future__ import annotations

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from .. import config


class BM25Index:
    """A BM25 index built over a fixed list of document chunks.

    Tokenization is a simple case-folded whitespace split, matching the vector
    path's lowercase behavior closely enough for keyword overlap.
    """

    def __init__(self, chunks: list[Document]):
        self._chunks = chunks
        self._bm25 = BM25Okapi([self._tokenize(doc.page_content) for doc in chunks])

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return text.casefold().split()

    def search(self, query: str, top_k: int = config.BM25_TOP_K) -> list[Document]:
        """Return the top-k chunks by BM25 score, dropping zero-score matches."""
        scores = self._bm25.get_scores(self._tokenize(query))
        top_indices = scores.argsort()[::-1][:top_k]
        return [self._chunks[idx] for idx in top_indices if scores[idx] > 0]
