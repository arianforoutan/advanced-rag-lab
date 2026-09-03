"""Cross-encoder reranking of retrieved candidates."""

from __future__ import annotations

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from langsmith import traceable
from .. import config


class Reranker:
    """Wraps a cross-encoder that scores (query, chunk) pairs jointly.

    Cross-encoders are more accurate than the bi-encoder used for retrieval, so
    they refine the fused candidate set down to the few best chunks.
    """

    def __init__(self, model_name: str = config.RERANKER_MODEL):
        self._model = CrossEncoder(model_name)
    @traceable(name="Cross-Encoder Rerank", run_type="chain")
    def rerank(
        self,
        query: str,
        docs: list[Document],
        top_k: int = config.RERANK_TOP_K,
    ) -> list[Document]:
        """Return the ``top_k`` most relevant documents for the query, best first."""
        if not docs:
            return []

        pairs = [[query, doc.page_content] for doc in docs]
        scores = self._model.predict(pairs)
        scored_docs = sorted(zip(docs, scores), key=lambda item: item[1], reverse=True)
        return [doc for doc, _ in scored_docs[:top_k]]
