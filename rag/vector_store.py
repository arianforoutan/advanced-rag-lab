"""Build or load the Chroma vector store and expose a policy-filtered retriever."""

from __future__ import annotations

import logging
import os

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from . import config

logger = logging.getLogger(__name__)


def get_or_create_vector_store(
    chunks: list[Document],
    embedding,
    persist_dir=config.CHROMA_DIR,
) -> Chroma:
    """Load the persisted Chroma store if present, otherwise build it from chunks."""
    persist_dir = str(persist_dir)
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        logger.info("Loading existing Chroma vector store from %s", persist_dir)
        return Chroma(persist_directory=persist_dir, embedding_function=embedding)

    logger.info("Creating new Chroma vector store from %d chunks", len(chunks))
    return Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=persist_dir,
    )


def build_public_retriever(
    vectorstore: Chroma,
    allowed_doc_types: list[str] = config.ALLOWED_DOC_TYPES,
    k: int = config.VECTOR_TOP_K,
) -> VectorStoreRetriever:
    """Return a similarity retriever restricted to the allowed document types."""
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": k,
            "filter": {"doc_type": {"$in": allowed_doc_types}},
        },
    )
