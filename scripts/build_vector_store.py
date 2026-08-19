"""Build (or load) the Chroma vector store from the knowledge base.

    python scripts/build_vector_store.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Make the project root importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.ingestion import filter_public_chunks, load_documents, split_documents  # noqa: E402
from rag.llm import build_embeddings  # noqa: E402
from rag.vector_store import get_or_create_vector_store  # noqa: E402


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logger = logging.getLogger(__name__)

    documents = load_documents()
    chunks = split_documents(documents)
    public_chunks = filter_public_chunks(chunks)

    embedding = build_embeddings()
    get_or_create_vector_store(chunks, embedding)

    logger.info(
        "Documents: %d | Total chunks: %d | Public chunks: %d",
        len(documents),
        len(chunks),
        len(public_chunks),
    )


if __name__ == "__main__":
    main()
