"""Populate the Neo4j knowledge graph from the public knowledge-base chunks.

Requires a running Neo4j instance (see docker-compose.yml):

    docker compose up -d
    python scripts/build_graph.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Make the project root importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.graph import GraphExtractor, GraphStore  # noqa: E402
from rag.ingestion import filter_public_chunks, load_documents, split_documents  # noqa: E402
from rag.llm import build_llm  # noqa: E402


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logger = logging.getLogger(__name__)

    documents = load_documents()
    chunks = split_documents(documents)
    public_chunks = filter_public_chunks(chunks)

    llm = build_llm()
    extractor = GraphExtractor(llm)

    graph_store = GraphStore.connect()
    graph_store.ensure_constraint()
    graph_store.populate(public_chunks, extractor)

    stats = graph_store.stats()
    logger.info(
        "Graph now holds %d entities and %d relationships",
        stats["entities"],
        stats["relationships"],
    )


if __name__ == "__main__":
    main()
