"""Neo4j-backed storage for the extracted knowledge graph."""

from __future__ import annotations

import logging

from langchain_core.documents import Document
from langchain_neo4j import Neo4jGraph

from .. import config
from .extraction import GraphExtractor
from .schema import GraphExtraction

logger = logging.getLogger(__name__)

# Upsert entities as :Entity nodes and relationships as :RELATION edges. MERGE
# keeps ingestion idempotent, so re-running never duplicates nodes or edges.
_POPULATE_CYPHER = """
UNWIND $entities AS entity
MERGE (e:Entity {name: entity.name})
SET e.type = entity.type

WITH 1 as dummy
UNWIND $relationships AS rel
MERGE (source:Entity {name: rel.source})
MERGE (target:Entity {name: rel.target})
MERGE (source)-[r:RELATION {type: rel.type}]->(target)
"""


class GraphStore:
    """Thin wrapper around a ``Neo4jGraph`` connection for the Insurellm graph."""

    def __init__(self, graph: Neo4jGraph):
        self._graph = graph

    @classmethod
    def connect(
        cls,
        uri: str = config.NEO4J_URI,
        username: str | None = config.NEO4J_USERNAME,
        password: str | None = config.NEO4J_PASSWORD,
    ) -> "GraphStore":
        """Open a connection to Neo4j. Raises if the database is unreachable."""
        graph = Neo4jGraph(url=uri, username=username, password=password)
        return cls(graph)

    def query(self, cypher: str, params: dict | None = None):
        return self._graph.query(cypher, params=params or {})

    def ensure_constraint(self) -> None:
        """Guarantee entity names are unique so MERGE upserts behave correctly."""
        self.query(
            "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE;"
        )

    def populate(self, documents: list[Document], extractor: GraphExtractor) -> None:
        """Extract a graph from each document and upsert it into Neo4j.

        Per-document failures are logged and skipped so a single bad extraction
        never aborts the whole ingestion run.
        """
        total = len(documents)
        logger.info("Processing %d documents for graph ingestion", total)

        for idx, document in enumerate(documents, start=1):
            try:
                extraction: GraphExtraction = extractor.extract(document.page_content)
                entities_payload = [e.model_dump() for e in extraction.entities]
                rels_payload = [r.model_dump() for r in extraction.relationships]

                if entities_payload or rels_payload:
                    self.query(
                        _POPULATE_CYPHER,
                        {"entities": entities_payload, "relationships": rels_payload},
                    )

                if idx % 10 == 0 or idx == total:
                    logger.info("Processed %d/%d documents", idx, total)
            except Exception as exc:  # noqa: BLE001 - keep ingesting on per-doc errors
                logger.warning("Error processing document %d: %s", idx, exc)

    def stats(self) -> dict:
        """Return the current node and relationship counts."""
        entities = self.query("MATCH (n:Entity) RETURN count(n) AS total_entities")
        relations = self.query(
            "MATCH ()-[r:RELATION]->() RETURN count(r) AS total_relations"
        )
        return {
            "entities": entities[0]["total_entities"] if entities else 0,
            "relationships": relations[0]["total_relations"] if relations else 0,
        }
