"""Query-time graph search: pull entities from a query, fetch related facts."""

from __future__ import annotations

import logging

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from .store import GraphStore

logger = logging.getLogger(__name__)

ENTITY_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Extract all key named entities (people, companies, products, roles, "
            "locations) from the query. Return them as comma-separated values only. "
            "If none found, return nothing.",
        ),
        ("human", "{query}"),
    ]
)

# Match the query's entities by name (case-insensitive) and return their direct
# relationships as readable "Source [TYPE] Target" facts.
_SUBGRAPH_CYPHER = """
MATCH (e:Entity)
WHERE toLower(e.name) IN [x IN $entities | toLower(x)]
MATCH (e)-[r:RELATION]-(target:Entity)
RETURN e.name + ' [' + r.type + '] ' + target.name AS fact
LIMIT 25
"""


class GraphSearcher:
    """Extract entities from a user query and retrieve related graph facts."""

    def __init__(self, llm, graph_store: GraphStore):
        self._graph_store = graph_store
        self._entity_chain = ENTITY_EXTRACTION_PROMPT | llm | StrOutputParser()

    def search(self, query: str) -> str:
        """Return bulleted graph facts related to the query, or "" if none/on error.

        The graph is a best-effort context source, so any failure (including an
        unreachable database) degrades gracefully to an empty result.
        """
        try:
            raw_entities = self._entity_chain.invoke({"query": query})
            entities = [e.strip() for e in raw_entities.split(",") if e.strip()]
            if not entities:
                return ""

            results = self._graph_store.query(_SUBGRAPH_CYPHER, {"entities": entities})
            facts = [record["fact"] for record in results]
            if facts:
                return "\n".join(f"- {fact}" for fact in facts)
        except Exception as exc:  # noqa: BLE001 - graph context is optional
            logger.warning("Graph search error: %s", exc)

        return ""
