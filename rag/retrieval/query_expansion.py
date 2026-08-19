"""LLM-based query expansion to broaden retrieval recall."""

from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

QUERY_EXPANSION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a search query optimizer for the Insurellm knowledge base.
Generate exactly 3 alternative search queries. Preserve the exact intent; do not add or reinterpret entities.
Return only the queries, one per line, and do not answer the question.""",
        ),
        ("human", "{query}"),
    ]
)


def generate_queries(llm, query: str) -> list[str]:
    """Return alternative phrasings of ``query`` (one per non-empty output line)."""
    chain = QUERY_EXPANSION_PROMPT | llm | StrOutputParser()
    response = chain.invoke({"query": query})
    return [line.strip() for line in response.splitlines() if line.strip()]
