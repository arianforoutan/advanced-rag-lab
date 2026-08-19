"""Extract a schema-constrained knowledge graph from document text via the LLM."""

from __future__ import annotations

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

from .schema import (
    ALLOWED_ENTITY_TYPES,
    ALLOWED_RELATIONSHIP_TYPES,
    GraphExtraction,
)


def _build_prompt() -> ChatPromptTemplate:
    # The allowed type lists are baked into the system message; the parser's
    # format instructions are injected at call time via {format_instructions}.
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"""
You extract structured knowledge graph information from company documents.

Your task:
1. Identify meaningful entities.
2. Identify meaningful relationships between those entities.
3. Return ONLY information explicitly supported by the document.
4. Do NOT invent entities or relationships.
5. Use only the allowed entity types.
6. Use only the allowed relationship types.
7. Keep entity names consistent.
8. Do not create nodes for simple scalar facts such as salary, date of birth,
   contract amount, performance score, or dates unless the schema explicitly
   requires them as entities.

Allowed entity types:
{sorted(ALLOWED_ENTITY_TYPES)}

Allowed relationship types:
{sorted(ALLOWED_RELATIONSHIP_TYPES)}

Important:
- Employees are valid entities and must NOT be ignored.
- A company, product, contract, employee, job position, department,
  location, or project should be extracted when clearly supported.
- Do not infer relationships merely because two entities appear in the
  same document.

{{format_instructions}}
""",
            ),
            ("human", "Extract the knowledge graph from this document:\n\n{document}"),
        ]
    )


class GraphExtractor:
    """LLM chain that turns document text into a validated ``GraphExtraction``."""

    def __init__(self, llm):
        self._parser = PydanticOutputParser(pydantic_object=GraphExtraction)
        self._chain = _build_prompt() | llm | self._parser

    def extract(self, text: str) -> GraphExtraction:
        return self._chain.invoke(
            {
                "document": text,
                "format_instructions": self._parser.get_format_instructions(),
            }
        )
