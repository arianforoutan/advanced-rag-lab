"""Pydantic schemas and the allowed type vocabulary for graph extraction."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Entity(BaseModel):
    name: str
    type: str


class Relationship(BaseModel):
    source: str
    type: str
    target: str


class GraphExtraction(BaseModel):
    entities: list[Entity] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)


# The extractor is constrained to these vocabularies so the graph stays clean
# and queryable rather than accumulating free-form node and edge types.
ALLOWED_ENTITY_TYPES = {
    "Employee",
    "Company",
    "Product",
    "Contract",
    "JobPosition",
    "Department",
    "Location",
    "Project",
    "Document",
}

ALLOWED_RELATIONSHIP_TYPES = {
    "WORKS_FOR",
    "HAS_ROLE",
    "LOCATED_IN",
    "WORKED_ON",
    "MENTIONED_IN",
    "OFFERS",
    "HAS_CONTRACT",
    "POSTS",
    "IN_DEPARTMENT",
    "WITH",
    "FOR_PRODUCT",
}
