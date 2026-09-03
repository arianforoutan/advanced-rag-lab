"""Input guardrails applied before any retrieval happens."""

from __future__ import annotations
from langsmith import traceable
from . import config

@traceable(name="Guardrail Check", run_type="chain")
def question_policy(query: str) -> str | None:
    """Return a canned refusal for disallowed queries, or ``None`` to proceed.

    - Queries mentioning private/personal terms get the sensitive-info refusal.
    - Queries that mention no Insurellm-related term are treated as off-topic.
    """
    normalized = query.casefold()

    if any(term in normalized for term in config.SENSITIVE_QUERY_TERMS):
        return config.SENSITIVE_RESPONSE

    if not any(term in normalized for term in config.INSURELLM_TERMS):
        return config.OFF_TOPIC_RESPONSE

    return None
