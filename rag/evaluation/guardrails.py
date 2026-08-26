"""Guardrail evaluation: does the input policy refuse the right queries?

The guardrail is deterministic and needs no model, so this layer is a fast,
free, pass/fail regression check for the two explicit policy features: refusing
off-topic questions and refusing requests for sensitive personal data.
"""

from __future__ import annotations

import logging

from .. import config
from ..guardrails import question_policy
from .dataset import GUARDRAIL_CASES

logger = logging.getLogger(__name__)


def _classify(response: str | None) -> str:
    """Map a ``question_policy`` return value to a policy-outcome label."""
    if response is None:
        return "allow"
    if response == config.SENSITIVE_RESPONSE:
        return "sensitive"
    if response == config.OFF_TOPIC_RESPONSE:
        return "off_topic"
    return "other"


def run_guardrail_evaluation(cases: list[dict] = GUARDRAIL_CASES) -> dict:
    """Return the guardrail pass rate and per-case details."""
    details = []
    passed = 0

    for case in cases:
        actual = _classify(question_policy(case["query"]))
        ok = actual == case["expected"]
        passed += ok
        details.append(
            {
                "query": case["query"],
                "expected": case["expected"],
                "actual": actual,
                "ok": ok,
            }
        )
        logger.info(
            "[%s] %s (expected=%s, actual=%s)",
            "PASS" if ok else "FAIL",
            case["query"],
            case["expected"],
            actual,
        )

    total = len(cases)
    return {
        "pass_rate": passed / total if total else 0,
        "passed": passed,
        "total": total,
        "details": details,
    }
