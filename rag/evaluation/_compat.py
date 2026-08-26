"""Compatibility shims for third-party version mismatches.

``ragas`` 0.4.x imports ``langchain_community.chat_models.vertexai.ChatVertexAI``
at module load time, but ``langchain-community`` 0.4.x removed that submodule
(Vertex AI moved to its own integration package). Since this project never uses
Vertex AI, we register a stub module so ``import ragas`` succeeds without having
to pin/downgrade packages in the shared virtualenv.

The shim is intentionally minimal: ragas only needs the ``ChatVertexAI`` symbol
to exist for isinstance checks in its LLM factory. We use a plain ChatOpenAI, so
the stub is never actually instantiated.
"""

from __future__ import annotations

import importlib
import logging
import sys
import types

logger = logging.getLogger(__name__)

_MISSING_MODULE = "langchain_community.chat_models.vertexai"


def patch_langchain_community_vertexai() -> None:
    """Register a stub for the removed ``chat_models.vertexai`` module if needed.

    Idempotent and a no-op when the real module is importable. Must be called
    before ``import ragas``.
    """
    if _MISSING_MODULE in sys.modules:
        return

    try:
        importlib.import_module(_MISSING_MODULE)
        return  # Real module is present; no shim required.
    except ModuleNotFoundError:
        pass

    module = types.ModuleType(_MISSING_MODULE)

    class ChatVertexAI:  # noqa: D401 - stub only; never instantiated here.
        """Stub for the removed ``langchain_community`` Vertex AI chat model."""

    module.ChatVertexAI = ChatVertexAI
    sys.modules[_MISSING_MODULE] = module
    logger.debug(
        "Registered compatibility stub for %s (ragas <-> langchain-community mismatch)",
        _MISSING_MODULE,
    )
