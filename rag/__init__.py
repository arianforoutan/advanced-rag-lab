"""Insurellm Graph + Hybrid RAG.

A modular re-implementation of the original ``test_1.ipynb`` notebook. The
typical entry points are:

    from rag import RagPipeline, build_agent, ask

    pipeline = RagPipeline.build()
    agent = build_agent(pipeline)
    print(ask(agent, "What products does Insurellm offer?"))
"""

from __future__ import annotations

from .agent import ask, build_agent
from .pipeline import RagPipeline

__all__ = ["RagPipeline", "build_agent", "ask"]
