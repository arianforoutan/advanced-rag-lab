"""Build the Insurellm LangChain agent around the RAG pipeline."""

from __future__ import annotations

from langchain.agents import create_agent
from langchain_core.tools import tool

from . import config
from .pipeline import RagPipeline


def build_agent(pipeline: RagPipeline):
    """Create an agent whose only tool searches the Insurellm knowledge base.

    The tool is a thin closure over ``pipeline.search_knowledge_base`` so the
    agent shares the pipeline's already-loaded models and connections.
    """

    @tool
    def search_knowledge_base(query: str) -> str:
        """Search the Insurellm knowledge base (Text & Knowledge Graph) for relevant documents and relationships."""
        return pipeline.search_knowledge_base(query)

    return create_agent(
        model=pipeline.llm,
        tools=[search_knowledge_base],
        system_prompt=config.AGENT_INSTRUCTIONS,
    )


def ask(agent, question: str) -> str:
    """Send one question to the agent and return its final text answer."""
    response = agent.invoke({"messages": [{"role": "user", "content": question}]})
    return response["messages"][-1].content
