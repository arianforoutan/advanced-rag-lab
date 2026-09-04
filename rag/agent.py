"""Build the Insurellm LangChain agent around the RAG pipeline."""
from __future__ import annotations

import logging
from langchain.agents import create_agent
from langchain_core.tools import tool
from . import config
from .cache import PersistentSemanticCache
from .llm import build_embeddings
from .pipeline import RagPipeline

logger = logging.getLogger(__name__)

_cache_instance = PersistentSemanticCache(
    embeddings_model=build_embeddings(),
    cache_dir=config.CACHE_DIR,
    dimension=config.CACHE_EMBEDDING_DIM,
    similarity_threshold=config.CACHE_SIMILARITY_THRESHOLD,
)

def build_agent(pipeline: RagPipeline):
    """Create an agent whose only tool searches the Insurellm knowledge base."""
    @tool
    def search_knowledge_base(query: str) -> str:
        """Search the Insurellm knowledge base (Text & Knowledge Graph) for relevant documents and relationships."""
        return pipeline.search_knowledge_base(query)

    return create_agent(
        model=pipeline.llm,
        tools=[search_knowledge_base],
        system_prompt=config.AGENT_INSTRUCTIONS,
    )

def ask(agent, question: str, pipeline: RagPipeline | None = None) -> str:
    """Send one question to the agent and return its final text answer, checking persistent cache first."""
    cached_answer = _cache_instance.get(question)
    if cached_answer is not None:
        return cached_answer

    response = agent.invoke({"messages": [{"role": "user", "content": question}]})
    final_answer = response["messages"][-1].content

    if final_answer:
        _cache_instance.put(question, final_answer)

    return final_answer