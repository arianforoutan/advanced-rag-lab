"""Factory functions for the shared language and embedding models."""

from __future__ import annotations

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

from . import config


def build_llm() -> ChatOpenAI:
    """Create the chat model used for generation, query expansion, and extraction."""
    return ChatOpenAI(
        model=config.CHAT_MODEL,
        base_url=config.GAPGPT_BASE_URL,
        api_key=config.GAPGPT_API_KEY,
    )


def build_embeddings() -> HuggingFaceEmbeddings:
    """Create the sentence-transformer embeddings used by the Chroma store."""
    return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
