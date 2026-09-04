"""Central configuration for the Insurellm RAG pipeline.

All tunable constants, model names, filesystem paths, credentials, guardrail
term sets, and the agent system prompt live here so the rest of the package
never reads environment variables or hard-codes paths directly.

Paths are resolved relative to this package (``BASE_DIR``) rather than the
current working directory, so scripts run identically from anywhere.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load variables from the nearest .env once, at import time. ``override=True``
# mirrors the original notebook so a .env value wins over a stale shell export.
load_dotenv(override=True)


# ---------------------------------------------------------------------------
# Filesystem paths
# ---------------------------------------------------------------------------
# BASE_DIR is the test_RAG project root (this file lives in test_RAG/rag/).
BASE_DIR = Path(__file__).resolve().parent.parent
KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge-base"
CHROMA_DIR = BASE_DIR / "ch_database"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHAT_MODEL = "gpt-5-nano"
RERANKER_MODEL = "BAAI/bge-reranker-base"

# The model used as the Ragas judge during evaluation. Kept separate from
# CHAT_MODEL so the evaluator can differ from the generator (avoiding
# self-evaluation bias); override with the JUDGE_MODEL environment variable.
JUDGE_MODEL = os.getenv("JUDGE_MODEL", CHAT_MODEL)


# ---------------------------------------------------------------------------
# Credentials / endpoints (read from the environment / .env)
# ---------------------------------------------------------------------------
GAPGPT_API_KEY = os.getenv("GAPGPT_API_KEY")
GAPGPT_BASE_URL = os.getenv("GAPGPT_BASE_URL")

NEO4J_URI = "bolt://localhost:8687"
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


# ---------------------------------------------------------------------------
# Retrieval tuning
# ---------------------------------------------------------------------------
VECTOR_TOP_K = 5          # candidates pulled from the Chroma vector store per query
BM25_TOP_K = 5            # candidates pulled from the BM25 index per query
RRF_K = 60                # Reciprocal Rank Fusion smoothing constant
HYBRID_CANDIDATE_K = 10   # fused candidates handed to the reranker
RERANK_TOP_K = 3          # documents kept after cross-encoder reranking


# ---------------------------------------------------------------------------
# Document access policy
# ---------------------------------------------------------------------------
# Document folders (doc_type) the assistant is allowed to retrieve from.
ALLOWED_DOC_TYPES = ["company", "products", "contracts", "employees"]


# ---------------------------------------------------------------------------
# Semantic Cache Configuration
# ---------------------------------------------------------------------------
CACHE_DIR = BASE_DIR / "cache_store"
CACHE_SIMILARITY_THRESHOLD = 0.90
CACHE_EMBEDDING_DIM = 384  # بعد خروجی مدل all-MiniLM-L6-v2




# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------
# Queries containing these terms are treated as requests for private data.
SENSITIVE_QUERY_TERMS = {
    "salary",
    "compensation",
    "bonus",
    "performance review",
    "date of birth",
    "birthday",
    "home address",
    "phone number",
}

# A query must mention at least one of these terms to be considered on-topic.
INSURELLM_TERMS = {
    "insurellm",
    "carllm",
    "homellm",
    "lifellm",
    "healthllm",
    "bizllm",
    "markellm",
    "claimllm",
    "rellm",
    "products",
    "company",
    "contracts",
    "employees",
}

OFF_TOPIC_RESPONSE = "I can only answer questions related to Insurellm."
SENSITIVE_RESPONSE = (
    "I can help with Insurellm's public company, product, and contract information, "
    "but I cannot provide employee personal or compensation information."
)


# ---------------------------------------------------------------------------
# Agent system prompt
# ---------------------------------------------------------------------------
AGENT_INSTRUCTIONS = """
You are a specialized AI assistant strictly representing the company Insurellm.

RULES:
1. You MUST ALWAYS use the `search_knowledge_base` tool first for ANY question asked by the user.
2. You are ONLY allowed to answer questions related to Insurellm, its company information, products, contracts, and employees.
3. If the user asks general knowledge questions, off-topic questions (e.g., geography, general coding, weather, math), or anything NOT related to Insurellm, politely refuse to answer and state: "I can only answer questions related to Insurellm."
4. Do NOT use your pre-trained general knowledge to answer off-topic queries.
"""
