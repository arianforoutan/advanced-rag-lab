"""The RAG pipeline: assembles every component and runs the retrieval flow.

``RagPipeline`` is the single object the agent and scripts depend on. Build it
once with :meth:`RagPipeline.build`, then call :meth:`search_knowledge_base`
(directly, or through the agent tool created in :mod:`rag.agent`).
"""

from __future__ import annotations

import logging

from . import config
from .graph import GraphSearcher, GraphStore
from .guardrails import question_policy
from .ingestion import filter_public_chunks, load_documents, split_documents
from .llm import build_embeddings, build_llm
from .retrieval import BM25Index, Reranker, generate_queries, hybrid_search
from .vector_store import build_public_retriever, get_or_create_vector_store
from langsmith import traceable
logger = logging.getLogger(__name__)


class RagPipeline:
    """Guardrails, graph search, hybrid retrieval, and reranking in one place."""

    def __init__(
        self,
        *,
        llm,
        vectorstore,
        retriever,
        bm25_index: BM25Index,
        reranker: Reranker,
        graph_searcher: GraphSearcher | None = None,
    ):
        self.llm = llm
        self.vectorstore = vectorstore
        self.retriever = retriever
        self.bm25_index = bm25_index
        self.reranker = reranker
        self.graph_searcher = graph_searcher

    @classmethod
    def build(cls, *, enable_graph: bool = True) -> "RagPipeline":
        """Construct every component from the knowledge base and configuration.

        Documents are always loaded and chunked because BM25 indexes the chunk
        list directly. The vector store is loaded from disk if it already exists.
        The Neo4j graph is optional: when ``enable_graph`` is true we attempt to
        connect, and fall back to text-only retrieval if it is unreachable.
        """
        documents = load_documents()
        chunks = split_documents(documents)
        public_chunks = filter_public_chunks(chunks)
        logger.info("Total chunks: %d", len(chunks))
        logger.info("Public chunks: %d", len(public_chunks))

        embedding = build_embeddings()
        vectorstore = get_or_create_vector_store(chunks, embedding)
        retriever = build_public_retriever(vectorstore)
        bm25_index = BM25Index(public_chunks)
        reranker = Reranker()
        llm = build_llm()

        graph_searcher = cls._try_connect_graph(llm) if enable_graph else None

        return cls(
            llm=llm,
            vectorstore=vectorstore,
            retriever=retriever,
            bm25_index=bm25_index,
            reranker=reranker,
            graph_searcher=graph_searcher,
        )

    @staticmethod
    def _try_connect_graph(llm) -> GraphSearcher | None:
        """Connect to Neo4j, or return ``None`` if it is unavailable."""
        try:
            graph_store = GraphStore.connect()
            
            graph_store.query("RETURN 1 AS ok")
            logger.info("Connected to Neo4j knowledge graph")
            return GraphSearcher(llm, graph_store)
        except Exception as exc:  
            logger.warning("Knowledge graph unavailable, continuing text-only: %s", exc)
            return None


    @traceable(
        name="Advanced RAG Retrieval Flow",
        run_type="chain",
        metadata={"component": "rag_pipeline"}
    )
    def search_knowledge_base(self, query: str) -> str:
        """Run the full retrieval flow and return combined context for the LLM.

        Steps: guardrail check -> graph facts -> query expansion -> hybrid
        search (vector + BM25 + RRF) -> cross-encoder rerank -> merge graph and
        text context.
        """
        policy_response = question_policy(query)
        if policy_response:
            return policy_response

        graph_context = ""
        if self.graph_searcher is not None:
            graph_context = self.graph_searcher.search(query)
        if graph_context:
            logger.info("Graph facts found:\n%s", graph_context)
        else:
            logger.info("Graph facts: none found")

        queries = generate_queries(self.llm, query)
        logger.info("Generated alternative queries: %s", queries)

        docs = hybrid_search(self.retriever, self.bm25_index, query, queries)
        if not docs:
            logger.info("No documents found via hybrid search")
            return graph_context or "No relevant information found."

        docs = docs[: config.HYBRID_CANDIDATE_K]
        logger.info(
            "Top chunks after hybrid search (RRF): %s",
            [doc.metadata.get("source") for doc in docs],
        )

        reranked_docs = self.reranker.rerank(query, docs, top_k=config.RERANK_TOP_K)
        logger.info(
            "Top chunks after reranking: %s",
            [doc.metadata.get("source") for doc in reranked_docs],
        )

        text_context = "\n\n".join(doc.page_content for doc in reranked_docs)

        combined = []
        if graph_context:
            combined.append(f"### KNOWLEDGE GRAPH RELATIONSHIPS:\n{graph_context}")
        if text_context:
            combined.append(f"### DETAILED TEXT CONTEXT:\n{text_context}")

        return "\n\n".join(combined) if combined else "No relevant information found."
