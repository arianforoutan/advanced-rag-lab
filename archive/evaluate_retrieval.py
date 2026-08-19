"""Evaluate the Hybrid Retrieval pipeline before and after reranking.

This file is intentionally independent from test_1.ipynb. It reads the existing
knowledge base and Chroma database; it does not create, delete, or update either.
"""

import glob
import os

from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder


# Keep these settings aligned with the retrieval cells in test_1.ipynb.
ALLOWED_DOC_TYPES = ["company", "products", "contracts"]
DB_NAME = "ch_database"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
VECTOR_TOP_K = 5
RRF_CANDIDATE_K = 10
RERANK_TOP_K = 3

EVALUATION_SET = [
    {
        "query": "What products does Insurellm offer?",
        "expected_sources": {"overview.md", "about.md"},
    },
    {
        "query": "What are the features of Rellm?",
        "expected_sources": {"Rellm.md"},
    },
    {
        "query": "What does Claimllm do?",
        "expected_sources": {"Claimllm.md", "overview.md"},
    },
    {
        "query": "Which Insurellm contracts use Homellm?",
        "expected_sources": {"Homellm.md"},
    },
]


def load_public_chunks():
    """Load and split markdown files, then exclude employee/HR documents."""
    documents = []
    for folder in glob.glob("knowledge-base/*"):
        doc_type = os.path.basename(folder)
        loader = DirectoryLoader(
            folder,
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )
        for document in loader.load():
            document.metadata["doc_type"] = doc_type
            documents.append(document)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return [
        chunk
        for chunk in splitter.split_documents(documents)
        if chunk.metadata.get("doc_type") in ALLOWED_DOC_TYPES
    ]


def reciprocal_rank_fusion(ranked_results, k=60):
    """Fuse Vector and BM25 result lists while preserving distinct chunks."""
    scores = {}
    unique_documents = {}

    for results in ranked_results:
        for rank, document in enumerate(results, start=1):
            key = (document.metadata.get("source", ""), document.page_content)
            scores[key] = scores.get(key, 0.0) + 1 / (k + rank)
            unique_documents[key] = document

    ranked_keys = sorted(scores, key=scores.get, reverse=True)
    return [unique_documents[key] for key in ranked_keys]


def source_names(documents):
    return [os.path.basename(document.metadata.get("source", "")) for document in documents]


def calculate_metrics(results_by_query, k):
    """Calculate Hit@k and MRR from result lists and expected source filenames."""
    hits = 0
    reciprocal_ranks = []

    for example, results in results_by_query:
        names = source_names(results[:k])
        first_match = next(
            (
                rank
                for rank, name in enumerate(names, start=1)
                if name in example["expected_sources"]
            ),
            None,
        )
        hits += first_match is not None
        reciprocal_ranks.append(1 / first_match if first_match else 0)
        print(f"{example['query']}\n  Sources: {names}\n  Match rank: {first_match}")

    total = len(results_by_query)
    return {
        f"Hit@{k}": hits / total if total else 0,
        "MRR": sum(reciprocal_ranks) / total if total else 0,
    }


def main():
    public_chunks = load_public_chunks()
    embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embedding)
    vector_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": VECTOR_TOP_K,
            "filter": {"doc_type": {"$in": ALLOWED_DOC_TYPES}},
        },
    )
    bm25 = BM25Okapi([chunk.page_content.casefold().split() for chunk in public_chunks])
    reranker = CrossEncoder("BAAI/bge-reranker-base")

    before_rerank = []
    after_rerank = []

    for example in EVALUATION_SET:
        query = example["query"]

        # Hybrid Retrieval: Vector Search + BM25, then RRF.
        vector_results = vector_retriever.invoke(query)
        bm25_scores = bm25.get_scores(query.casefold().split())
        bm25_indices = bm25_scores.argsort()[::-1][:VECTOR_TOP_K]
        bm25_results = [
            public_chunks[index]
            for index in bm25_indices
            if bm25_scores[index] > 0
        ]
        hybrid_results = reciprocal_rank_fusion([vector_results, bm25_results])
        before_rerank.append((example, hybrid_results))

        # Rerank only the strongest Hybrid candidates.
        candidates = hybrid_results[:RRF_CANDIDATE_K]
        pairs = [[query, document.page_content] for document in candidates]
        scores = reranker.predict(pairs)
        reranked = [
            document
            for document, _ in sorted(
                zip(candidates, scores), key=lambda item: item[1], reverse=True
            )
        ]
        after_rerank.append((example, reranked))

    print("\n=== Hybrid Retrieval: before reranking ===")
    before_metrics = calculate_metrics(before_rerank, VECTOR_TOP_K)
    print(f"Metrics: {before_metrics}")

    print("\n=== Hybrid Retrieval: after reranking ===")
    after_metrics = calculate_metrics(after_rerank, RERANK_TOP_K)
    print(f"Metrics: {after_metrics}")


if __name__ == "__main__":
    main()
