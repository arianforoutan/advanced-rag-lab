"""End-to-end generation quality via Ragas (LLM-as-judge metrics).

For each labelled query we run the production retrieval path (query expansion ->
hybrid vector/BM25 search -> cross-encoder rerank), generate an answer grounded
in those retrieved chunks, then score the (question, contexts, answer, reference)
tuple with four Ragas metrics:

- ``ResponseGroundedness``               — is the answer supported by the contexts?
- ``ResponseRelevancy``                  — does the answer address the question?
- ``LLMContextPrecisionWithReference``   — are the retrieved chunks relevant?
- ``LLMContextRecall``                   — do the chunks cover the reference answer?

The judge model is separate from the generator (``config.JUDGE_MODEL``) to avoid
self-evaluation bias, and defaults to the same model only when not overridden.
Ragas and its LangChain wrappers are imported lazily so the other evaluation
layers remain usable even if ragas cannot be imported in this environment.
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from .. import config
from ..llm import build_embeddings, build_llm
from ..pipeline import RagPipeline
from ..retrieval import generate_queries, hybrid_search
from . import _compat
from .dataset import EVALUATION_SET

logger = logging.getLogger(__name__)


def _generate_response(llm, query: str, context: str) -> str:
    """Answer ``query`` using only ``context`` — the answer Ragas will score."""
    system_prompt = (
        "You are an expert assistant. Answer the question using ONLY the provided "
        "context. If the context does not contain enough information, state that clearly."
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {query}"
    response = llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    )
    return response.content if hasattr(response, "content") else str(response)


def run_ragas_evaluation(
    pipeline: RagPipeline,
    examples: list[dict] = EVALUATION_SET,
    *,
    judge_llm=None,
    embeddings=None,
):
    """Run the Ragas generation metrics and return the ``EvaluationResult``.

    ``judge_llm`` / ``embeddings`` default to a freshly built judge model
    (``config.JUDGE_MODEL``) and the shared sentence-transformer embeddings.
    """
    # Register the langchain-community compatibility stub before importing ragas.
    _compat.patch_langchain_community_vertexai()

    from datasets import Dataset
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseGroundedness,
        ResponseRelevancy,
    )

    judge = LangchainLLMWrapper(judge_llm or build_llm(config.JUDGE_MODEL))
    judge_embeddings = LangchainEmbeddingsWrapper(embeddings or build_embeddings())

    user_inputs, responses, retrieved_contexts, references = [], [], [], []

    logger.info("Collecting pipeline outputs for %d queries...", len(examples))
    for example in examples:
        query = example["query"]

        alternatives = generate_queries(pipeline.llm, query)
        docs = hybrid_search(pipeline.retriever, pipeline.bm25_index, query, alternatives)
        candidates = docs[: config.HYBRID_CANDIDATE_K]
        reranked_docs = pipeline.reranker.rerank(
            query, candidates, top_k=config.RERANK_TOP_K
        )

        contexts = [doc.page_content for doc in reranked_docs]
        answer = _generate_response(pipeline.llm, query, "\n\n".join(contexts))

        user_inputs.append(query)
        responses.append(answer)
        retrieved_contexts.append(contexts)
        references.append(example["reference"])

    dataset = Dataset.from_dict(
        {
            "user_input": user_inputs,
            "response": responses,
            "retrieved_contexts": retrieved_contexts,
            "reference": references,
        }
    )

    metrics = [
        ResponseGroundedness(llm=judge),
        ResponseRelevancy(llm=judge, embeddings=judge_embeddings),
        LLMContextPrecisionWithReference(llm=judge),
        LLMContextRecall(llm=judge),
    ]

    logger.info("Evaluating with Ragas (judge model: %s)...", config.JUDGE_MODEL)
    return evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=judge,
        embeddings=judge_embeddings,
    )
