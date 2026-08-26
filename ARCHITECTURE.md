# معماری پروژهٔ Insurellm — Graph + Hybrid RAG

```mermaid
flowchart TB
    subgraph KB["knowledge-base/ (76 md files)"]
        KB1[company] ~~~ KB2[products] ~~~ KB3[contracts] ~~~ KB4[employees]
    end

    subgraph BUILD["Build-time: ingestion و index سازی"]
        direction TB
        LOAD["load_documents<br/>doc_type = نام پوشه"]
        SPLIT["split_documents<br/>Recursive 1000/200 → 413 chunk"]
        LOAD --> SPLIT
        SPLIT --> CHROMA[("Chroma vector store<br/>all-MiniLM-L6-v2<br/>persist: ch_database/")]
        SPLIT --> BM25["BM25Index<br/>(in-memory)"]
        SPLIT --> PUB["filter_public_chunks<br/>(doc_type مجاز)"]
        PUB --> EXTRACT["GraphExtractor (LLM)<br/>schema-constrained"]
        EXTRACT --> NEO[("Neo4j graph<br/>Entity / RELATION")]
    end
    KB --> LOAD

    subgraph QUERY["Query-time: RagPipeline.search_knowledge_base"]
        direction TB
        Q0["سؤال کاربر"] --> AGENT["Agent (create_agent)<br/>model = gpt-5-nano<br/>اول باید tool را صدا بزند"]
        AGENT -->|"search_knowledge_base(query)"| G1{"① Guardrail<br/>question_policy"}
        G1 -->|"sensitive / off-topic"| REFUSE["پاسخ آمادهٔ رد ↩"]
        G1 -->|pass| G2["② Graph search (اختیاری)<br/>LLM entities → Cypher"]
        G2 --> QE["③ Query expansion<br/>LLM → ۳ کوئری جایگزین"]
        QE --> HS["④ Hybrid search<br/>vector + BM25 → RRF<br/>top 10"]
        HS --> RR["⑤ Rerank<br/>bge-reranker-base → top 3"]
        RR --> MERGE["⑥ Merge: graph facts + text context"]
        MERGE -->|"context string"| AGENT
        AGENT --> ANS["پاسخ نهایی"]
    end

    CHROMA -.->|"retriever + doc_type filter"| HS
    BM25 -.-> HS
    NEO -.-> G2

    subgraph EVAL["scripts/evaluate.py — ارزیابی ۳ لایه"]
        E1["Retrieval<br/>Hit@k / MRR"] ~~~ E2["Generation<br/>Ragas (judge)"] ~~~ E3["Guardrails<br/>pass/fail"]
    end
    QUERY -.-> EVAL
```

## مدل‌ها
- **LLM اصلی**: `gpt-5-nano` از طریق GAPGPT (OpenAI-compatible) — هم برای پاسخ‌گویی agent، هم query expansion، هم entity extraction، هم graph extraction.
- **Embeddings**: `all-MiniLM-L6-v2` (لوکال، sentence-transformers).
- **Reranker**: `BAAI/bge-reranker-base` (cross-encoder لوکال).
- **Judge ارزیابی**: `JUDGE_MODEL` (پیش‌فرض = همان nano، قابل override).

## دو لایهٔ Guardrail
1. **ورودی** (`question_policy`): قبل از هر retrieval — کوئری sensitive یا off-topic را رد می‌کند.
2. **دسترسی سند** (`doc_type`): هم retriever برداری فیلتر `$in ALLOWED_DOC_TYPES` دارد، هم BM25 فقط روی `public_chunks` ساخته می‌شود.
