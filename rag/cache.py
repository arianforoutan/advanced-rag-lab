"""Persistent Semantic Cache using FAISS and SQLite for Insurellm RAG."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
import faiss
import numpy as np


logger = logging.getLogger(__name__)


class PersistentSemanticCache:
    def __init__(
        self ,
        embeddings_model,
        cache_dir: Path,
        dimension: int = 384, 
        similarity_threshold: float = 0.90,
    ):

        self.embeddings = embeddings_model
        self.threshold = similarity_threshold
        self.dimension = dimension

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = self.cache_dir / "semantic_cache.db"
        self.index_path = self.cache_dir / "faiss.index"

        self._init_sqlite()
        self._init_faiss()

    def _init_sqlite(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_records (
                    id INTEGER PRIMARY KEY,
                    query TEXT NOT NULL,
                    response TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            conn.commit()


    def _init_faiss(self) -> None:
        if self.index_path.exists():
            logger.info("Loading existing FAISS cache index from %s", self.index_path)
            self.index = faiss.read_index(str(self.index_path))
        else:
            logger.info("Creating new FAISS Inner-Product index (Dimension: %d)", self.dimension)
            self.index = faiss.IndexFlatIP(self.dimension)


    def _save_index(self) -> None:
        faiss.write_index(self.index, str(self.index_path))

    def _embed_and_normalize(self, text: str) -> np.ndarray:
        vec = np.array(self.embeddings.embed_query(text), dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(vec)
        return vec


    def get(self, query: str) -> str | None:
        if self.index.ntotal == 0:
            return None

        q_vec = self._embed_and_normalize(query)
        similarities, indices = self.index.search(q_vec, 1)

        best_score = float(similarities[0][0])
        best_idx = int(indices[0][0])

        if best_idx != -1 and best_score >= self.threshold:
            row_id = best_idx + 1
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT query, response FROM cache_records WHERE id = ?", (row_id,)
                )
                row = cursor.fetchone()

            if row:
                matched_query, cached_response = row
                logger.info(
                    "⚡ Semantic Cache HIT! (Score: %.4f) | Matched: '%s'",
                    best_score,
                    matched_query,
                )
                return cached_response

        logger.info("Semantic Cache MISS (Best score: %.4f)", best_score if best_idx != -1 else 0.0)
        return None

    def put(self, query: str, response: str) -> None:
        q_vec = self._embed_and_normalize(query)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO cache_records (query, response) VALUES (?, ?)",
                (query, response),
            )
            conn.commit()

        self.index.add(q_vec)
        self._save_index()
        logger.info("Cached query successfully saved to persistent disk.")