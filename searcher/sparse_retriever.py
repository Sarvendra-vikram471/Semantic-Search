# searcher/sparse_retriever.py

import sqlite3
import math
import yaml
from collections import defaultdict


class SparseRetriever:
    """
    BM25 (Okapi BM25) lexical retrieval over the SQLite chunk store.

    Why BM25 alongside semantic search?
    - Dense retrieval can miss exact keyword matches (product codes, names, IDs)
    - BM25 is great for rare/specific terms that embeddings smooth over
    - Hybrid = best of both worlds

    BM25 formula:
        score(q, d) = Σ IDF(t) × (tf × (k1+1)) / (tf + k1 × (1 - b + b × dl/avgdl))
    """

    def __init__(self, config_path="config.yaml"):
        with open(config_path) as f:
            config = yaml.safe_load(f)

        self.db_path = f"{config['data_dir']}/metadata.db"
        self.k1 = 1.5   # term frequency saturation
        self.b = 0.75   # length normalisation

        # Build in-memory BM25 index from SQLite on startup
        self._corpus = []       # list of (chunk_id, token_list)
        self._avgdl = 0.0
        self._df = defaultdict(int)   # term → doc frequency
        self._build_index()

    def _build_index(self):
        """Load all chunks from SQLite and compute BM25 statistics."""
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("SELECT id, chunk_text FROM chunks").fetchall()
        conn.close()

        total_len = 0
        for chunk_id, text in rows:
            tokens = text.lower().split()
            self._corpus.append((chunk_id, tokens))
            total_len += len(tokens)
            for token in set(tokens):
                self._df[token] += 1

        self._avgdl = total_len / len(rows) if rows else 1.0
        self._N = len(rows)

    def _idf(self, term: str) -> float:
        """Inverse document frequency for a term."""
        df = self._df.get(term, 0)
        return math.log((self._N - df + 0.5) / (df + 0.5) + 1)

    def retrieve(self, query: str, top_k: int = 20) -> list[dict]:
        """
        Run BM25 retrieval over the corpus.

        Args:
            query (str) — raw or rewritten query (NOT expanded — BM25 is lexical)
            top_k (int) — number of results to return

        Returns:
            list[dict] with chunk_id and sparse_score, sorted descending
        """
        query_terms = query.lower().split()
        scores = {}

        for chunk_id, tokens in self._corpus:
            dl = len(tokens)
            score = 0.0
            tf_map = defaultdict(int)
            for t in tokens:
                tf_map[t] += 1

            for term in query_terms:
                if term not in tf_map:
                    continue
                tf = tf_map[term]
                idf = self._idf(term)
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
                score += idf * numerator / denominator

            if score > 0:
                scores[chunk_id] = score

        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # Fetch text for top results
        conn = sqlite3.connect(self.db_path)
        results = []
        for chunk_id, score in sorted_results:
            row = conn.execute(
                "SELECT chunk_text, filepath, chunk_index FROM chunks WHERE id = ?",
                (chunk_id,)
            ).fetchone()
            if row:
                results.append({
                    "chunk_id": chunk_id,
                    "chunk_text": row[0],
                    "filepath": row[1],
                    "chunk_index": row[2],
                    "sparse_score": score,
                })
        conn.close()
        return results


if __name__ == "__main__":
    sr = SparseRetriever()
    results = sr.retrieve("quarterly budget", top_k=5)
    for r in results:
        print(f"[{r['sparse_score']:.4f}] {r['filepath']} → {r['chunk_text'][:80]}")