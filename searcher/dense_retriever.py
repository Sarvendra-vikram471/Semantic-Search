# searcher/dense_retriever.py

import numpy as np
import faiss
import sqlite3
import yaml
from indexer.embedder import Embedder


class DenseRetriever:
    """
    Embeds the (expanded) query and searches the FAISS index for
    nearest-neighbor chunks by semantic similarity.

    This is the core semantic search — finds chunks that MEAN the same
    thing as the query, even if they share no keywords.
    """

    def __init__(self, config_path="config.yaml"):
        with open(config_path) as f:
            config = yaml.safe_load(f)

        self.data_dir = config["data_dir"]
        self.faiss_path = f"{self.data_dir}/index.faiss"
        self.db_path = f"{self.data_dir}/metadata.db"
        self.top_k = config.get("top_k", 20)  # fetch more than needed; reranker will trim

        self.embedder = Embedder(config_path)
        self.index = faiss.read_index(self.faiss_path)

    def retrieve(self, query: str, top_k: int = None) -> list[dict]:
        """
        Embed the query and search FAISS for the closest chunk vectors.

        Args:
            query (str) — expanded query string
            top_k (int) — number of results

        Returns:
            list[dict] each containing:
                chunk_id, chunk_text, filepath, chunk_index,
                dense_score (float, lower = more similar in L2 space)
        """
        k = top_k or self.top_k

        query_vec = self.embedder.embed_single(query)
        query_vec = np.array([query_vec], dtype="float32")

        distances, ids = self.index.search(query_vec, k)

        results = []
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for dist, chunk_id in zip(distances[0], ids[0]):
            if chunk_id == -1:
                continue
            row = cursor.execute(
                "SELECT chunk_text, filepath, chunk_index FROM chunks WHERE id = ?",
                (int(chunk_id),)
            ).fetchone()
            if row:
                results.append({
                    "chunk_id": int(chunk_id),
                    "chunk_text": row[0],
                    "filepath": row[1],
                    "chunk_index": row[2],
                    "dense_score": float(dist),
                })

        conn.close()
        return results


if __name__ == "__main__":
    dr = DenseRetriever()
    results = dr.retrieve("quarterly budget report", top_k=5)
    for r in results:
        print(f"[{r['dense_score']:.4f}] {r['filepath']} → {r['chunk_text'][:80]}")