# searcher/reranker.py

import yaml
from sentence_transformers import CrossEncoder


class Reranker:
    """
    Cross-encoder reranking for precision improvement on top-k candidates.

    Why rerank after fusion?
    - Bi-encoders (used in DenseRetriever) embed query and doc INDEPENDENTLY
      — fast but they can't model their interaction
    - Cross-encoders process (query, document) TOGETHER — much more accurate
      but too slow to run over the whole corpus
    - Solution: use bi-encoder to shortlist ~20 candidates, then cross-encoder
      to precisely rank them → speed of FAISS + accuracy of cross-encoder

    Model: ms-marco-MiniLM-L-6-v2 (fast, strong on passage relevance)
    """

    def __init__(self, config_path="config.yaml"):
        with open(config_path) as f:
            config = yaml.safe_load(f)

        model_name = config.get("reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.enabled = config.get("reranking_enabled", True)
        if self.enabled:
            print(f"Loading reranker model '{model_name}'...")
            self.model = CrossEncoder(model_name)
            print("Reranker loaded.")

    def rerank(self, query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
        """
        Rerank candidates using cross-encoder relevance scores.

        Args:
            query      — original (not expanded) user query
            candidates — list of chunk dicts from FusionRanker
            top_k      — number of final results to return

        Returns:
            list[dict] — reranked results, each with rerank_score added
        """
        if not self.enabled or not candidates:
            return candidates[:top_k]

        pairs = [(query, c["chunk_text"]) for c in candidates]
        scores = self.model.predict(pairs)

        for i, score in enumerate(scores):
            candidates[i]["rerank_score"] = float(score)

        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]


if __name__ == "__main__":
    reranker = Reranker()
    candidates = [
        {"chunk_id": 1, "chunk_text": "Q3 budget shows 15% revenue increase", "filepath": "/a.pdf", "chunk_index": 0, "rrf_score": 0.03},
        {"chunk_id": 2, "chunk_text": "The weather today is sunny and warm",   "filepath": "/b.txt", "chunk_index": 0, "rrf_score": 0.02},
    ]
    results = reranker.rerank("quarterly revenue", candidates, top_k=2)
    for r in results:
        print(f"[rerank {r['rerank_score']:.4f}] {r['chunk_text']}")