# searcher/fusion_ranker.py


class FusionRanker:
    """
    Reciprocal Rank Fusion (RRF) merges dense and sparse result lists
    into a single ranked list without needing to normalise scores.

    Why RRF instead of score averaging?
    - Dense scores (L2 distance) and BM25 scores live on different scales
    - Normalising them requires knowing min/max, which varies per query
    - RRF uses only the RANK of each result, making it scale-invariant

    RRF formula:
        score(d) = Σ 1 / (k + rank(d, list_i))
        where k=60 is a constant that dampens the effect of very high ranks
    """

    def __init__(self, k: int = 60):
        self.k = k

    def fuse(
        self,
        dense_results: list[dict],
        sparse_results: list[dict],
        top_k: int = 10,
    ) -> list[dict]:
        """
        Fuse two ranked result lists using Reciprocal Rank Fusion.

        Args:
            dense_results  — from DenseRetriever, ordered by dense_score ascending
                             (lower L2 distance = better)
            sparse_results — from SparseRetriever, ordered by sparse_score descending
            top_k          — number of results to return after fusion

        Returns:
            list[dict] — merged results sorted by rrf_score descending,
                         each dict includes all original fields + rrf_score
        """
        rrf_scores = {}
        chunk_data = {}

        # Dense list: lower distance = better rank = higher RRF contribution
        for rank, result in enumerate(dense_results):
            cid = result["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (self.k + rank + 1)
            chunk_data[cid] = result

        # Sparse list: higher BM25 score = better rank = higher RRF contribution
        for rank, result in enumerate(sparse_results):
            cid = result["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (self.k + rank + 1)
            if cid not in chunk_data:
                chunk_data[cid] = result

        sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)[:top_k]

        fused = []
        for cid in sorted_ids:
            entry = dict(chunk_data[cid])
            entry["rrf_score"] = rrf_scores[cid]
            fused.append(entry)

        return fused


if __name__ == "__main__":
    dense = [
        {"chunk_id": 1, "chunk_text": "budget summary Q3", "filepath": "/a.pdf", "chunk_index": 0, "dense_score": 0.12},
        {"chunk_id": 2, "chunk_text": "revenue report",     "filepath": "/b.pdf", "chunk_index": 0, "dense_score": 0.25},
    ]
    sparse = [
        {"chunk_id": 2, "chunk_text": "revenue report",    "filepath": "/b.pdf", "chunk_index": 0, "sparse_score": 8.1},
        {"chunk_id": 3, "chunk_text": "financial overview", "filepath": "/c.pdf", "chunk_index": 0, "sparse_score": 5.2},
    ]
    ranker = FusionRanker()
    results = ranker.fuse(dense, sparse, top_k=5)
    for r in results:
        print(f"[RRF {r['rrf_score']:.5f}] {r['filepath']} → {r['chunk_text'][:60]}")