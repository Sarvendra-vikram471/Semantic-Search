# evaluation/query_runner.py

from searcher.search_engine import SearchEngine


class QueryRunner:
    """
    Runs all evaluation queries through your SearchEngine and collects
    the ranked result lists for scoring.

    The results are formatted exactly as the Evaluator expects:
        {query_id: [(doc_id, score), ...]}   ranked best-first
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.engine = SearchEngine(config_path)

    def _extract_doc_id(self, filepath: str) -> str:
        """
        Convert "scifact://12345" back to doc_id "12345".
        """
        return filepath.replace("scifact://", "")

    def run(
        self,
        queries: dict,
        top_k: int = 100,
        mode: str = "full",
    ) -> dict:
        """
        Run all queries and return ranked results.

        Args:
            queries — {query_id: query_text}
            top_k   — number of results per query (use 100 for eval)
            mode    — pipeline variant to test:
                        "dense"   → dense retrieval only (no BM25, no reranker)
                        "sparse"  → BM25 only
                        "hybrid"  → dense + BM25 + RRF (no reranker)
                        "full"    → complete pipeline with reranker

        Returns:
            dict — {query_id: [(doc_id, rank_score), ...]}
                   rank_score is rrf_score or rerank_score depending on mode
        """
        results = {}
        total = len(queries)

        for i, (query_id, query_text) in enumerate(queries.items(), 1):
            if i % 50 == 0:
                print(f"  Running query {i}/{total}...")

            try:
                if mode == "dense":
                    raw = self.engine.dense_retriever.retrieve(query_text, top_k=top_k)
                    ranked = [
                        (self._extract_doc_id(r["filepath"]), -r["dense_score"])
                        for r in raw
                    ]

                elif mode == "sparse":
                    raw = self.engine.sparse_retriever.retrieve(query_text, top_k=top_k)
                    ranked = [
                        (self._extract_doc_id(r["filepath"]), r["sparse_score"])
                        for r in raw
                    ]

                elif mode == "hybrid":
                    dense_raw  = self.engine.dense_retriever.retrieve(query_text, top_k=top_k)
                    sparse_raw = self.engine.sparse_retriever.retrieve(query_text, top_k=top_k)
                    fused      = self.engine.fusion_ranker.fuse(dense_raw, sparse_raw, top_k=top_k)
                    ranked = [
                        (self._extract_doc_id(r["filepath"]), r["rrf_score"])
                        for r in fused
                    ]

                else:  # full pipeline
                    output = self.engine.search(query_text, top_k=top_k)
                    ranked = [
                        (self._extract_doc_id(r["filepath"]), r.get("rerank_score", r.get("rrf_score", 0)))
                        for r in output["results"]
                    ]

                # Deduplicate by doc_id (multiple chunks per doc → keep best score)
                seen = {}
                for doc_id, score in ranked:
                    if doc_id not in seen or score > seen[doc_id]:
                        seen[doc_id] = score

                results[query_id] = sorted(seen.items(), key=lambda x: x[1], reverse=True)

            except Exception as e:
                print(f"  Error on query {query_id}: {e}")
                results[query_id] = []

        return results


if __name__ == "__main__":
    from evaluation.dataset_loader import DatasetLoader

    loader  = DatasetLoader("data/scifact")
    queries = loader.load_queries()

    runner  = QueryRunner()
    results = runner.run(queries, top_k=10, mode="full")

    sample_qid = list(results.keys())[0]
    print(f"\nQuery {sample_qid} top results:")
    for doc_id, score in results[sample_qid][:5]:
        print(f"  doc {doc_id}  score={score:.4f}")