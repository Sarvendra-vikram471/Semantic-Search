# searcher/search_engine.py

import yaml
from searcher.query_understanding import QueryUnderstanding
from searcher.dense_retriever import DenseRetriever
from searcher.sparse_retriever import SparseRetriever
from searcher.fusion_ranker import FusionRanker
from searcher.reranker import Reranker
from searcher.facet_filter import FacetFilter
from searcher.highlighter import Highlighter


class SearchEngine:
    """
    Orchestrates the full search pipeline end-to-end:

        raw query
            → QueryUnderstanding  (expand + rewrite)
            → DenseRetriever      (semantic FAISS search)
            → SparseRetriever     (BM25 lexical search)
            → FusionRanker        (RRF merge)
            → Reranker            (cross-encoder precision)
            → FacetFilter         (type / date / size / directory)
            → Highlighter         (preview + HTML highlights)
            → final results
    """

    def __init__(self, config_path="config.yaml"):
        self.config_path = config_path
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.query_understanding = QueryUnderstanding(config_path)
        self.dense_retriever = DenseRetriever(config_path)
        self.sparse_retriever = SparseRetriever(config_path)
        self.fusion_ranker = FusionRanker(k=60)
        self.reranker = Reranker(config_path)
        self.facet_filter = FacetFilter()
        self.highlighter = Highlighter(preview_words=30)

        self.candidate_k = self.config.get("candidate_k", 20)
        self.final_k = self.config.get("top_k", 5)

    def search(
        self,
        query: str,
        top_k: int = None,
        file_type: list[str] = None,
        date_after=None,
        date_before=None,
        min_size: int = None,
        max_size: int = None,
        directory: str = None,
    ) -> dict:
        """
        Run the full search pipeline.

        Args:
            query       — natural language user query
            top_k       — number of final results (overrides config)
            file_type   — e.g. [".pdf", ".docx"]
            date_after  — datetime; exclude older files
            date_before — datetime; exclude newer files
            min_size    — min file size in bytes
            max_size    — max file size in bytes
            directory   — restrict to this directory

        Returns:
            dict:
                query_info  — dict from QueryUnderstanding
                results     — list of final result dicts, each with:
                                filepath, chunk_text, chunk_index,
                                preview, preview_html,
                                dense_score (if present),
                                sparse_score (if present),
                                rrf_score, rerank_score
        """
        k = top_k or self.final_k

        # Step 1 — query understanding
        query_info = self.query_understanding.process(query)

        # Step 2 — dense retrieval (uses expanded query for better semantic reach)
        dense_results = self.dense_retriever.retrieve(
            query_info["expanded"], top_k=self.candidate_k
        )

        # Step 3 — sparse retrieval (uses rewritten query; expansion hurts BM25)
        sparse_results = self.sparse_retriever.retrieve(
            query_info["rewritten"], top_k=self.candidate_k
        )

        # Step 4 — RRF fusion
        fused = self.fusion_ranker.fuse(dense_results, sparse_results, top_k=self.candidate_k)

        # Step 5 — cross-encoder reranking
        reranked = self.reranker.rerank(query_info["original"], fused, top_k=k * 2)

        # Step 6 — facet filtering
        filtered = self.facet_filter.filter(
            reranked,
            file_type=file_type,
            date_after=date_after,
            date_before=date_before,
            min_size=min_size,
            max_size=max_size,
            directory=directory,
        )

        # Trim to top_k after filtering
        final = filtered[:k]

        # Step 7 — highlight previews
        final = self.highlighter.annotate(final, query_info["original"])

        return {
            "query_info": query_info,
            "results": final,
        }


if __name__ == "__main__":
    engine = SearchEngine()
    output = engine.search("quarterly budget report", top_k=3)

    print(f"Query     : {output['query_info']['original']}")
    print(f"Expanded  : {output['query_info']['expanded']}")
    print(f"Results   : {len(output['results'])}\n")

    for i, r in enumerate(output["results"], 1):
        print(f"--- Result {i} ---")
        print(f"File     : {r['filepath']}")
        print(f"Preview  : {r['preview']}")
        print(f"RRF      : {r.get('rrf_score', 'n/a'):.5f}")
        print(f"Rerank   : {r.get('rerank_score', 'n/a'):.4f}")
        print()