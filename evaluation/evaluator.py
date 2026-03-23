# evaluation/evaluator.py

import math
from collections import defaultdict


class Evaluator:
    """
    Computes standard IR evaluation metrics by comparing your
    system's ranked results against the ground-truth qrels.

    Metrics implemented:
        NDCG@k  — Normalized Discounted Cumulative Gain
                  Measures ranking quality; rewards relevant docs appearing early
        MAP@k   — Mean Average Precision
                  Average of precision computed at each relevant doc position
        Recall@k — Fraction of relevant docs found in top-k
        P@k      — Precision at k (fraction of top-k that are relevant)
        MRR      — Mean Reciprocal Rank (position of first relevant result)
    """

    def ndcg_at_k(self, ranked: list, relevant: dict, k: int) -> float:
        """
        NDCG@k — the most important metric for ranked retrieval.
        Score of 1.0 = perfect ranking, 0.0 = no relevant docs found.
        """
        dcg = 0.0
        for i, (doc_id, _) in enumerate(ranked[:k]):
            rel = relevant.get(doc_id, 0)
            if rel > 0:
                dcg += rel / math.log2(i + 2)   # i+2 because log2(1)=0

        # Ideal DCG — best possible ranking
        ideal_rels = sorted(relevant.values(), reverse=True)[:k]
        idcg = sum(
            rel / math.log2(i + 2)
            for i, rel in enumerate(ideal_rels)
            if rel > 0
        )

        return dcg / idcg if idcg > 0 else 0.0

    def map_at_k(self, ranked: list, relevant: dict, k: int) -> float:
        """
        MAP@k — average precision across all relevant document positions.
        """
        num_relevant = 0
        sum_precision = 0.0

        for i, (doc_id, _) in enumerate(ranked[:k]):
            if relevant.get(doc_id, 0) > 0:
                num_relevant += 1
                precision_at_i = num_relevant / (i + 1)
                sum_precision += precision_at_i

        total_relevant = sum(1 for v in relevant.values() if v > 0)
        if total_relevant == 0:
            return 0.0
        return sum_precision / total_relevant

    def recall_at_k(self, ranked: list, relevant: dict, k: int) -> float:
        """
        Recall@k — what fraction of all relevant docs appear in top-k.
        """
        total_relevant = sum(1 for v in relevant.values() if v > 0)
        if total_relevant == 0:
            return 0.0
        found = sum(
            1 for doc_id, _ in ranked[:k]
            if relevant.get(doc_id, 0) > 0
        )
        return found / total_relevant

    def precision_at_k(self, ranked: list, relevant: dict, k: int) -> float:
        """
        P@k — fraction of the top-k results that are relevant.
        """
        if k == 0:
            return 0.0
        hits = sum(
            1 for doc_id, _ in ranked[:k]
            if relevant.get(doc_id, 0) > 0
        )
        return hits / k

    def mrr(self, ranked: list, relevant: dict) -> float:
        """
        MRR — reciprocal of the rank of the first relevant result.
        Score of 1.0 = first result is relevant.
        """
        for i, (doc_id, _) in enumerate(ranked):
            if relevant.get(doc_id, 0) > 0:
                return 1.0 / (i + 1)
        return 0.0

    def evaluate(
        self,
        all_results: dict,
        qrels: dict,
        k_values: list = None,
    ) -> dict:
        """
        Compute all metrics across all queries and average them.

        Args:
            all_results — {query_id: [(doc_id, score), ...]}  from QueryRunner
            qrels       — {query_id: {doc_id: relevance}}     from DatasetLoader
            k_values    — list of k values to evaluate at, e.g. [1, 5, 10, 100]

        Returns:
            dict — {
                "NDCG@10": 0.42,
                "MAP@100": 0.38,
                "Recall@100": 0.71,
                "P@10": 0.15,
                "MRR": 0.55,
                "num_queries": 300,
                "queries_with_results": 298,
            }
        """
        if k_values is None:
            k_values = [1, 5, 10, 100]

        # Accumulate per-query scores
        scores = defaultdict(list)

        num_queries         = 0
        queries_with_results = 0

        for query_id, ranked in all_results.items():
            relevant = qrels.get(query_id, {})
            if not relevant:
                continue   # skip queries with no ground truth

            num_queries += 1
            if ranked:
                queries_with_results += 1

            for k in k_values:
                scores[f"NDCG@{k}"].append(self.ndcg_at_k(ranked, relevant, k))
                scores[f"MAP@{k}"].append(self.map_at_k(ranked, relevant, k))
                scores[f"Recall@{k}"].append(self.recall_at_k(ranked, relevant, k))
                scores[f"P@{k}"].append(self.precision_at_k(ranked, relevant, k))

            scores["MRR"].append(self.mrr(ranked, relevant))

        # Average across queries
        summary = {
            metric: round(sum(vals) / len(vals), 4) if vals else 0.0
            for metric, vals in scores.items()
        }
        summary["num_queries"]          = num_queries
        summary["queries_with_results"] = queries_with_results

        return summary