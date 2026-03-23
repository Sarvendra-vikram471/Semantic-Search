# evaluation/dataset_loader.py

import json
import csv
import os


class DatasetLoader:
    """
    Loads the SciFact dataset from the BEIR format.

    BEIR format (same across all BEIR datasets):
        corpus.jsonl  — one JSON per line: {_id, title, text}
        queries.jsonl — one JSON per line: {_id, text}
        qrels/test.tsv — TSV: query_id  doc_id  relevance_score (0 or 1)

    SciFact specifics:
        - 5,183 scientific claim documents
        - 300 test queries (scientific claims)
        - Relevance: 1 = document supports/refutes the claim
    """

    def __init__(self, dataset_path: str):
        """
        Args:
            dataset_path — path to the scifact/ folder
                           e.g. "data/scifact"
        """
        self.dataset_path = dataset_path
        self.corpus_path  = os.path.join(dataset_path, "corpus.jsonl")
        self.queries_path = os.path.join(dataset_path, "queries.jsonl")
        self.qrels_path   = os.path.join(dataset_path, "qrels", "test.tsv")

    def load_corpus(self) -> dict:
        """
        Load all documents from corpus.jsonl.

        Returns:
            dict — {doc_id: {"title": str, "text": str}}
        """
        corpus = {}
        with open(self.corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                doc = json.loads(line.strip())
                doc_id = str(doc["_id"])
                # Combine title + text — same as how a real document would look
                corpus[doc_id] = {
                    "title": doc.get("title", ""),
                    "text":  doc.get("text", ""),
                }
        print(f"Loaded {len(corpus)} documents from corpus")
        return corpus

    def load_queries(self) -> dict:
        """
        Load test queries from queries.jsonl.

        Returns:
            dict — {query_id: query_text}
        """
        queries = {}
        with open(self.queries_path, "r", encoding="utf-8") as f:
            for line in f:
                q = json.loads(line.strip())
                queries[str(q["_id"])] = q["text"]
        print(f"Loaded {len(queries)} queries")
        return queries

    def load_qrels(self) -> dict:
        """
        Load relevance judgments from qrels/test.tsv.

        Returns:
            dict — {query_id: {doc_id: relevance_score}}
            relevance_score is 1 (relevant) or 0 (not relevant)
        """
        qrels = {}
        with open(self.qrels_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader)  # skip header: query-id  corpus-id  score
            for row in reader:
                if len(row) < 3:
                    continue
                query_id, doc_id, score = str(row[0]), str(row[1]), int(row[2])
                if query_id not in qrels:
                    qrels[query_id] = {}
                qrels[query_id][doc_id] = score
        print(f"Loaded qrels for {len(qrels)} queries")
        return qrels


if __name__ == "__main__":
    loader = DatasetLoader("data/scifact")
    corpus  = loader.load_corpus()
    queries = loader.load_queries()
    qrels   = loader.load_qrels()

    # Show a sample
    sample_qid = list(queries.keys())[0]
    print(f"\nSample query [{sample_qid}]: {queries[sample_qid]}")
    print(f"Relevant docs: {qrels.get(sample_qid, {})}")