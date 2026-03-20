# searcher/query_understanding.py

import yaml
from nltk.corpus import wordnet
import nltk

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)


class QueryUnderstanding:
    """
    Expands and rewrites the raw user query before retrieval.

    Two strategies:
    1. WordNet synonym expansion — "apples" → ["apple", "malus", "orchard", "fruit"]
    2. (Optional) T5 query rewriting — rephrase for better embedding alignment

    Why expand?
    - A search for "apples" should also find "fruit", "orchard", "nutrition"
    - Embedding models are good but can miss synonyms in different domains
    - Expansion bridges the vocabulary gap before we even hit FAISS
    """

    def __init__(self, config_path="config.yaml"):
        with open(config_path) as f:
            config = yaml.safe_load(f)
        self.max_synonyms = config.get("max_synonyms", 5)
        self.expansion_enabled = config.get("query_expansion", True)

    def expand(self, query: str) -> str:
        """
        Expand the query using WordNet synonyms.

        Args:
            query (str) — raw user query

        Returns:
            str — expanded query string (original + synonyms appended)

        Example:
            "apple nutrition" → "apple nutrition fruit malus orchard dietary"
        """
        if not self.expansion_enabled:
            return query

        words = query.lower().split()
        synonyms = set()

        for word in words:
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    cleaned = lemma.name().replace("_", " ")
                    if cleaned.lower() != word:
                        synonyms.add(cleaned)
                if len(synonyms) >= self.max_synonyms:
                    break

        expanded = query
        if synonyms:
            expanded = query + " " + " ".join(list(synonyms)[:self.max_synonyms])

        return expanded

    def rewrite(self, query: str) -> str:
        """
        Lightweight query rewriting — normalise and clean the query.
        (Plug in a T5/FLAN-T5 model here for more powerful rewriting.)

        Args:
            query (str) — raw user query

        Returns:
            str — cleaned query
        """
        # Basic normalisation: strip extra spaces, lowercase
        query = " ".join(query.strip().split())
        return query

    def process(self, query: str) -> dict:
        """
        Full query understanding pipeline.

        Returns:
            dict with:
                original   — the raw input
                rewritten  — cleaned version
                expanded   — synonym-expanded version for dense retrieval
        """
        rewritten = self.rewrite(query)
        expanded = self.expand(rewritten)
        return {
            "original": query,
            "rewritten": rewritten,
            "expanded": expanded,
        }


if __name__ == "__main__":
    qu = QueryUnderstanding()
    result = qu.process("quarterly budget report")
    print(f"Original : {result['original']}")
    print(f"Rewritten: {result['rewritten']}")
    print(f"Expanded : {result['expanded']}")