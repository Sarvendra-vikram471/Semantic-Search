# searcher/highlighter.py


class Highlighter:
    """
    Extracts the most relevant passage from a chunk and highlights
    query-matching terms for display in search results.

    Why highlight?
    - The full chunk may be 500 words; the user needs a ~2-sentence preview
    - Highlighting query terms helps users quickly judge relevance
    - This is purely display logic — does not affect ranking
    """

    def __init__(self, preview_words: int = 30):
        self.preview_words = preview_words

    def extract_preview(self, chunk_text: str, query: str) -> str:
        """
        Find the sentence in chunk_text most relevant to the query
        and return a short preview around it.

        Strategy: find the window of preview_words words that contains
        the most query term matches.

        Args:
            chunk_text  — full text of the chunk
            query       — original user query

        Returns:
            str — the best preview snippet
        """
        words = chunk_text.split()
        if len(words) <= self.preview_words:
            return chunk_text

        query_terms = set(query.lower().split())
        best_score = -1
        best_start = 0

        for i in range(len(words) - self.preview_words + 1):
            window = words[i: i + self.preview_words]
            score = sum(1 for w in window if w.lower().strip(".,;:") in query_terms)
            if score > best_score:
                best_score = score
                best_start = i

        snippet = " ".join(words[best_start: best_start + self.preview_words])

        # Add ellipsis if truncated
        if best_start > 0:
            snippet = "..." + snippet
        if best_start + self.preview_words < len(words):
            snippet = snippet + "..."

        return snippet

    def highlight_html(self, text: str, query: str) -> str:
        """
        Wrap query-matching words in <mark> tags for HTML display.

        Args:
            text  — preview snippet
            query — original user query

        Returns:
            str — HTML string with <mark> tags around matching words
        """
        query_terms = set(query.lower().split())
        highlighted_words = []

        for word in text.split():
            clean = word.lower().strip(".,;:!?")
            if clean in query_terms:
                highlighted_words.append(f"<mark>{word}</mark>")
            else:
                highlighted_words.append(word)

        return " ".join(highlighted_words)

    def annotate(self, results: list[dict], query: str) -> list[dict]:
        """
        Add preview and highlighted_preview to each result dict.

        Args:
            results — list of chunk dicts
            query   — original user query

        Returns:
            list[dict] — same results with 'preview' and 'preview_html' added
        """
        for result in results:
            preview = self.extract_preview(result["chunk_text"], query)
            result["preview"] = preview
            result["preview_html"] = self.highlight_html(preview, query)
        return results