# evaluation/indexer_bridge.py

import numpy as np
from indexer.chunker import Chunker
from indexer.embedder import Embedder
from indexer.store import Store


class IndexerBridge:
    """
    Feeds the BEIR corpus directly into your existing indexing pipeline.

    The corpus documents are NOT real files on disk — they come from JSONL.
    So we bypass the Crawler/Extractor and inject text directly into
    Chunker → Embedder → Store.

    Each document gets a fake filepath: "scifact://{doc_id}"
    This lets the Store treat them like any other indexed file,
    and the Evaluator can later match doc_id back from results.
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.chunker = Chunker(chunk_size=500, overlap=50)
        self.embedder = Embedder(config_path)
        self.store    = Store(config_path)

    def index_corpus(self, corpus: dict, batch_size: int = 64):
        """
        Index the entire corpus into FAISS + SQLite.

        Args:
            corpus     — {doc_id: {"title": str, "text": str}}
            batch_size — number of chunks to embed at once (memory control)
        """
        doc_ids = list(corpus.keys())
        total   = len(doc_ids)
        print(f"Indexing {total} documents...")

        # Clear any previous SciFact index entries
        # (safe to run multiple times)
        existing_hashes = self.store.load_hashes()
        scifact_files   = [fp for fp in existing_hashes if fp.startswith("scifact://")]
        for fp in scifact_files:
            self.store.remove_file_chunks(fp)
        if scifact_files:
            print(f"Cleared {len(scifact_files)} previously indexed documents")

        chunk_buffer   = []   # list of chunk dicts
        text_buffer    = []   # list of chunk texts for batch embedding

        def flush(chunk_buffer, text_buffer):
            if not chunk_buffer:
                return
            embeddings = self.embedder.embed_chunks(text_buffer)
            embeddings = np.array(embeddings, dtype="float32")
            self.store.add_chunks(chunk_buffer, embeddings)

        for i, doc_id in enumerate(doc_ids, 1):
            doc = corpus[doc_id]
            # Combine title and body — title is often the key claim
            full_text = f"{doc['title']} {doc['text']}".strip()
            if not full_text:
                continue

            # Use scifact://doc_id as the fake filepath
            fake_path = f"scifact://{doc_id}"
            chunks    = self.chunker.chunk_file(full_text, fake_path)

            for chunk in chunks:
                chunk_buffer.append(chunk)
                text_buffer.append(chunk["text"])

            # Save file-level info (hash = doc_id for reproducibility)
            self.store.save_file_info(fake_path, doc_id, len(chunks))

            # Flush every batch_size documents
            if len(chunk_buffer) >= batch_size:
                flush(chunk_buffer, text_buffer)
                chunk_buffer.clear()
                text_buffer.clear()

            if i % 500 == 0:
                print(f"  Indexed {i}/{total}...")

        # Flush remainder
        flush(chunk_buffer, text_buffer)
        print(f"Done. Total vectors: {self.store.get_total_vectors()}")


if __name__ == "__main__":
    from evaluation.dataset_loader import DatasetLoader

    loader = DatasetLoader("data/scifact")
    corpus = loader.load_corpus()

    bridge = IndexerBridge()
    bridge.index_corpus(corpus, batch_size=64)