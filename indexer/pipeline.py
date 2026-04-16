# indexer/pipeline.py

import hashlib
import os

from evaluation.dataset_loader import DatasetLoader
from indexer.crawler import Crawler
from indexer.extractor import Extractor
from indexer.chunker import Chunker
from indexer.embedder import Embedder
from indexer.store import Store


class IndexingPipeline:
    """
    Wires all indexer modules together.
    
    The flow for each file:
        Crawler (discover + hash check)
            → Extractor (file → raw text)
                → Chunker (text → chunks with metadata)
                    → Embedder (chunks → vectors)
                        → Store (vectors → FAISS, metadata → SQLite)
    """

    def __init__(self, config_path="config.yaml"):
        """
        Initialize all pipeline components.
        """
        self.config_path = config_path
        self.crawler = Crawler(config_path)
        self.extractor = Extractor()
        self.chunker = Chunker(chunk_size=500, overlap=50)
        self.embedder = Embedder(config_path)
        self.store = Store(config_path)

    def _iter_dataset_documents(self):
        """
        Yield BEIR corpus documents as synthetic files so hosted deployments
        can build an index from dataset folders containing corpus.jsonl.
        """
        for dataset_path in self.crawler.watch_paths:
            corpus_path = os.path.join(dataset_path, "corpus.jsonl")
            if not os.path.exists(corpus_path):
                continue

            dataset_name = os.path.basename(os.path.normpath(dataset_path))

            try:
                corpus = DatasetLoader(dataset_path).load_corpus()
            except Exception as e:
                print(f"[Pipeline] Could not load dataset corpus from {dataset_path}: {e}")
                continue

            for doc_id, doc in corpus.items():
                title = (doc.get("title") or "").strip()
                body = (doc.get("text") or "").strip()
                text = "\n\n".join(part for part in [title, body] if part).strip()
                if not text:
                    continue

                synthetic_path = f"{dataset_name}://{doc_id}"
                synthetic_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                yield synthetic_path, synthetic_hash, text

    def run(self):
        """
        Execute the full indexing pipeline.
        """
        known_hashes = self.store.load_hashes()
        print("Scanning for new/modified files...")
        files_to_process, current_hashes, deleted_files = self.crawler.get_new_and_modified(known_hashes)

        dataset_documents = list(self._iter_dataset_documents())
        known_dataset_hashes = {
            filepath: file_hash
            for filepath, file_hash in known_hashes.items()
            if "://" in filepath
        }

        for filepath, file_hash, text in dataset_documents:
            current_hashes[filepath] = file_hash
            if known_dataset_hashes.get(filepath) != file_hash:
                files_to_process.append((filepath, text))

        current_dataset_paths = {filepath for filepath, _, _ in dataset_documents}
        deleted_files = set(deleted_files) | (
            set(known_dataset_hashes.keys()) - current_dataset_paths
        )

        for filepath in deleted_files:
            self.store.remove_file_chunks(filepath)

        if not files_to_process:
            print("Index is up to date.")
            print(f"Total vectors: {self.store.get_total_vectors()}")
            return

        total = len(files_to_process)
        for i, item in enumerate(files_to_process, 1):
            if isinstance(item, tuple):
                filepath, text = item
            else:
                filepath = item
                text = self.extractor.extract(filepath)

            print(f"[{i}/{total}] {filepath}")
            if not text.strip():
                print(f"  Skipping (no text extracted)")
                continue
            chunks = self.chunker.chunk_file(text, filepath)
            chunk_texts = [c["text"] for c in chunks]
            embeddings = self.embedder.embed_chunks(chunk_texts)
            self.store.remove_file_chunks(filepath)
            self.store.add_chunks(chunks, embeddings)
            self.store.save_file_info(filepath, current_hashes[filepath], len(chunks))

        print(f"\nProcessed {total} files.")
        print(f"Total vectors: {self.store.get_total_vectors()}")


# --- Test it ---
if __name__ == "__main__":
    pipeline = IndexingPipeline()
    pipeline.run()
