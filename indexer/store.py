# indexer/store.py

import os
import sqlite3
import numpy as np
import faiss
import yaml


class Store:
    """
    Handles two storage systems:

    1. FAISS — stores dense vectors for fast similarity search
               Uses IndexHNSWFlat instead of IndexFlatL2
               HNSW = Hierarchical Navigable Small World graph
               - IndexFlatL2  : scans every vector (slow at scale)
               - IndexHNSWFlat: graph-based navigation (fast, same accuracy)

    2. SQLite — stores metadata about each chunk
    """

    # HNSW parameter — higher = more accurate but more memory
    # 32 is the standard default, good balance for this use case
    HNSW_M = 32

    def __init__(self, config_path="config.yaml"):
        """
        Load config, set up file paths, initialize FAISS index and SQLite.
        """
        config_path = os.path.abspath(config_path)
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        config_dir = os.path.dirname(config_path)
        data_dir = config["data_dir"]
        self.data_dir = data_dir if os.path.isabs(data_dir) else os.path.normpath(os.path.join(config_dir, data_dir))
        os.makedirs(self.data_dir, exist_ok=True)

        self.faiss_path = os.path.join(self.data_dir, "index.faiss")
        self.db_path    = os.path.join(self.data_dir, "metadata.db")

        self._init_db()
        self._load_or_create_index()

    def _init_db(self):
        """
        Create SQLite tables if they don't already exist.
        """
        conn   = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chunks (
                id          INTEGER PRIMARY KEY,
                filepath    TEXT    NOT NULL,
                chunk_text  TEXT    NOT NULL,
                chunk_index INTEGER,
                FOREIGN KEY (filepath) REFERENCES files(filepath)
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS files (
                filepath     TEXT PRIMARY KEY,
                file_hash    TEXT NOT NULL,
                total_chunks INTEGER
            )
        ''')

        conn.commit()
        conn.close()

    def _load_or_create_index(self):
        """
        Load an existing FAISS index from disk, or set to None.
        The actual index is created on first add_chunks() call
        so we know the embedding dimension at that point.
        """
        if os.path.exists(self.faiss_path):
            self.index = faiss.read_index(self.faiss_path)
            print(f"[Store] Loaded FAISS index — {self.index.ntotal} vectors")
        else:
            self.index = None
            print("[Store] No existing index found — will create on first insert")

    def _create_hnsw_index(self, dimension: int):
        """
        Create a new HNSW-based FAISS index.

        Why HNSW over FlatL2:
            FlatL2   — exact search, O(n) per query, slow at scale
            HNSWFlat — approximate search, O(log n) per query, same accuracy
                       for top-k retrieval tasks

        IndexIDMap2 wraps HNSW to support custom integer IDs and deletion.

        Args:
            dimension — embedding size (384 for MiniLM and BGE-small)
        """
        hnsw_index      = faiss.IndexHNSWFlat(dimension, self.HNSW_M)
        hnsw_index.hnsw.efSearch     = 64   # search quality — higher = better recall
        hnsw_index.hnsw.efConstruction = 64 # build quality  — higher = better graph
        self.index      = faiss.IndexIDMap2(hnsw_index)
        print(f"[Store] Created HNSW index — dim={dimension}, M={self.HNSW_M}")

    def get_next_id(self):
        """
        Get the next available chunk ID from SQLite.
        """
        conn   = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(id) FROM chunks")
        result = cursor.fetchone()[0]
        conn.close()
        return 0 if result is None else result + 1

    def add_chunks(self, chunks_with_metadata, embeddings):
        """
        Add new chunks and their embeddings to both FAISS and SQLite.

        Args:
            chunks_with_metadata (list[dict]) — from chunker.chunk_file()
                Each dict has: text, filepath, chunk_index
            embeddings (numpy.ndarray) — shape (num_chunks, embedding_dim)
                From embedder.embed_chunks()
        """
        embeddings = embeddings.astype("float32")

        # create index on first insert — dimension comes from embeddings
        if self.index is None:
            dimension = embeddings.shape[1]
            self._create_hnsw_index(dimension)

        start_id = self.get_next_id()
        ids      = np.array(
            [start_id + i for i in range(len(chunks_with_metadata))],
            dtype=np.int64
        )

        self.index.add_with_ids(embeddings, ids)
        faiss.write_index(self.index, self.faiss_path)

        # save chunk metadata to SQLite
        conn   = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for i, chunk in enumerate(chunks_with_metadata):
            vector_id = start_id + i
            cursor.execute(
                "INSERT INTO chunks (id, filepath, chunk_text, chunk_index) "
                "VALUES (?, ?, ?, ?)",
                (vector_id, chunk["filepath"], chunk["text"], chunk["chunk_index"])
            )

        conn.commit()
        conn.close()

    def save_file_info(self, filepath, file_hash, total_chunks):
        """
        Save or update file info in SQLite.

        Args:
            filepath     — file path or fake path e.g. "scifact://12345"
            file_hash    — SHA256 hash or doc_id string
            total_chunks — number of chunks this file was split into
        """
        conn   = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO files (filepath, file_hash, total_chunks) "
            "VALUES (?, ?, ?)",
            (filepath, file_hash, total_chunks)
        )
        conn.commit()
        conn.close()

    def load_hashes(self):
        """
        Load all stored file hashes from SQLite.

        Returns:
            dict — {filepath: hash_string}
        """
        conn   = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT filepath, file_hash FROM files")
        rows   = cursor.fetchall()
        conn.close()
        return {row[0]: row[1] for row in rows}

    def remove_file_chunks(self, filepath):
        """
        Delete all chunks for a file from both SQLite and FAISS.

        Args:
            filepath — the filepath to remove
        """
        conn   = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        ids = cursor.execute(
            "SELECT id FROM chunks WHERE filepath = ?", (filepath,)
        ).fetchall()

        cursor.execute("DELETE FROM chunks WHERE filepath = ?", (filepath,))
        cursor.execute("DELETE FROM files  WHERE filepath = ?", (filepath,))
        conn.commit()
        conn.close()

        if ids and self.index is not None:
            id_array = np.array([i[0] for i in ids], dtype=np.int64)
            self.index.remove_ids(id_array)
            faiss.write_index(self.index, self.faiss_path)

    def get_total_vectors(self):
        """
        Return how many vectors are in the FAISS index.

        Returns:
            int — number of vectors, or 0 if index is empty
        """
        if self.index is None:
            return 0
        return self.index.ntotal


if __name__ == "__main__":
    store = Store()

    fake_chunks = [
        {"text": "quarterly budget report summary",       "filepath": "/docs/report.pdf",   "chunk_index": 0},
        {"text": "revenue increased by fifteen percent",  "filepath": "/docs/report.pdf",   "chunk_index": 1},
        {"text": "python machine learning tutorial",      "filepath": "/docs/tutorial.txt", "chunk_index": 0},
    ]

    fake_embeddings = np.random.rand(3, 384).astype("float32")

    print(f"Vectors before: {store.get_total_vectors()}")
    store.add_chunks(fake_chunks, fake_embeddings)
    print(f"Vectors after:  {store.get_total_vectors()}")
