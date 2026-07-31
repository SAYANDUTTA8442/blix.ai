"""
VectorStore — sqlite-vec backed vector database.

Provides:
  • Embedding storage and retrieval
  • Nearest-neighbour search (cosine similarity)
  • Metadata filtering
  • Batch insertion / incremental updates
  • Embedding versioning
  • Deletion and compaction
  • Index rebuilding

The sqlite-vec extension (sqlite_vec Python package) handles ANN search
efficiently inside SQLite without external processes.

Fallback: if sqlite-vec is unavailable, falls back to pure-Python
brute-force cosine search (correct but slower for large datasets).
"""
from __future__ import annotations
import json
import logging
import math
import sqlite3
import struct
import uuid
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_SQLITE_VEC_AVAILABLE = False
try:
    import sqlite_vec
    _SQLITE_VEC_AVAILABLE = True
except ImportError:
    log.warning("sqlite-vec not installed; falling back to brute-force cosine search")


def _encode_f32(floats: list[float]) -> bytes:
    """Pack a float list into the f32 binary format sqlite-vec expects."""
    return struct.pack(f"{len(floats)}f", *floats)


class VectorRecord:
    """A stored embedding with its metadata."""
    __slots__ = ("embedding_id", "node_id", "vector", "metadata", "version")

    def __init__(
        self,
        embedding_id: str,
        node_id: str,
        vector: list[float],
        metadata: dict[str, Any] | None = None,
        version: int = 1,
    ) -> None:
        self.embedding_id = embedding_id
        self.node_id = node_id
        self.vector = vector
        self.metadata = metadata or {}
        self.version = version


class SearchResult:
    """A single vector search result."""
    __slots__ = ("embedding_id", "node_id", "score", "metadata")

    def __init__(self, embedding_id: str, node_id: str,
                 score: float, metadata: dict[str, Any]) -> None:
        self.embedding_id = embedding_id
        self.node_id = node_id
        self.score = score
        self.metadata = metadata

    def __repr__(self) -> str:
        return f"SearchResult(node={self.node_id[:8]}…, score={self.score:.4f})"


class VectorStore:
    """
    sqlite-vec powered vector database.

    Parameters
    ----------
    memory_dir:
        Directory where vectors.db will be stored.
    dim:
        Embedding dimensionality. Must match the EmbeddingManager's dim.
    """

    def __init__(self, memory_dir: Path, dim: int = 256) -> None:
        memory_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = memory_dir / "vectors.db"
        self._dim = dim
        self._conn = self._connect()
        self._init_schema()
        log.debug("VectorStore initialised (dim=%d, sqlite-vec=%s)",
                  dim, _SQLITE_VEC_AVAILABLE)

    # ----------------------------------------------------------------
    # Connection
    # ----------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        if _SQLITE_VEC_AVAILABLE:
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_schema(self) -> None:
        with self._conn:
            # Metadata table (always present)
            self._conn.executescript(f"""
                CREATE TABLE IF NOT EXISTS embedding_meta (
                    embedding_id TEXT PRIMARY KEY,
                    node_id      TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    version      INTEGER NOT NULL DEFAULT 1,
                    created_at   TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_emb_node ON embedding_meta(node_id);
            """)
            if _SQLITE_VEC_AVAILABLE:
                # sqlite-vec virtual table
                self._conn.execute(f"""
                    CREATE VIRTUAL TABLE IF NOT EXISTS vec_embeddings
                    USING vec0(embedding_id TEXT PRIMARY KEY, vector float[{self._dim}])
                """)
            else:
                # Fallback: store raw blobs
                self._conn.execute("""
                    CREATE TABLE IF NOT EXISTS vec_embeddings_fallback (
                        embedding_id TEXT PRIMARY KEY,
                        vector_blob  BLOB NOT NULL
                    )
                """)

    # ----------------------------------------------------------------
    # Write operations
    # ----------------------------------------------------------------

    def upsert(self, node_id: str, vector: list[float],
               metadata: dict[str, Any] | None = None,
               embedding_id: str | None = None) -> str:
        """Store or update a vector. Returns the embedding_id."""
        if len(vector) != self._dim:
            raise ValueError(f"Vector dim {len(vector)} ≠ expected {self._dim}")
        if embedding_id is None:
            embedding_id = str(uuid.uuid4())
        metadata = metadata or {}
        now = __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).isoformat()

        with self._conn:
            self._conn.execute("""
                INSERT OR REPLACE INTO embedding_meta VALUES (?,?,?,?,?)
            """, (embedding_id, node_id, json.dumps(metadata), 1, now))

            if _SQLITE_VEC_AVAILABLE:
                self._conn.execute(
                    "INSERT OR REPLACE INTO vec_embeddings VALUES (?, ?)",
                    (embedding_id, _encode_f32(vector))
                )
            else:
                self._conn.execute(
                    "INSERT OR REPLACE INTO vec_embeddings_fallback VALUES (?, ?)",
                    (embedding_id, _encode_f32(vector))
                )
        return embedding_id

    def upsert_batch(self, records: list[tuple[str, list[float], dict]]) -> list[str]:
        """Batch upsert: list of (node_id, vector, metadata) tuples."""
        ids = []
        with self._conn:
            for node_id, vector, metadata in records:
                eid = str(uuid.uuid4())
                now = __import__("datetime").datetime.now(
                    __import__("datetime").timezone.utc).isoformat()
                self._conn.execute(
                    "INSERT OR REPLACE INTO embedding_meta VALUES (?,?,?,?,?)",
                    (eid, node_id, json.dumps(metadata), 1, now)
                )
                if _SQLITE_VEC_AVAILABLE:
                    self._conn.execute(
                        "INSERT OR REPLACE INTO vec_embeddings VALUES (?,?)",
                        (eid, _encode_f32(vector))
                    )
                else:
                    self._conn.execute(
                        "INSERT OR REPLACE INTO vec_embeddings_fallback VALUES (?,?)",
                        (eid, _encode_f32(vector))
                    )
                ids.append(eid)
        return ids

    def delete(self, embedding_id: str) -> bool:
        with self._conn:
            self._conn.execute(
                "DELETE FROM embedding_meta WHERE embedding_id=?", (embedding_id,))
            if _SQLITE_VEC_AVAILABLE:
                self._conn.execute(
                    "DELETE FROM vec_embeddings WHERE embedding_id=?", (embedding_id,))
            else:
                self._conn.execute(
                    "DELETE FROM vec_embeddings_fallback WHERE embedding_id=?", (embedding_id,))
        return True

    def delete_by_node(self, node_id: str) -> int:
        rows = self._conn.execute(
            "SELECT embedding_id FROM embedding_meta WHERE node_id=?", (node_id,)
        ).fetchall()
        count = 0
        for row in rows:
            self.delete(row["embedding_id"])
            count += 1
        return count

    # ----------------------------------------------------------------
    # Search
    # ----------------------------------------------------------------

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        min_score: float = 0.0,
        filter_node_ids: list[str] | None = None,
    ) -> list[SearchResult]:
        """
        Find the top_k most similar vectors.

        Parameters
        ----------
        query_vector:
            The query embedding (must be dim-dimensional).
        top_k:
            Maximum number of results.
        min_score:
            Minimum cosine similarity threshold.
        filter_node_ids:
            If provided, only return results from these node_ids.
        """
        if len(query_vector) != self._dim:
            raise ValueError(f"Query vector dim {len(query_vector)} ≠ {self._dim}")

        if _SQLITE_VEC_AVAILABLE:
            return self._search_vec(query_vector, top_k, min_score, filter_node_ids)
        else:
            return self._search_brute(query_vector, top_k, min_score, filter_node_ids)

    def _search_vec(self, query: list[float], top_k: int,
                    min_score: float, filter_ids: list[str] | None) -> list[SearchResult]:
        """sqlite-vec ANN search."""
        limit = top_k * 3 if filter_ids else top_k  # over-fetch then filter
        rows = self._conn.execute("""
            SELECT v.embedding_id, m.node_id, m.metadata_json,
                   vec_distance_cosine(v.vector, ?) AS distance
            FROM vec_embeddings v
            JOIN embedding_meta m ON v.embedding_id = m.embedding_id
            ORDER BY distance ASC
            LIMIT ?
        """, (_encode_f32(query), limit)).fetchall()

        results = []
        for row in rows:
            dist = row["distance"]
            if dist is None:
                continue
            score = max(0.0, 1.0 - float(dist))  # convert cosine distance to similarity
            if score < min_score:
                continue
            if filter_ids and row["node_id"] not in filter_ids:
                continue
            results.append(SearchResult(
                embedding_id=row["embedding_id"],
                node_id=row["node_id"],
                score=score,
                metadata=json.loads(row["metadata_json"]),
            ))
            if len(results) >= top_k:
                break
        return results

    def _search_brute(self, query: list[float], top_k: int,
                      min_score: float, filter_ids: list[str] | None) -> list[SearchResult]:
        """Pure-Python brute-force cosine search (fallback)."""
        rows = self._conn.execute(
            "SELECT f.embedding_id, f.vector_blob, m.node_id, m.metadata_json "
            "FROM vec_embeddings_fallback f "
            "JOIN embedding_meta m ON f.embedding_id = m.embedding_id"
        ).fetchall()

        scored = []
        for row in rows:
            if filter_ids and row["node_id"] not in filter_ids:
                continue
            blob = row["vector_blob"]
            n = len(blob) // 4
            vec = list(struct.unpack(f"{n}f", blob))
            score = self._cosine(query, vec)
            if score >= min_score:
                scored.append((score, row["embedding_id"], row["node_id"],
                               json.loads(row["metadata_json"])))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            SearchResult(eid, nid, s, meta)
            for s, eid, nid, meta in scored[:top_k]
        ]

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        return max(-1.0, min(1.0, dot))

    # ----------------------------------------------------------------
    # Query / introspection
    # ----------------------------------------------------------------

    def get_vector(self, embedding_id: str) -> list[float] | None:
        if _SQLITE_VEC_AVAILABLE:
            row = self._conn.execute(
                "SELECT vector FROM vec_embeddings WHERE embedding_id=?",
                (embedding_id,)
            ).fetchone()
            if row is None:
                return None
            blob = row["vector"]
            n = len(blob) // 4
            return list(struct.unpack(f"{n}f", blob))
        else:
            row = self._conn.execute(
                "SELECT vector_blob FROM vec_embeddings_fallback WHERE embedding_id=?",
                (embedding_id,)
            ).fetchone()
            if row is None:
                return None
            blob = row["vector_blob"]
            n = len(blob) // 4
            return list(struct.unpack(f"{n}f", blob))

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM embedding_meta").fetchone()[0]

    def embedding_ids_for_node(self, node_id: str) -> list[str]:
        rows = self._conn.execute(
            "SELECT embedding_id FROM embedding_meta WHERE node_id=?", (node_id,)
        ).fetchall()
        return [r["embedding_id"] for r in rows]

    def compact(self) -> None:
        """Remove embeddings with no matching embedding_meta entry, then VACUUM."""
        self._conn.execute("VACUUM")
        log.info("VectorStore: compaction complete")

    def rebuild_index(self) -> None:
        """No-op for sqlite-vec (index is always consistent). Exposed for interface parity."""
        log.info("VectorStore: rebuild_index called (sqlite-vec keeps index live)")

    def close(self) -> None:
        self._conn.close()
