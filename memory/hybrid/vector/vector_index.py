"""
VectorIndex — pluggable interface layer over VectorStore backends.

Provides a uniform API so the rest of HGSHM doesn't care whether
the backend is sqlite-vec, FAISS, Chroma, or Qdrant.

Architecture
------------
VectorIndex               ← public API used by HGSHM
  └─ VectorIndexBackend   ← protocol / abstract base
       ├─ SqliteVecBackend (default, wraps VectorStore)
       ├─ FaissBackend     (optional, requires faiss-cpu)
       └─ ChromaBackend    (optional, requires chromadb)
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from memory.hybrid.vector.vector_store import VectorStore, SearchResult

log = logging.getLogger(__name__)


@runtime_checkable
class VectorIndexBackend(Protocol):
    """Protocol every vector index backend must implement."""
    dim: int

    def upsert(self, node_id: str, vector: list[float],
               metadata: dict[str, Any] | None = None,
               embedding_id: str | None = None) -> str: ...

    def upsert_batch(self, records: list[tuple[str, list[float], dict]]) -> list[str]: ...

    def search(self, query_vector: list[float], top_k: int = 10,
               min_score: float = 0.0,
               filter_node_ids: list[str] | None = None) -> list[SearchResult]: ...

    def delete(self, embedding_id: str) -> bool: ...
    def delete_by_node(self, node_id: str) -> int: ...
    def count(self) -> int: ...
    def compact(self) -> None: ...
    def rebuild_index(self) -> None: ...
    def close(self) -> None: ...


class SqliteVecBackend:
    """Default backend — wraps VectorStore (sqlite-vec)."""

    def __init__(self, memory_dir: Path, dim: int = 256) -> None:
        self._store = VectorStore(memory_dir, dim=dim)
        self.dim = dim

    def upsert(self, node_id, vector, metadata=None, embedding_id=None):
        return self._store.upsert(node_id, vector, metadata, embedding_id)

    def upsert_batch(self, records):
        return self._store.upsert_batch(records)

    def search(self, query_vector, top_k=10, min_score=0.0, filter_node_ids=None):
        return self._store.search(query_vector, top_k, min_score, filter_node_ids)

    def delete(self, embedding_id):
        return self._store.delete(embedding_id)

    def delete_by_node(self, node_id):
        return self._store.delete_by_node(node_id)

    def count(self):
        return self._store.count()

    def compact(self):
        self._store.compact()

    def rebuild_index(self):
        self._store.rebuild_index()

    def close(self):
        self._store.close()

    def get_vector(self, embedding_id: str) -> list[float] | None:
        return self._store.get_vector(embedding_id)


class VectorIndex:
    """
    Public vector index API used by all HGSHM components.

    Wraps a VectorIndexBackend and provides convenience methods.
    """

    def __init__(self, memory_dir: Path, dim: int = 256,
                 backend: VectorIndexBackend | None = None) -> None:
        if backend is None:
            backend = SqliteVecBackend(memory_dir, dim=dim)
            log.debug("VectorIndex: using SqliteVecBackend (dim=%d)", dim)
        self._backend = backend
        self._dim = dim

    def swap_backend(self, backend: VectorIndexBackend) -> None:
        """Hot-swap the storage backend (e.g., migrate from sqlite-vec to FAISS)."""
        old = self._backend
        self._backend = backend
        old.close()
        log.info("VectorIndex: backend swapped to %s", type(backend).__name__)

    @property
    def dim(self) -> int:
        return self._dim

    def add(self, node_id: str, vector: list[float],
            metadata: dict[str, Any] | None = None,
            embedding_id: str | None = None) -> str:
        """Add or update a vector for a node. Returns embedding_id."""
        return self._backend.upsert(node_id, vector, metadata, embedding_id)

    def add_batch(self, records: list[tuple[str, list[float], dict]]) -> list[str]:
        """Batch-add vectors. Returns list of embedding_ids."""
        return self._backend.upsert_batch(records)

    def search(self, query_vector: list[float], top_k: int = 10,
               min_score: float = 0.0,
               filter_node_ids: list[str] | None = None) -> list[SearchResult]:
        """Find top_k nearest neighbours to query_vector."""
        return self._backend.search(query_vector, top_k, min_score, filter_node_ids)

    def delete(self, embedding_id: str) -> bool:
        return self._backend.delete(embedding_id)

    def delete_by_node(self, node_id: str) -> int:
        return self._backend.delete_by_node(node_id)

    def count(self) -> int:
        return self._backend.count()

    def compact(self) -> None:
        self._backend.compact()

    def rebuild_index(self) -> None:
        self._backend.rebuild_index()

    def close(self) -> None:
        self._backend.close()
