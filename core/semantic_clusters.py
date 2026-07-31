"""
Semantic Memory Clustering — Blix v0.3.1  (Issue 3)

Addresses: "Hierarchy is temporal, not semantic."

Human memory is topic-based and goal-based, not only time-organized.
Discussions about "transformers", "RAG", "attention" across 6 months
should be grouped together regardless of when they occurred.

This module adds a ``SemanticClusterIndex`` that:
1. Groups MemoryEntry objects by topic/theme using cosine similarity
   over their embedding vectors.
2. Maintains named ``SemanticCluster`` objects (analogous to "concept nodes").
3. Provides cluster-aware retrieval: a query retrieves the best cluster
   first, then ranks within it — complementing temporal hierarchy.

The clustering uses a simple online nearest-neighbour algorithm
(no sklearn dependency): each new memory is assigned to the closest
existing centroid if similarity > threshold, else a new cluster is created.

Python 3.10 compatible.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Cluster model
# ---------------------------------------------------------------------------


@dataclass
class SemanticCluster:
    """
    A group of thematically related memories.

    Fields
    ------
    cluster_id:
        Unique string id, e.g. ``"cluster_7"``.
    label:
        Human-readable name, e.g. ``"transformers / attention"`` (auto-generated
        from the most common topics in member memories).
    centroid:
        Mean embedding of all member memories.  Updated incrementally.
    member_ids:
        MemoryEntry ids belonging to this cluster.
    dominant_topics:
        Most frequent topic tags across members.
    created_at / updated_at:
        Timestamps.
    """

    cluster_id: str
    label: str
    centroid: list[float]        # JSON-serialisable; convert to np.ndarray for maths
    member_ids: list[int] = field(default_factory=list)
    dominant_topics: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def centroid_array(self) -> np.ndarray:
        return np.array(self.centroid, dtype=np.float32)

    def to_dict(self) -> dict:
        return {
            "cluster_id": self.cluster_id,
            "label": self.label,
            "centroid": self.centroid,
            "member_ids": self.member_ids,
            "dominant_topics": self.dominant_topics,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SemanticCluster":
        return cls(
            cluster_id=d["cluster_id"],
            label=d.get("label", d["cluster_id"]),
            centroid=d["centroid"],
            member_ids=d.get("member_ids", []),
            dominant_topics=d.get("dominant_topics", []),
            created_at=d.get("created_at", ""),
            updated_at=d.get("updated_at", ""),
        )


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------


class SemanticClusterIndex:
    """
    Online semantic clustering index for MemoryEntry objects.

    Maintains a set of ``SemanticCluster`` objects.  New memories are
    assigned via nearest-centroid; clusters evolve incrementally.

    Parameters
    ----------
    clusters_file:
        Path to ``semantic_clusters.json``.
    similarity_threshold:
        Minimum cosine similarity to assign a memory to an existing cluster.
        If no cluster exceeds this threshold, a new cluster is created.
    max_label_topics:
        Number of dominant topics to include in the auto-generated label.
    """

    def __init__(
        self,
        clusters_file: Path,
        similarity_threshold: float = 0.65,
        max_label_topics: int = 3,
    ) -> None:
        self._file = clusters_file
        self._threshold = similarity_threshold
        self._max_topics = max_label_topics
        self._clusters: dict[str, SemanticCluster] = {}
        self._next_id = 0
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self._file.exists():
            return
        try:
            with self._file.open("r", encoding="utf-8") as fh:
                raw = json.load(fh)
            for item in raw:
                c = SemanticCluster.from_dict(item)
                self._clusters[c.cluster_id] = c
            if self._clusters:
                self._next_id = max(
                    int(cid.replace("cluster_", "")) for cid in self._clusters
                ) + 1
            log.info("SemanticClusterIndex: loaded %d clusters.", len(self._clusters))
        except Exception as exc:
            log.warning("SemanticClusterIndex: load failed (%s)", exc)

    def _save(self) -> None:
        self._file.parent.mkdir(parents=True, exist_ok=True)
        with self._file.open("w", encoding="utf-8") as fh:
            json.dump(
                [c.to_dict() for c in self._clusters.values()],
                fh, indent=2, ensure_ascii=False,
            )

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def add_memory(
        self,
        memory_id: int,
        embedding: np.ndarray,
        topics: list[str],
    ) -> str:
        """
        Assign a memory to the nearest cluster (or create a new one).

        Returns the cluster_id assigned.
        """
        embedding = _normalise(embedding)
        best_id, best_sim = self._nearest_cluster(embedding)

        if best_id is not None and best_sim >= self._threshold:
            cluster = self._clusters[best_id]
            self._update_cluster(cluster, memory_id, embedding, topics)
            log.debug(
                "SemanticClusterIndex: memory %d → cluster %s (sim=%.3f)",
                memory_id, best_id, best_sim,
            )
            return best_id
        else:
            cid = f"cluster_{self._next_id}"
            self._next_id += 1
            label = ", ".join(topics[:self._max_topics]) or cid
            cluster = SemanticCluster(
                cluster_id=cid,
                label=label,
                centroid=embedding.tolist(),
                member_ids=[memory_id],
                dominant_topics=topics[:self._max_topics],
            )
            self._clusters[cid] = cluster
            log.debug("SemanticClusterIndex: memory %d → new %s", memory_id, cid)
            self._save()
            return cid

    def _nearest_cluster(
        self, embedding: np.ndarray
    ) -> tuple[Optional[str], float]:
        """Find the cluster with the highest cosine similarity to embedding."""
        best_id: Optional[str] = None
        best_sim = -1.0
        for cid, cluster in self._clusters.items():
            centroid = cluster.centroid_array()
            sim = float(np.dot(embedding, centroid))  # both normalised
            if sim > best_sim:
                best_sim = sim
                best_id = cid
        return best_id, best_sim

    def _update_cluster(
        self,
        cluster: SemanticCluster,
        memory_id: int,
        embedding: np.ndarray,
        topics: list[str],
    ) -> None:
        """Incrementally update centroid and membership."""
        if memory_id not in cluster.member_ids:
            cluster.member_ids.append(memory_id)
        n = len(cluster.member_ids)
        # Running mean centroid update
        old = cluster.centroid_array()
        new_centroid = _normalise(old * ((n - 1) / n) + embedding * (1 / n))
        cluster.centroid = new_centroid.tolist()
        # Update dominant topics (union, capped)
        all_topics = cluster.dominant_topics + topics
        freq: dict[str, int] = {}
        for t in all_topics:
            freq[t] = freq.get(t, 0) + 1
        cluster.dominant_topics = sorted(freq, key=lambda t: -freq[t])[:self._max_topics]
        cluster.label = ", ".join(cluster.dominant_topics) or cluster.cluster_id
        cluster.updated_at = datetime.now(timezone.utc).isoformat()
        self._save()

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_cluster_for_query(
        self, query_embedding: np.ndarray, top_k: int = 3
    ) -> list[tuple[SemanticCluster, float]]:
        """
        Return the top-k clusters most similar to the query embedding,
        sorted by descending similarity.

        Returns list of (cluster, similarity) pairs.
        """
        q = _normalise(query_embedding)
        scored: list[tuple[SemanticCluster, float]] = []
        for cluster in self._clusters.values():
            centroid = cluster.centroid_array()
            sim = float(np.dot(q, centroid))
            scored.append((cluster, sim))
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]

    def get_cluster_members(self, cluster_id: str) -> list[int]:
        """Return all MemoryEntry ids belonging to a cluster."""
        c = self._clusters.get(cluster_id)
        return list(c.member_ids) if c else []

    def get_cluster(self, cluster_id: str) -> Optional[SemanticCluster]:
        return self._clusters.get(cluster_id)

    def list_clusters(self) -> list[SemanticCluster]:
        return sorted(
            self._clusters.values(),
            key=lambda c: len(c.member_ids),
            reverse=True,
        )

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def cluster_count(self) -> int:
        return len(self._clusters)

    def summary(self) -> str:
        """One-line summary for the CLI /stats command."""
        if not self._clusters:
            return "No semantic clusters yet."
        top = sorted(self._clusters.values(), key=lambda c: -len(c.member_ids))[:3]
        parts = [f'"{c.label}" ({len(c.member_ids)}m)' for c in top]
        return f"{self.cluster_count} clusters: " + ", ".join(parts)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalise(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-8 else v
