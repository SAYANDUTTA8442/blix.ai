"""
Failure Pattern Mining — Blix v0.3.10  (New module 7)

Discovers recurring failure patterns across
``agents.failure_memory.FailureMemory`` records by clustering, rather
than relying solely on the existing Jaccard-similarity exact-match
lookup (which only finds failures textually similar to ONE query —
it can't surface "these 8 distinct-looking failures are actually all
instances of the same underlying retrieval problem").

Implementation note on the spec's named tools: the spec names HDBSCAN
and UMAP. Neither is installed in this environment and there's no
network path to install/download them here (network is allowlisted to
package indices only, and even so, HDBSCAN/UMAP wheels aren't reachable
in this sandbox's restricted set). Rather than faking their presence,
this module uses ``sklearn.cluster.DBSCAN`` (density-based, same
underlying family as HDBSCAN — no need to pre-specify cluster count)
on TF-IDF text features (``sklearn.feature_extraction.text.TfidfVectorizer``)
as a genuinely-functional substitute with the same shape of output: a
cluster id per failure, density-based (not centroid-based) so
oddly-shaped or differently-sized failure clusters are still found.
If HDBSCAN/UMAP become available in a future environment, swapping the
estimator is a one-line change — the clustering pipeline and output
contract here are written to make that swap trivial.

Feeds:
    reflection.reflection_engine.ReflectionEngine   — cluster summaries become reflection material
    metacognition.self_model.SelfModelStore           — recurring clusters can flag a weak capability domain

Python 3.10 compatible.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from agents.failure_memory import FailureMemory, FailureRecord
from utils.logger import get_logger

log = get_logger(__name__)

_MIN_RECORDS_TO_CLUSTER = 6   # below this, clustering is statistically meaningless


@dataclass
class FailureCluster:
    """One discovered cluster of related failure records."""

    cluster_id: int
    records: list[FailureRecord] = field(default_factory=list)
    representative_terms: list[str] = field(default_factory=list)
    total_occurrences: int = 0
    discovered_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "cluster_id": self.cluster_id,
            "size": len(self.records),
            "total_occurrences": self.total_occurrences,
            "representative_terms": self.representative_terms,
            "sample_failures": [r.failure for r in self.records[:3]],
            "tools_involved": sorted({r.tool for r in self.records if r.tool}),
            "discovered_at": self.discovered_at,
        }

    @property
    def is_noise(self) -> bool:
        """DBSCAN labels outliers as cluster -1 — not part of any recurring pattern."""
        return self.cluster_id == -1


class FailureClusterer:
    """
    Clusters ``FailureMemory`` records to discover recurring failure
    patterns that exact-match lookup would miss.

    Parameters
    ----------
    failure_memory:
        ``FailureMemory`` — source of failure records to cluster.
    min_records_to_cluster:
        Minimum failure records before clustering is attempted —
        clustering fewer than this produces statistically meaningless
        groupings.
    eps:
        DBSCAN neighborhood radius (in cosine-distance-derived space)
        — smaller values produce tighter, more numerous clusters.
    min_samples:
        DBSCAN minimum cluster size.
    """

    def __init__(
        self,
        failure_memory: FailureMemory,
        min_records_to_cluster: int = _MIN_RECORDS_TO_CLUSTER,
        eps: float = 0.88,
        min_samples: int = 2,
    ) -> None:
        self._failure_memory = failure_memory
        self._min_records = min_records_to_cluster
        self._eps = eps
        self._min_samples = min_samples

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------

    def discover_clusters(self) -> list[FailureCluster]:
        """
        Cluster all current failure records by TF-IDF similarity of
        their (task_title + failure) text using DBSCAN.

        Returns an empty list if there aren't enough records yet —
        clustering output below the minimum sample threshold would be
        noise, not signal.
        """
        records = self._failure_memory.most_common_failures(top_k=self._failure_memory.count)
        if len(records) < self._min_records:
            return []

        try:
            from sklearn.cluster import DBSCAN
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError:
            log.warning("FailureClusterer: scikit-learn unavailable; skipping clustering.")
            return []

        texts = [f"{r.task_title} {r.failure}" for r in records]
        try:
            vectorizer = TfidfVectorizer(max_features=200, stop_words="english")
            X = vectorizer.fit_transform(texts)
            labels = DBSCAN(eps=self._eps, min_samples=self._min_samples, metric="cosine").fit_predict(X.toarray())
            feature_names = vectorizer.get_feature_names_out()
        except ValueError:
            # Degenerate input (e.g. all-empty after stopword removal) — nothing to cluster.
            return []

        clusters: dict[int, list[int]] = {}
        for idx, label in enumerate(labels):
            clusters.setdefault(int(label), []).append(idx)

        results = []
        for cluster_id, indices in clusters.items():
            cluster_records = [records[i] for i in indices]
            terms = self._top_terms(X, indices, feature_names) if cluster_id != -1 else []
            results.append(FailureCluster(
                cluster_id=cluster_id, records=cluster_records, representative_terms=terms,
                total_occurrences=sum(r.occurrences for r in cluster_records),
            ))
        return sorted(results, key=lambda c: (-c.total_occurrences, c.cluster_id))

    @staticmethod
    def _top_terms(X, indices: list[int], feature_names, top_k: int = 5) -> list[str]:
        """Mean TF-IDF weight per term across a cluster's documents, top-k highest."""
        sub = X[indices].toarray()
        mean_weights = sub.mean(axis=0)
        top_idx = mean_weights.argsort()[::-1][:top_k]
        return [feature_names[i] for i in top_idx if mean_weights[i] > 0]

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def recurring_clusters(self, min_size: int = 2) -> list[FailureCluster]:
        """Non-noise clusters with at least ``min_size`` records — the actually-actionable patterns."""
        clusters = self.discover_clusters()
        return [c for c in clusters if not c.is_noise and len(c.records) >= min_size]

    def summarize_for_reflection(self) -> list[str]:
        """
        Render recurring clusters as short text summaries suitable for
        feeding into ``reflection.meta_reflection.MetaReflectionEngine``
        or ``reflection.reflection_engine.ReflectionEngine.reflect()``.
        """
        summaries = []
        for cluster in self.recurring_clusters():
            terms = ", ".join(cluster.representative_terms[:3]) or "unspecified pattern"
            summaries.append(
                f"Recurring failure pattern ({len(cluster.records)} related failures, "
                f"{cluster.total_occurrences} total occurrences) involving: {terms}."
            )
        return summaries
