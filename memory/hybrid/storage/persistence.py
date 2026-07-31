"""
HGSHM Persistence Layer.

Uses two SQLite databases:
  1. hgshm.db  — nodes, edges, clusters (structured data)
  2. vectors.db — vector embeddings via sqlite-vec

Both are co-located in memory_dir and journaled separately so
vector index rebuilds don't corrupt the graph data.
"""
from __future__ import annotations
import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Iterator

log = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────
# Schema DDL
# ────────────────────────────────────────────────────────────────────

_NODES_DDL = """
CREATE TABLE IF NOT EXISTS memory_nodes (
    node_id          TEXT PRIMARY KEY,
    text             TEXT NOT NULL,
    memory_type      TEXT NOT NULL,
    hierarchy_level  INTEGER NOT NULL,
    confidence       REAL NOT NULL,
    importance       REAL NOT NULL,
    embedding_id     TEXT,
    concept_id       TEXT,
    source           TEXT NOT NULL,
    epistemic_status TEXT NOT NULL,
    metadata_json    TEXT NOT NULL,
    tags_json        TEXT NOT NULL,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL,
    last_accessed_at TEXT,
    valid_from       TEXT,
    valid_until      TEXT,
    access_count     INTEGER NOT NULL DEFAULT 0,
    version          INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_nodes_type      ON memory_nodes(memory_type);
CREATE INDEX IF NOT EXISTS idx_nodes_hierarchy ON memory_nodes(hierarchy_level);
CREATE INDEX IF NOT EXISTS idx_nodes_concept   ON memory_nodes(concept_id);
CREATE INDEX IF NOT EXISTS idx_nodes_updated   ON memory_nodes(updated_at);
CREATE INDEX IF NOT EXISTS idx_nodes_importance ON memory_nodes(importance DESC);
"""

_EDGES_DDL = """
CREATE TABLE IF NOT EXISTS memory_edges (
    edge_id        TEXT PRIMARY KEY,
    source_id      TEXT NOT NULL,
    target_id      TEXT NOT NULL,
    relation       TEXT NOT NULL,
    confidence     REAL NOT NULL,
    weight         REAL NOT NULL,
    provenance     TEXT NOT NULL,
    metadata_json  TEXT NOT NULL,
    evidence_count INTEGER NOT NULL DEFAULT 1,
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL,
    FOREIGN KEY (source_id) REFERENCES memory_nodes(node_id),
    FOREIGN KEY (target_id) REFERENCES memory_nodes(node_id)
);
CREATE INDEX IF NOT EXISTS idx_edges_source   ON memory_edges(source_id);
CREATE INDEX IF NOT EXISTS idx_edges_target   ON memory_edges(target_id);
CREATE INDEX IF NOT EXISTS idx_edges_relation ON memory_edges(relation);
CREATE INDEX IF NOT EXISTS idx_edges_src_rel  ON memory_edges(source_id, relation);
CREATE INDEX IF NOT EXISTS idx_edges_tgt_rel  ON memory_edges(target_id, relation);
"""

_CLUSTERS_DDL = """
CREATE TABLE IF NOT EXISTS memory_clusters (
    cluster_id             TEXT PRIMARY KEY,
    name                   TEXT NOT NULL,
    node_ids_json          TEXT NOT NULL,
    centroid_embedding_id  TEXT,
    concept_node_id        TEXT,
    coherence              REAL NOT NULL DEFAULT 0.0,
    tags_json              TEXT NOT NULL,
    metadata_json          TEXT NOT NULL,
    created_at             TEXT NOT NULL,
    updated_at             TEXT NOT NULL
);
"""

_HISTORY_DDL = """
CREATE TABLE IF NOT EXISTS node_history (
    history_id  TEXT PRIMARY KEY,
    node_id     TEXT NOT NULL,
    version     INTEGER NOT NULL,
    snapshot    TEXT NOT NULL,
    changed_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_history_node ON node_history(node_id, version);
"""


class HGSHMStore:
    """
    Low-level SQLite persistence for HGSHM.

    All CRUD operations on MemoryNode, MemoryEdge, and MemoryCluster
    go through this class. GraphStore, VectorStore, etc. depend on it
    but should not call sqlite directly.
    """

    def __init__(self, memory_dir: Path) -> None:
        memory_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = memory_dir / "hgshm.db"
        self._conn = self._connect()
        self._init_schema()
        log.debug("HGSHMStore initialised at %s", self._db_path)

    # ----------------------------------------------------------------
    # Connection management
    # ----------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.executescript(_NODES_DDL)
            self._conn.executescript(_EDGES_DDL)
            self._conn.executescript(_CLUSTERS_DDL)
            self._conn.executescript(_HISTORY_DDL)

    def close(self) -> None:
        self._conn.close()

    # ----------------------------------------------------------------
    # Node CRUD
    # ----------------------------------------------------------------

    def save_node(self, node: "MemoryNode") -> None:  # type: ignore[name-defined]
        d = node.to_dict()
        with self._conn:
            self._conn.execute("""
                INSERT OR REPLACE INTO memory_nodes VALUES (
                    :node_id,:text,:memory_type,:hierarchy_level,
                    :confidence,:importance,:embedding_id,:concept_id,
                    :source,:epistemic_status,
                    :metadata_json,:tags_json,
                    :created_at,:updated_at,:last_accessed_at,
                    :valid_from,:valid_until,:access_count,:version
                )
            """, {**d,
                  "metadata_json": json.dumps(d["metadata"]),
                  "tags_json":     json.dumps(d["tags"])})

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM memory_nodes WHERE node_id=?", (node_id,)
        ).fetchone()
        return self._row_to_node_dict(row) if row else None

    def delete_node(self, node_id: str) -> bool:
        with self._conn:
            self._conn.execute("PRAGMA foreign_keys=OFF")
            cur = self._conn.execute(
                "DELETE FROM memory_nodes WHERE node_id=?", (node_id,))
            self._conn.execute("PRAGMA foreign_keys=ON")
        return cur.rowcount > 0

    def all_nodes(
        self,
        memory_type: str | None = None,
        hierarchy_level: int | None = None,
        min_confidence: float = 0.0,
        min_importance: float = 0.0,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        clauses = ["confidence >= ?", "importance >= ?"]
        params: list[Any] = [min_confidence, min_importance]
        if memory_type:
            clauses.append("memory_type = ?"); params.append(memory_type)
        if hierarchy_level is not None:
            clauses.append("hierarchy_level = ?"); params.append(hierarchy_level)
        where = " AND ".join(clauses)
        params += [limit, offset]
        rows = self._conn.execute(
            f"SELECT * FROM memory_nodes WHERE {where} "
            f"ORDER BY importance DESC, updated_at DESC LIMIT ? OFFSET ?",
            params,
        ).fetchall()
        return [self._row_to_node_dict(r) for r in rows]

    def count_nodes(self, memory_type: str | None = None) -> int:
        if memory_type:
            return self._conn.execute(
                "SELECT COUNT(*) FROM memory_nodes WHERE memory_type=?", (memory_type,)
            ).fetchone()[0]
        return self._conn.execute("SELECT COUNT(*) FROM memory_nodes").fetchone()[0]

    def nodes_by_tags(
        self,
        required_tags: list[str],
        memory_type: str | None = None,
        limit: int = 500,
        order_by: str = "importance",
    ) -> list[dict[str, Any]]:
        """
        Return nodes that contain ALL of the given tags.

        Uses SQLite ``json_each()`` to query the ``tags_json`` column
        directly — O(k×index) where k = len(required_tags), not O(n) over
        all nodes.  Replaces ``all_nodes(limit=N) + Python filter`` in
        SystemMemory and UserMemory (ISSUE-008).

        Parameters
        ----------
        required_tags : list[str]
            Every tag in this list must be present in the node's tags.
        memory_type : str | None
            Optional additional filter on memory_type column.
        limit : int
            Maximum rows to return (default 500).
        order_by : str
            Column to order by — ``"importance"`` or ``"updated_at"``.
        """
        if not required_tags:
            return self.all_nodes(memory_type=memory_type, limit=limit)

        # Build one EXISTS subquery per required tag.
        # All values are parameterised — no string interpolation of user data.
        exists_clauses = " AND ".join(
            "EXISTS (SELECT 1 FROM json_each(tags_json) WHERE value = ?)"
            for _ in required_tags
        )
        params: list[Any] = list(required_tags)

        type_clause = ""
        if memory_type:
            type_clause = " AND memory_type = ?"
            params.append(memory_type)

        order_col = "importance" if order_by == "importance" else "updated_at"
        params.append(limit)

        sql = (
            f"SELECT * FROM memory_nodes"
            f" WHERE {exists_clauses}{type_clause}"
            f" ORDER BY {order_col} DESC LIMIT ?"
        )
        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_node_dict(r) for r in rows]

    def count_by_tag(self, tag: str, memory_type: str | None = None) -> int:
        """
        Count nodes containing ``tag`` using json_each — O(index), not O(n).
        """
        if memory_type:
            row = self._conn.execute("""
                SELECT COUNT(node_id) FROM memory_nodes
                WHERE EXISTS (SELECT 1 FROM json_each(tags_json) WHERE value = ?)
                  AND memory_type = ?
            """, (tag, memory_type)).fetchone()
        else:
            row = self._conn.execute("""
                SELECT COUNT(node_id) FROM memory_nodes
                WHERE EXISTS (SELECT 1 FROM json_each(tags_json) WHERE value = ?)
            """, (tag,)).fetchone()
        return int(row[0]) if row else 0

    def stats_by_tag(self, tag: str) -> dict[str, int]:
        """
        Return a {memory_type: count} breakdown for nodes containing ``tag``.

        Uses GROUP BY — one DB round-trip instead of loading all nodes.
        Used by SystemMemory.stats() and UserMemory.stats() (ISSUE-008).
        """
        rows = self._conn.execute("""
            SELECT memory_type, COUNT(node_id) AS cnt
            FROM memory_nodes
            WHERE EXISTS (SELECT 1 FROM json_each(tags_json) WHERE value = ?)
            GROUP BY memory_type
        """, (tag,)).fetchall()
        return {row["memory_type"]: int(row["cnt"]) for row in rows}

    def search_nodes_by_text(self, query_tokens: list[str], limit: int = 50) -> list[dict[str, Any]]:
        """Simple token-overlap search (fast, no FTS5 required)."""
        if not query_tokens:
            return []
        # Use LIKE for each token — union of matches
        clauses = " OR ".join(["LOWER(text) LIKE ?" for _ in query_tokens])
        params = [f"%{t.lower()}%" for t in query_tokens] + [limit]
        rows = self._conn.execute(
            f"SELECT * FROM memory_nodes WHERE {clauses} "
            f"ORDER BY importance DESC LIMIT ?", params
        ).fetchall()
        return [self._row_to_node_dict(r) for r in rows]

    def _row_to_node_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        d["metadata"] = json.loads(d.pop("metadata_json", "{}"))
        d["tags"]     = json.loads(d.pop("tags_json", "[]"))
        return d

    # ----------------------------------------------------------------
    # Edge CRUD
    # ----------------------------------------------------------------

    def save_edge(self, edge: "MemoryEdge") -> None:  # type: ignore[name-defined]
        d = edge.to_dict()
        with self._conn:
            self._conn.execute("""
                INSERT OR REPLACE INTO memory_edges VALUES (
                    :edge_id,:source_id,:target_id,:relation,
                    :confidence,:weight,:provenance,:metadata_json,
                    :evidence_count,:created_at,:updated_at
                )
            """, {**d, "metadata_json": json.dumps(d["metadata"])})

    def get_edge(self, edge_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM memory_edges WHERE edge_id=?", (edge_id,)
        ).fetchone()
        return self._row_to_edge_dict(row) if row else None

    def find_edge(self, source_id: str, target_id: str, relation: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM memory_edges WHERE source_id=? AND target_id=? AND relation=?",
            (source_id, target_id, relation)
        ).fetchone()
        return self._row_to_edge_dict(row) if row else None

    def edges_from(self, node_id: str, relation: str | None = None) -> list[dict[str, Any]]:
        if relation:
            rows = self._conn.execute(
                "SELECT * FROM memory_edges WHERE source_id=? AND relation=?",
                (node_id, relation)
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM memory_edges WHERE source_id=?", (node_id,)
            ).fetchall()
        return [self._row_to_edge_dict(r) for r in rows]

    def edges_to(self, node_id: str, relation: str | None = None) -> list[dict[str, Any]]:
        if relation:
            rows = self._conn.execute(
                "SELECT * FROM memory_edges WHERE target_id=? AND relation=?",
                (node_id, relation)
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM memory_edges WHERE target_id=?", (node_id,)
            ).fetchall()
        return [self._row_to_edge_dict(r) for r in rows]

    def all_edges(self, relation: str | None = None, limit: int = 5000) -> list[dict[str, Any]]:
        if relation:
            rows = self._conn.execute(
                "SELECT * FROM memory_edges WHERE relation=? LIMIT ?", (relation, limit)
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM memory_edges ORDER BY updated_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._row_to_edge_dict(r) for r in rows]

    def delete_edge(self, edge_id: str) -> bool:
        with self._conn:
            cur = self._conn.execute("DELETE FROM memory_edges WHERE edge_id=?", (edge_id,))
        return cur.rowcount > 0

    def count_edges(self, relation: str | None = None) -> int:
        if relation:
            return self._conn.execute(
                "SELECT COUNT(*) FROM memory_edges WHERE relation=?", (relation,)
            ).fetchone()[0]
        return self._conn.execute("SELECT COUNT(*) FROM memory_edges").fetchone()[0]

    def _row_to_edge_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        d["metadata"] = json.loads(d.pop("metadata_json", "{}"))
        return d

    # ----------------------------------------------------------------
    # Cluster CRUD
    # ----------------------------------------------------------------

    def save_cluster(self, cluster: "MemoryCluster") -> None:  # type: ignore[name-defined]
        d = cluster.to_dict()
        with self._conn:
            self._conn.execute("""
                INSERT OR REPLACE INTO memory_clusters VALUES (
                    :cluster_id,:name,:node_ids_json,
                    :centroid_embedding_id,:concept_node_id,:coherence,
                    :tags_json,:metadata_json,:created_at,:updated_at
                )
            """, {**d,
                  "node_ids_json":  json.dumps(d["node_ids"]),
                  "tags_json":      json.dumps(d["tags"]),
                  "metadata_json":  json.dumps(d["metadata"])})

    def get_cluster(self, cluster_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM memory_clusters WHERE cluster_id=?", (cluster_id,)
        ).fetchone()
        return self._row_to_cluster_dict(row) if row else None

    def all_clusters(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM memory_clusters ORDER BY updated_at DESC"
        ).fetchall()
        return [self._row_to_cluster_dict(r) for r in rows]

    def delete_cluster(self, cluster_id: str) -> bool:
        with self._conn:
            cur = self._conn.execute(
                "DELETE FROM memory_clusters WHERE cluster_id=?", (cluster_id,))
        return cur.rowcount > 0

    def count_clusters(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM memory_clusters").fetchone()[0]

    def _row_to_cluster_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        d["node_ids"]  = json.loads(d.pop("node_ids_json", "[]"))
        d["tags"]      = json.loads(d.pop("tags_json", "[]"))
        d["metadata"]  = json.loads(d.pop("metadata_json", "{}"))
        return d

    # ----------------------------------------------------------------
    # History
    # ----------------------------------------------------------------

    def save_history(self, node_id: str, version: int, snapshot: dict) -> None:
        import uuid as _uuid
        with self._conn:
            self._conn.execute("""
                INSERT OR IGNORE INTO node_history VALUES (?,?,?,?,?)
            """, (str(_uuid.uuid4()), node_id, version,
                  json.dumps(snapshot),
                  __import__("datetime").datetime.now(
                      __import__("datetime").timezone.utc).isoformat()))

    def get_history(self, node_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM node_history WHERE node_id=? ORDER BY version ASC",
            (node_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ----------------------------------------------------------------
    # Bulk / maintenance
    # ----------------------------------------------------------------

    def vacuum(self) -> None:
        self._conn.execute("VACUUM")

    def stats(self) -> dict[str, int]:
        return {
            "nodes":    self.count_nodes(),
            "edges":    self.count_edges(),
            "clusters": self.count_clusters(),
        }
