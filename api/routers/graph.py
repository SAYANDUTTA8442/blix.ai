"""
/graph router — Blix v0.3.3

Endpoints
---------
GET  /graph                    — full graph snapshot (nodes + edges)
GET  /graph/nodes              — list nodes, optional kind filter
GET  /graph/nodes/{id}         — node detail + neighbours
GET  /graph/path               — BFS shortest path
GET  /graph/centrality         — top nodes by degree centrality
POST /graph/relations          — add / update a relation
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from api.context import BlixContext
from api.deps import get_context
from api.models import (
    GraphEdgeItem, GraphNeighboursResponse, GraphNodeItem,
    GraphPathResponse, GraphStatsResponse, UpsertRelationRequest,
)
from core.memory_graph import EntityKind, RelationKind

router = APIRouter(prefix="/graph", tags=["Graph"])


def _node_item(n: object) -> GraphNodeItem:
    return GraphNodeItem(
        id=getattr(n, "id"),
        kind=getattr(n, "kind"),
        label=getattr(n, "label"),
        aliases=list(getattr(n, "aliases", [])),
    )


def _edge_item(e: object) -> GraphEdgeItem:
    return GraphEdgeItem(
        from_id=getattr(e, "from_id"),
        relation=getattr(e, "relation"),
        to_id=getattr(e, "to_id"),
        confidence=getattr(e, "confidence", 1.0),
    )


@router.get("", response_model=GraphStatsResponse, summary="Full graph snapshot")
async def graph_snapshot(
    limit_nodes: int = Query(default=100, ge=1, le=500),
    limit_edges: int = Query(default=200, ge=1, le=1000),
    ctx: BlixContext = Depends(get_context),
) -> GraphStatsResponse:
    """Return the full graph (nodes + edges) up to the requested limits."""
    nodes = ctx.graph.list_nodes()[:limit_nodes]
    edges = []
    for node in ctx.graph.list_nodes():
        edges.extend(ctx.graph.get_edges(from_id=node.id))
    edges = edges[:limit_edges]
    return GraphStatsResponse(
        node_count=ctx.graph.node_count,
        edge_count=ctx.graph.edge_count,
        nodes=[_node_item(n) for n in nodes],
        edges=[_edge_item(e) for e in edges],
    )


@router.get("/nodes", response_model=list[GraphNodeItem], summary="List graph nodes")
async def list_nodes(
    kind: str | None = Query(default=None, description="Filter by entity kind"),
    ctx: BlixContext = Depends(get_context),
) -> list[GraphNodeItem]:
    kind_enum = None
    if kind is not None:
        try:
            kind_enum = EntityKind(kind)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Unknown entity kind: {kind!r}")
    nodes = ctx.graph.list_nodes(kind=kind_enum)
    return [_node_item(n) for n in nodes]


@router.get("/nodes/{node_id}", response_model=GraphNeighboursResponse, summary="Node detail + neighbours")
async def get_node(node_id: str, ctx: BlixContext = Depends(get_context)) -> GraphNeighboursResponse:
    node = ctx.graph.get_node(node_id)
    if node is None:
        raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found.")
    neighbours = ctx.graph_reasoner.related_entities(node_id, depth=1)
    nb_items = [
        {"id": n.id, "label": n.label, "kind": n.kind, "hop": hop}
        for n, hop in neighbours
    ]
    return GraphNeighboursResponse(node_id=node_id, neighbours=nb_items)


@router.get("/path", response_model=GraphPathResponse, summary="Shortest path between nodes")
async def graph_path(
    from_id: str = Query(...),
    to_id: str = Query(...),
    ctx: BlixContext = Depends(get_context),
) -> GraphPathResponse:
    path = ctx.graph_reasoner.shortest_path(from_id, to_id)
    if path is None:
        return GraphPathResponse(from_id=from_id, to_id=to_id, path=None, hops=None)
    return GraphPathResponse(
        from_id=from_id,
        to_id=to_id,
        path={"nodes": path.nodes, "relations": path.relations, "confidence": path.total_confidence},
        hops=len(path.nodes) - 1,
    )


@router.get("/centrality", summary="Top nodes by degree centrality")
async def centrality(
    top_k: int = Query(default=10, ge=1, le=50),
    ctx: BlixContext = Depends(get_context),
) -> list[dict]:
    """Return the top-k most connected entities in the graph."""
    top = ctx.graph_reasoner.most_central_nodes(top_k)
    return [{"node_id": nid, "centrality": round(c, 4)} for nid, c in top]


@router.post("/relations", summary="Add or update a relation")
async def upsert_relation(
    req: UpsertRelationRequest,
    ctx: BlixContext = Depends(get_context),
) -> dict:
    """Add or merge an entity relationship into the memory graph."""
    try:
        from_kind = EntityKind(req.from_kind)
        to_kind = EntityKind(req.to_kind)
        relation = RelationKind(req.relation)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    ctx.graph.upsert_relation(
        from_label=req.from_label,
        from_kind=from_kind,
        relation=relation,
        to_label=req.to_label,
        to_kind=to_kind,
        confidence=req.confidence,
    )
    return {
        "from": req.from_label, "relation": req.relation, "to": req.to_label,
        "node_count": ctx.graph.node_count, "edge_count": ctx.graph.edge_count,
    }
