"""
API request/response models — Blix v0.3.3

All FastAPI endpoint I/O is typed through Pydantic models defined here.
Nothing in this module imports from core/reflection/knowledge — it is a
pure data-shape layer that routers and tests can import without
triggering BlixContext construction.

Python 3.10 compatible.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ===========================================================================
# Chat
# ===========================================================================


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=16_000, description="User message text.")
    session_id: Optional[str] = Field(default=None, description="Optional session label for tracking.")


class ChatResponse(BaseModel):
    reply: str
    session_id: Optional[str] = None
    memory_id: Optional[int] = None


# ===========================================================================
# Memory
# ===========================================================================


class MemoryItem(BaseModel):
    id: int
    input: str
    output: str
    timestamp: datetime
    topics: list[str] = []
    importance: Optional[float] = None
    extracted_facts: list[str] = []
    lifecycle_state: str = "active"


class MemoryListResponse(BaseModel):
    memories: list[MemoryItem]
    total: int
    page: int
    page_size: int


class MemorySearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=10, ge=1, le=50)


class MemorySearchResponse(BaseModel):
    query: str
    results: list[MemoryItem]


# ===========================================================================
# Knowledge (canonical facts + synthesis)
# ===========================================================================


class CanonicalFactItem(BaseModel):
    fact_id: str
    fact: str
    confidence: float
    evidence_count: int
    topic: str
    variants: list[str] = []


class CanonicalFactsResponse(BaseModel):
    facts: list[CanonicalFactItem]
    total: int


class SynthesisRequest(BaseModel):
    source_memory_ids: list[int] = Field(default_factory=list)
    include_projects: bool = True
    include_facts: bool = True
    include_insights: bool = False


class KnowledgeReportItem(BaseModel):
    report_id: str
    title: str
    narrative: str
    key_points: list[str]
    topics: list[str]
    created_at: str


class KnowledgeReportsResponse(BaseModel):
    reports: list[KnowledgeReportItem]
    total: int


# ===========================================================================
# Reflection + Insights
# ===========================================================================


class InsightItem(BaseModel):
    insight: str
    confidence: float
    category: str = "trend"
    evidence: list[str] = []
    recommendation: str = ""
    created_at: str = ""


class InsightsResponse(BaseModel):
    insights: list[InsightItem]
    total: int


class ReflectionRunRequest(BaseModel):
    scope: str = Field(default="behavior", description="session|daily|weekly|project|behavior|learning")
    scope_ref: str = Field(default="", description="Scope identifier (e.g. project name, date).")
    material: Optional[str] = Field(default=None, description="Text to reflect on. If None, auto-collects from recent memories.")


class ReflectionRunResponse(BaseModel):
    scope: str
    scope_ref: str
    insights: list[InsightItem]


# ===========================================================================
# Graph
# ===========================================================================


class GraphNodeItem(BaseModel):
    id: str
    kind: str
    label: str
    aliases: list[str] = []


class GraphEdgeItem(BaseModel):
    from_id: str
    relation: str
    to_id: str
    confidence: float


class GraphStatsResponse(BaseModel):
    node_count: int
    edge_count: int
    nodes: list[GraphNodeItem]
    edges: list[GraphEdgeItem]


class GraphNeighboursResponse(BaseModel):
    node_id: str
    neighbours: list[dict]


class GraphPathResponse(BaseModel):
    from_id: str
    to_id: str
    path: Optional[dict] = None
    hops: Optional[int] = None


class UpsertRelationRequest(BaseModel):
    from_label: str
    from_kind: str = "topic"
    relation: str
    to_label: str
    to_kind: str = "topic"
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


# ===========================================================================
# Documents
# ===========================================================================


class DocumentUploadResponse(BaseModel):
    doc_id: str
    title: str
    format: str
    summary: str
    key_findings: list[str]
    concepts: list[str]
    related_topics: list[str]
    chunk_count: int
    raw_text_length: int


# ===========================================================================
# Stats / Dashboard
# ===========================================================================


class StatsResponse(BaseModel):
    memory_count: int
    embedding_index_size: int
    knowledge_facts: int
    projects: int
    projects_at_risk: int
    graph_nodes: int
    graph_edges: int
    goals: int
    active_goals: int
    insights: int
    reflection_records: int
    knowledge_reports: int
    semantic_clusters: int
    lifecycle_state_counts: dict[str, int]
    contradictions_unresolved: int
    session_count: int
    daily_summaries: int
    weekly_summaries: int
    background: dict


# ===========================================================================
# MQL
# ===========================================================================


class MQLRequest(BaseModel):
    command: str = Field(..., min_length=1, max_length=500)


class MQLResponse(BaseModel):
    command: str
    matched: bool
    text: str
    data: list = []


# ===========================================================================
# Goals
# ===========================================================================


class GoalItem(BaseModel):
    goal_id: str
    title: str
    description: str
    status: str
    priority: int
    progress: int
    related_project: str
    blockers: list[str] = []
    milestones: list[dict] = []
    tasks: list[dict] = []


class CreateGoalRequest(BaseModel):
    title: str = Field(..., min_length=1)
    description: str = ""
    priority: int = Field(default=3, ge=1, le=5)
    related_project: str = ""


class GoalProgressRequest(BaseModel):
    progress: int = Field(..., ge=0, le=100)


class GoalBlockerRequest(BaseModel):
    description: str = Field(..., min_length=1)


class GoalMilestoneRequest(BaseModel):
    title: str = Field(..., min_length=1)


class GoalListResponse(BaseModel):
    goals: list[GoalItem]
    total: int


# ===========================================================================
# Error
# ===========================================================================


class ErrorResponse(BaseModel):
    detail: str
    code: Optional[str] = None
