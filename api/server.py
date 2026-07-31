"""
FastAPI server — Blix v0.3.3  (Feature 1 flagship)

Creates and returns the FastAPI application with all routers mounted.
Can be run directly:

    uvicorn blix.api.server:app --reload --port 8000

Or imported as a library:

    from api.server import create_app, app

OpenAPI docs:   http://localhost:8000/docs
Redoc:          http://localhost:8000/redoc
Health check:   GET /health
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.context import BlixContext
from api.deps import set_context

log = logging.getLogger(__name__)


def create_app(memory_dir: Path | None = None) -> FastAPI:
    """
    Build and configure the FastAPI application.

    Parameters
    ----------
    memory_dir:
        Override the default ``memory/`` directory (useful for tests).
    """

    # ----------------------------------------------------------------
    # Lifespan: construct BlixContext on startup, shut down cleanly
    # ----------------------------------------------------------------

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncGenerator[None, None]:
        ctx = BlixContext.build(memory_dir)
        set_context(ctx)
        log.info("Blix API started — memory_dir=%s", ctx.memory_dir)
        yield
        ctx.shutdown()
        log.info("Blix API stopped cleanly.")

    # ----------------------------------------------------------------
    # App definition
    # ----------------------------------------------------------------

    application = FastAPI(
        title="Blix — Cognitive Knowledge Platform",
        description=(
            "REST API for Blix v0.3.3: chat, memory, knowledge graph, "
            "reflection, document processing, goal tracking, and dashboard statistics.\n\n"
            "All endpoints are async. Memory extraction and graph updates run in the "
            "background — chat latency is never blocked by post-processing."
        ),
        version="0.3.3",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # ----------------------------------------------------------------
    # CORS (permissive for local dev; tighten for production)
    # ----------------------------------------------------------------

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ----------------------------------------------------------------
    # Routers
    # ----------------------------------------------------------------

    from api.routers.chat import router as chat_router
    from api.routers.memory import router as memory_router
    from api.routers.knowledge import router as knowledge_router
    from api.routers.reflection import router as reflection_router
    from api.routers.graph import router as graph_router
    from api.routers.documents import router as documents_router
    from api.routers.stats_goals import stats_router, goals_router
    from api.routers.reasoning_research import reason_router, research_router
    from api.routers.agent import router as agent_router
    from api.routers.temporal import router as temporal_router
    from api.routers.metacognition import router as metacognition_router
    from api.routers.workspace import router as workspace_router
    from api.routers.ml import router as ml_router
    from api.routers.causality import router as causality_router
    from api.routers.search import router as search_router
    from api.routers.curiosity import router as curiosity_router
    from api.routers.world_model import router as world_model_router
    from api.routers.simulation import router as simulation_router
    from api.routers.agents import router as agents_router
    from api.routers.specialists import router as specialists_router

    application.include_router(chat_router)
    application.include_router(memory_router)
    application.include_router(knowledge_router)
    application.include_router(reflection_router)
    application.include_router(graph_router)
    application.include_router(documents_router)
    application.include_router(stats_router)
    application.include_router(goals_router)
    application.include_router(reason_router)
    application.include_router(research_router)
    application.include_router(agent_router)
    application.include_router(temporal_router)
    application.include_router(metacognition_router)
    application.include_router(workspace_router)
    application.include_router(ml_router)
    application.include_router(causality_router)
    application.include_router(search_router)
    application.include_router(curiosity_router)
    application.include_router(world_model_router)
    application.include_router(simulation_router)
    application.include_router(agents_router)
    application.include_router(specialists_router)

    # ----------------------------------------------------------------
    # Health / root
    # ----------------------------------------------------------------

    @application.get("/health", tags=["System"], summary="Health check")
    async def health() -> dict:
        return {"status": "ok", "version": "0.3.3"}

    @application.get("/", tags=["System"], summary="API info", include_in_schema=False)
    async def root() -> dict:
        return {
            "name": "Blix Cognitive Knowledge Platform",
            "version": "0.3.3",
            "docs": "/docs",
            "health": "/health",
        }

    return application


# ---------------------------------------------------------------------------
# Module-level app instance for uvicorn
# ---------------------------------------------------------------------------

app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
