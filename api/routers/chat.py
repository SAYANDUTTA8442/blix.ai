"""
/chat router — Blix v0.3.3

Endpoints
---------
POST /chat          — single-turn chat, returns full reply
POST /chat/stream   — streaming SSE (server-sent events) reply
POST /chat/mql      — route an MQL command directly
"""

from __future__ import annotations

import asyncio
import json
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from api.models import (
    ChatRequest, ChatResponse, MQLRequest, MQLResponse,
)
from api.deps import get_context
from api.context import BlixContext

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("", response_model=ChatResponse, summary="Single-turn chat")
async def chat(req: ChatRequest, ctx: BlixContext = Depends(get_context)) -> ChatResponse:
    """
    Send a message to Blix and receive a reply.

    Memory extraction, profile updates, and graph updates are processed
    in the background — this endpoint returns as soon as the LLM reply is ready.
    """
    try:
        # TutorAgent.chat() is synchronous (LLM call); run in thread pool
        # so we don't block the event loop.
        loop = asyncio.get_event_loop()
        reply = await loop.run_in_executor(None, ctx.agent.chat, req.message)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    # Retrieve the last-saved memory id for the caller's reference
    memories = ctx.memory_manager.get_all_memories()
    last_id = memories[-1].id if memories else None

    return ChatResponse(reply=reply, session_id=req.session_id, memory_id=last_id)


@router.post("/stream", summary="Streaming chat (SSE)")
async def chat_stream(
    req: ChatRequest,
    ctx: BlixContext = Depends(get_context),
) -> StreamingResponse:
    """
    Streaming variant: emits the reply token-by-token as Server-Sent Events.

    The LLM layer does not yet expose a true token-stream; we simulate
    streaming by chunking the completed reply into 10-character pieces so
    clients can render progressively. Replace with a native streaming
    provider in v0.4.

    Each SSE frame is:
        data: {"token": "..."}\n\n

    A final frame signals completion:
        data: {"done": true, "memory_id": N}\n\n
    """

    async def event_stream() -> AsyncGenerator[str, None]:
        try:
            loop = asyncio.get_event_loop()
            reply = await loop.run_in_executor(None, ctx.agent.chat, req.message)
        except RuntimeError as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
            return

        memories = ctx.memory_manager.get_all_memories()
        last_id = memories[-1].id if memories else None

        chunk_size = 10
        for i in range(0, len(reply), chunk_size):
            token = reply[i : i + chunk_size]
            yield f"data: {json.dumps({'token': token})}\n\n"
            await asyncio.sleep(0)  # yield control to event loop

        yield f"data: {json.dumps({'done': True, 'memory_id': last_id})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/mql", response_model=MQLResponse, summary="Run an MQL command")
async def mql_command(req: MQLRequest, ctx: BlixContext = Depends(get_context)) -> MQLResponse:
    """
    Execute a Memory Query Language command, e.g.:

        show active goals
        show project Blix
        show memories about transformers
        show contradictions
    """
    result = ctx.mql.run(req.command)
    return MQLResponse(
        command=req.command,
        matched=result.matched,
        text=result.text,
        data=result.data if isinstance(result.data, list) else [],
    )
