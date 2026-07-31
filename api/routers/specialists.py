"""
/specialists router — Blix v0.3.14
POST /specialists/consult             — consult all specialists
GET  /specialists/list                — list registered specialists
POST /specialists/{name}/consult      — consult one specific specialist
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from api.context import BlixContext
from api.deps import get_context

router = APIRouter(prefix="/specialists", tags=["Specialists"])

class ConsultRequest(BaseModel):
    topic: str = Field(..., min_length=1, max_length=500)
    context: dict = Field(default_factory=dict)

@router.post("/consult")
async def consult_all(req: ConsultRequest, ctx: BlixContext = Depends(get_context)) -> dict:
    result = ctx.specialist_consensus.decide(req.topic, **req.context)
    opinions = [
        {"specialist": o.specialist, "verdict": o.verdict,
         "confidence": round(o.confidence, 4), "rationale": o.rationale}
        for o in result.opinions
    ]
    return {
        "topic": result.topic,
        "majority_verdict": result.majority_verdict,
        "agreement_ratio": round(result.agreement_ratio, 4),
        "mean_confidence": round(result.mean_confidence, 4),
        "opinions": opinions,
        "is_contested": result.is_contested,
    }

@router.get("/list")
async def list_specialists(ctx: BlixContext = Depends(get_context)) -> dict:
    names = ctx.specialist_consensus.registered_names()
    return {"total": len(names), "specialists": names}

@router.post("/{specialist_name}/consult")
async def consult_one(
    specialist_name: str, req: ConsultRequest,
    ctx: BlixContext = Depends(get_context),
) -> dict:
    opinions = ctx.specialist_consensus.consult_all(req.topic, **req.context)
    matching = [o for o in opinions if o.specialist == specialist_name]
    if not matching:
        raise HTTPException(status_code=404, detail=f"Specialist '{specialist_name}' not found.")
    o = matching[0]
    return {"specialist": o.specialist, "verdict": o.verdict,
            "confidence": round(o.confidence, 4), "rationale": o.rationale}
