"""
/world_model router — Blix v0.3.14
POST /world_model/predict      — predict next latent state
POST /world_model/value        — score a latent state
POST /world_model/scenario/rank — rank candidate scenarios
GET  /world_model/status       — value network training status
"""
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from api.context import BlixContext
from api.deps import get_context

router = APIRouter(prefix="/world_model", tags=["World Model"])

class LatentStateInput(BaseModel):
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    complexity: float = Field(default=0.5, ge=0.0, le=1.0)
    risk: float = Field(default=0.0, ge=0.0, le=1.0)
    capability_estimate: float = Field(default=0.5, ge=0.0, le=1.0)
    recent_failure_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    dependency_density: float = Field(default=0.0, ge=0.0, le=1.0)

    def to_ls(self):
        from world_model.latent_world_model import LatentState
        return LatentState(**self.model_dump())

class ScenarioInput(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    description: str = Field(default="")
    state: LatentStateInput

@router.post("/predict")
async def predict_state(state: LatentStateInput, ctx: BlixContext = Depends(get_context)) -> dict:
    pred = ctx.latent_world_model.predict(state.to_ls())
    return pred.to_dict()

@router.post("/value")
async def score_state(state: LatentStateInput, ctx: BlixContext = Depends(get_context)) -> dict:
    v = ctx.value_network.value(state.to_ls())
    return {"value": round(v, 6), "is_trained": ctx.value_network.is_trained,
            "sample_count": ctx.value_network.sample_count}

@router.post("/scenario/rank")
async def rank_scenarios(scenarios: list[ScenarioInput], ctx: BlixContext = Depends(get_context)) -> dict:
    from world_model.scenario_ranker import Scenario
    objs = [Scenario(name=s.name, description=s.description, state=s.state.to_ls()) for s in scenarios]
    ranked = ctx.scenario_ranker.rank(objs)
    return {"total": len(ranked), "ranked": [r.to_dict() for r in ranked]}

@router.get("/status")
async def model_status(ctx: BlixContext = Depends(get_context)) -> dict:
    return {
        "value_network_trained": ctx.value_network.is_trained,
        "value_network_samples": ctx.value_network.sample_count,
        "world_model_trained": ctx.latent_world_model.is_trained,
        "world_model_samples": ctx.latent_world_model.sample_count,
    }
