"""
/simulation router — Blix v0.3.14
POST /simulation/trajectory/build    — build a multi-step trajectory
GET  /simulation/trajectory/all      — list in-memory trajectories
GET  /simulation/trajectory/deepest  — get deepest trajectory
POST /simulation/trajectory/clear    — clear all trajectories
"""
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from api.context import BlixContext
from api.deps import get_context

router = APIRouter(prefix="/simulation", tags=["Simulation"])

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

class TrajectoryStepInput(BaseModel):
    action: str = Field(..., min_length=1, max_length=200)
    resulting_state: LatentStateInput
    predicted_value_delta: float = Field(default=0.0)
    label: str = Field(default="")

class BuildTrajectoryRequest(BaseModel):
    start_state: LatentStateInput
    steps: list[TrajectoryStepInput] = Field(..., min_length=1, max_length=20)
    store_in_graph: bool = Field(default=True)

@router.post("/trajectory/build")
async def build_trajectory(req: BuildTrajectoryRequest, ctx: BlixContext = Depends(get_context)) -> dict:
    from simulation.trajectory_graph import TrajectoryBuilder
    builder = TrajectoryBuilder(req.start_state.to_ls())
    for step in req.steps:
        builder.step(step.action, step.resulting_state.to_ls(),
                     predicted_value_delta=step.predicted_value_delta, label=step.label)
    traj = builder.build()
    if req.store_in_graph:
        ctx.trajectory_graph.add(traj)
    return traj.to_dict()

@router.get("/trajectory/all")
async def list_trajectories(ctx: BlixContext = Depends(get_context)) -> dict:
    trajs = ctx.trajectory_graph.all_trajectories()
    return {"total": len(trajs), "trajectories": [t.to_dict() for t in trajs]}

@router.get("/trajectory/deepest")
async def deepest_trajectory(ctx: BlixContext = Depends(get_context)) -> dict:
    traj = ctx.trajectory_graph.deepest()
    if traj is None:
        return {"available": False}
    return {"available": True, "trajectory": traj.to_dict()}

@router.post("/trajectory/clear")
async def clear_trajectories(ctx: BlixContext = Depends(get_context)) -> dict:
    count_before = ctx.trajectory_graph.count
    ctx.trajectory_graph.clear()
    return {"cleared": count_before}
