from __future__ import annotations

from typing import List, Optional, Literal

from pydantic import BaseModel, Field


class Position(BaseModel):
    x: int = Field(ge=0)
    y: int = Field(ge=0)


class GridSize(BaseModel):
    width: int = Field(ge=1)
    height: int = Field(ge=1)


class AgentStyle(BaseModel):
    agent_id: str
    color_hex: str


class ViewShape(BaseModel):
    kind: Literal["square", "cross"] = "square"
    radius: int = Field(ge=0)


class EpisodeMeta(BaseModel):
    grid_size: GridSize
    goal: Position
    walls: List[Position] = Field(default_factory=list)
    view: ViewShape
    gradient_mode: Literal["bfs", "manhattan"] = "bfs"
    title: Optional[str] = None
    agent_styles: List[AgentStyle] = Field(default_factory=list)


class AgentState(BaseModel):
    agent_id: str
    pos: Position
    orientation: Optional[Literal["N", "E", "S", "W"]] = None
    action: Optional[
        Literal["MOVE_N", "MOVE_E", "MOVE_S", "MOVE_W", "STAY", "COMMUNICATE", "MARK"]
    ] = None
    status: Literal["ACTIVE", "FINISHED"] = Field(
        "ACTIVE",
        description="Lifecycle flag so renderers can hide finished agents without overloading the action field.",
    )


class Frame(BaseModel):
    t: int = Field(ge=0)
    agents: List[AgentState]


class EpisodeLog(BaseModel):
    meta: EpisodeMeta
    frames: List[Frame]
