"""Portable Pydantic schemas for the map‑sharing, no‑comm environment."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Type, Any

from pydantic import BaseModel, Field, create_model


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class Direction(str, Enum):
    N = "N"
    E = "E"
    S = "S"
    W = "W"


class AdjacentState(str, Enum):
    FREE = "FREE"
    WALL = "WALL"
    GOAL = "GOAL"
    AGENT = "AGENT"
    OUT_OF_BOUNDS = "OUT_OF_BOUNDS"


class MoveOutcome(str, Enum):
    OK = "OK"
    FINISHED = "FINISHED"
    BLOCK_WALL = "BLOCK_WALL"
    BLOCK_AGENT = "BLOCK_AGENT"
    SWAP_CONFLICT = "SWAP_CONFLICT"


# ---------------------------------------------------------------------------
# Core geometry
# ---------------------------------------------------------------------------


class Grid(BaseModel):
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    rows: List[List[str]] = Field(description="rows[y][x] -> symbol")


class Position(BaseModel):
    x: int = Field(ge=0)
    y: int = Field(ge=0)


# ---------------------------------------------------------------------------
# Observation fields
# ---------------------------------------------------------------------------


class AgentSelf(BaseModel):
    agent_id: str
    pos: Position


class NeighborInView(BaseModel):
    agent_id: str
    pos: Position


class AdjacentCell(BaseModel):
    dir: Direction
    state: AdjacentState


class LastResult(BaseModel):
    kind: MoveOutcome
    cell: Optional[Position] = None
    opponents: List[str] = Field(default_factory=list)


class Observation(BaseModel):
    protocol_version: str
    turn_index: int
    max_turns: int
    grid: Grid
    legend: Dict[str, str]
    self: AgentSelf
    neighbors_in_view: List[NeighborInView]
    adjacent: List[AdjacentCell]
    adjacent_frontiers: List[Position]
    goal_known: bool
    goal_pos: Optional[Position] = None
    last_result: LastResult
    map_sharing: str


# ---------------------------------------------------------------------------
# Decision schema
# ---------------------------------------------------------------------------


class MoveAction(BaseModel):
    kind: str = Field(default="MOVE", pattern="^MOVE$")
    direction: Direction


class StayAction(BaseModel):
    kind: str = Field(default="STAY", pattern="^STAY$")


AgentAction = MoveAction | StayAction


class Decision(BaseModel):
    action: AgentAction
    comment: str


# ---------------------------------------------------------------------------
# Strategy capability helpers (kept minimal for compatibility)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StrategyCapabilities:
    key: str
    message_types: List[Type[BaseModel]]
    allow_comm: bool
    action_kinds: List[str]


def resolve_strategy_capabilities(strategy: str) -> StrategyCapabilities:
    key = (strategy or "none").lower()
    return StrategyCapabilities(
        key=key,
        message_types=[],
        allow_comm=False,
        action_kinds=["MOVE", "STAY"],
    )


def _union_type(type_list: List[Type[Any]]) -> Type[Any]:
    union: Type[Any] = type_list[0]
    for typ in type_list[1:]:
        union = union | typ  # type: ignore[operator]
    return union


def build_decision_model(strategy: str) -> Type[BaseModel]:
    capabilities = resolve_strategy_capabilities(strategy)
    action_union = _union_type([MoveAction, StayAction])
    model_name = f"Decision_{capabilities.key}_nocomm"
    return create_model(
        model_name,
        action=(action_union, ...),
        comment=(str, ...),
    )


def coerce_decision(wire_decision: BaseModel) -> Decision:
    if isinstance(wire_decision, Decision):
        return wire_decision
    payload = wire_decision.model_dump()
    return Decision.model_validate(payload)
