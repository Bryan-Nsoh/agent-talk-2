"""Portable Pydantic schemas that describe the environment<>agent contract."""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class Direction(str, Enum):
    """Cardinal directions used for motion and orientation."""

    N = "N"
    E = "E"
    S = "S"
    W = "W"


class Octant(str, Enum):
    """Coarse bearings returned by the goal sensor and shared via messages."""

    N = "N"
    NE = "NE"
    E = "E"
    SE = "SE"
    S = "S"
    SW = "SW"
    W = "W"
    NW = "NW"


class StrengthBucket(str, Enum):
    """Quantised indication of signal strength toward the goal."""

    FAR = "FAR"
    MID = "MID"
    NEAR = "NEAR"


class BlockReason(str, Enum):
    """Minimal taxonomy for why a cell should be avoided."""

    WALL = "WALL"
    AGENT = "AGENT"
    DEAD_END = "DEAD_END"
    CONGESTION = "CONGESTION"


class ActionKind(str, Enum):
    """Single atomic choice available to the agent each turn."""

    MOVE = "MOVE"
    STAY = "STAY"
    COMMUNICATE = "COMMUNICATE"
    MARK = "MARK"


class GoalSensorMode(str, Enum):
    """Which sensing modality produced the reading."""

    BEARING = "BEARING"
    SCALAR_GRADIENT = "SCALAR_GRADIENT"
    SPARSE_PINGS = "SPARSE_PINGS"


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


class GridSize(BaseModel):
    """Global grid dimensions."""

    width: int = Field(ge=1, description="Number of columns along +X.")
    height: int = Field(ge=1, description="Number of rows along +Y.")


class Position(BaseModel):
    """Absolute integer coordinates."""

    x: int = Field(ge=0, description="Column index, 0-based from left.")
    y: int = Field(ge=0, description="Row index, 0-based from top.")


class RelativeOffset(BaseModel):
    """Relative displacement in the agent's local frame."""

    dx: int = Field(description="Offset along +X relative to the agent.")
    dy: int = Field(description="Offset along +Y relative to the agent.")


# ---------------------------------------------------------------------------
# Goal sensor readings
# ---------------------------------------------------------------------------


class GoalSensorBearing(BaseModel):
    """Noisy coarse bearing with optional strength buckets."""

    mode: Literal["BEARING"] = "BEARING"
    bearing: Optional[Octant] = Field(
        default=None, description="Octant pointing toward the goal when available."
    )
    strength: Optional[StrengthBucket] = Field(
        default=None, description="Optional monotonic indicator of proximity."
    )
    available: bool = Field(description="False when the sensor dropped out this turn.")


class GoalSensorScalar(BaseModel):
    """Quantised scalar gradient measurements."""

    mode: Literal["SCALAR_GRADIENT"] = "SCALAR_GRADIENT"
    value_bin: Optional[Literal["LOW", "MID", "HIGH"]] = Field(
        default=None, description="Scalar value bucket at the current cell."
    )
    north_gt_south: Optional[bool] = Field(
        default=None, description="True if value at y-1 > y+1 in the local patch."
    )
    east_gt_west: Optional[bool] = Field(
        default=None, description="True if value at x+1 > x-1 in the local patch."
    )
    available: bool = Field(description="False when no reading is provided.")


class GoalSensorPing(BaseModel):
    """Sparse radio ping with coarse arrival bearing."""

    mode: Literal["SPARSE_PINGS"] = "SPARSE_PINGS"
    heard: bool = Field(description="True if a ping was detected this turn.")
    approx_bearing: Optional[Octant] = Field(
        default=None, description="Coarse direction from which the ping arrived."
    )
    jitter_ms: Optional[int] = Field(
        default=None, description="Optional jitter bucket to support triangulation."
    )


GoalSensorReading = Union[GoalSensorBearing, GoalSensorScalar, GoalSensorPing]


# ---------------------------------------------------------------------------
# Artifact sensing
# ---------------------------------------------------------------------------


class ArtifactTrail(BaseModel):
    """Breadcrumb marking the direction traversed through this cell."""

    kind: Literal["TRAIL"] = "TRAIL"
    dir_entered: Direction = Field(description="Direction used to enter the cell.")
    ttl_remaining: int = Field(ge=0, description="Turns before evaporation.")


class ArtifactBearingSample(BaseModel):
    """Stored bearing measurement for delayed triangulation."""

    kind: Literal["BEARING_SAMPLE"] = "BEARING_SAMPLE"
    bearing: Octant = Field(description="Bearing recorded at this cell.")
    strength: Optional[StrengthBucket] = Field(default=None)
    measured_at: Position = Field(description="Absolute position where sampled.")
    turn_index: int = Field(ge=0, description="Turn when the reading was captured.")
    ttl_remaining: int = Field(ge=0, description="Turns before evaporation.")


class ArtifactNoGo(BaseModel):
    """Marker indicating the cell should be avoided."""

    kind: Literal["NO_GO"] = "NO_GO"
    reason: BlockReason = Field(description="Why the cell is undesirable.")
    ttl_remaining: int = Field(ge=0, description="Turns before evaporation.")


class ArtifactClaim(BaseModel):
    """Reservation signalling intent to occupy this cell soon."""

    kind: Literal["CLAIM"] = "CLAIM"
    next_turn_index: int = Field(
        ge=0, description="Turn when the placer intends to occupy the cell."
    )
    ttl_remaining: int = Field(ge=0, description="Turns before evaporation.")


class ArtifactGoalHint(BaseModel):
    """Soft guidance that the goal is nearby."""

    kind: Literal["GOAL_HINT"] = "GOAL_HINT"
    confidence: Literal["LOW", "MED", "HIGH"] = Field(
        description="Belief strength that the goal is near."
    )
    ttl_remaining: int = Field(ge=0, description="Turns before evaporation.")


PlacedArtifact = Union[
    ArtifactTrail,
    ArtifactBearingSample,
    ArtifactNoGo,
    ArtifactClaim,
    ArtifactGoalHint,
]


# ---------------------------------------------------------------------------
# Messaging
# ---------------------------------------------------------------------------


class BaseMsg(BaseModel):
    """Common envelope for all radio messages."""

    kind: str = Field(description="Discriminator for the message type.")
    sender_id: str = Field(description="Ephemeral agent id stable for this episode.")
    seq: int = Field(ge=0, description="Monotonically increasing per-sender sequence.")


class MsgHello(BaseMsg):
    """Presence handshake."""

    kind: Literal["HELLO"] = "HELLO"


class MsgHere(BaseMsg):
    """Share absolute position and orientation."""

    kind: Literal["HERE"] = "HERE"
    pos: Position = Field(description="Absolute position of the sender.")
    orientation: Direction = Field(description="Orientation of the sender.")


class MsgIntent(BaseMsg):
    """Announce the next move or stay intent."""

    kind: Literal["INTENT"] = "INTENT"
    next_action: Literal["MOVE_N", "MOVE_E", "MOVE_S", "MOVE_W", "STAY"] = Field(
        description="Planned action for collision avoidance."
    )


class MsgSense(BaseMsg):
    """Share a recent goal sensor reading."""

    kind: Literal["SENSE"] = "SENSE"
    at: Position = Field(description="Position where the reading was taken.")
    mode: GoalSensorMode = Field(description="Sensor modality used.")
    bearing: Optional[Octant] = Field(default=None)
    strength: Optional[StrengthBucket] = Field(default=None)
    value_bin: Optional[Literal["LOW", "MID", "HIGH"]] = Field(default=None)
    approx_bearing: Optional[Octant] = Field(default=None)


class MsgBlocked(BaseMsg):
    """Report a blocked cell so teammates can avoid it."""

    kind: Literal["BLOCKED"] = "BLOCKED"
    where: Position = Field(description="Absolute position of the blockage.")
    reason: BlockReason = Field(description="Why the cell is blocked.")


class MsgRequest(BaseMsg):
    """Minimal negotiation primitives."""

    kind: Literal["REQUEST"] = "REQUEST"
    req: Literal["YIELD", "GUIDE", "MEET"] = Field(description="Type of request.")
    target: Optional[Position] = Field(
        default=None, description="Optional absolute position relevant to the request."
    )


class MsgAck(BaseMsg):
    """Acknowledge receipt of a message."""

    kind: Literal["ACK"] = "ACK"
    ack_seq: int = Field(ge=0, description="Sequence number being acknowledged.")
    from_sender_id: str = Field(description="Sender id of the acknowledged message.")


class MsgBye(BaseMsg):
    """Signal that the sender is done moving."""

    kind: Literal["BYE"] = "BYE"


class MsgMarkInfo(BaseMsg):
    """Announce that an artifact was placed."""

    kind: Literal["MARK_INFO"] = "MARK_INFO"
    placed: PlacedArtifact = Field(
        description="Details about the artifact the sender just created."
    )


class MsgChat(BaseMsg):
    """Free-form message for natural-language coordination."""

    kind: Literal["CHAT"] = "CHAT"
    text: str = Field(
        description="Short sentence (<=96 chars) intended for nearby teammates.",
        max_length=96,
        min_length=1,
    )


OutgoingMessage = Union[
    MsgHello,
    MsgHere,
    MsgIntent,
    MsgSense,
    MsgBlocked,
    MsgRequest,
    MsgAck,
    MsgBye,
    MsgMarkInfo,
    MsgChat,
]


class ReceivedMessage(BaseModel):
    """Message delivered to the agent this turn."""

    envelope: OutgoingMessage = Field(description="Structured message payload.")
    hop_distance: int = Field(
        ge=0, description="Manhattan distance between sender and receiver."
    )
    age: int = Field(ge=0, description="Turns elapsed since the message was sent.")


# ---------------------------------------------------------------------------
# Local view and agent state
# ---------------------------------------------------------------------------


class LocalPatch(BaseModel):
    """Dense egocentric patch for the agent's current FOV."""

    radius: int = Field(ge=1, description="Visibility radius R. Patch size is 2R+1.")
    top_left_abs: Position = Field(
        description="Absolute coordinates of the top-left cell in the patch."
    )
    rows: List[str] = Field(
        description=(
            "Each string has length 2R+1 using characters: '.'=EMPTY, '#'=WALL, "
            "'G'=GOAL, 'A'=AGENT, '*'=ARTIFACT."
        )
    )


class NeighborSummary(BaseModel):
    """Structured summary for neighbors currently visible."""

    agent_id: str = Field(description="Ephemeral id of the neighbor.")
    abs_pos: Position = Field(description="Absolute position of the neighbor.")
    rel: RelativeOffset = Field(description="Offset relative to the observing agent.")


class CommLimits(BaseModel):
    """Communication limits for the current episode."""

    range: int = Field(ge=0, description="Inclusive Manhattan delivery radius r.")
    max_outbound_per_turn: int = Field(
        ge=0, description="Maximum messages this agent may send per turn."
    )
    max_payload_chars: int = Field(
        ge=1, description="Upper bound on serialized message size (advisory)."
    )


class MarkLimits(BaseModel):
    """Artifact placement limits."""

    max_ttl: int = Field(ge=1, description="Environment cap on artifact TTL.")
    allow_mark_info_broadcast: bool = Field(
        description="True if MARK_INFO messages are permitted."
    )


class AgentSelf(BaseModel):
    """Ego information for the observing agent."""

    agent_id: str = Field(description="Ephemeral id assigned at episode start.")
    abs_pos: Position = Field(description="Current absolute position.")
    orientation: Direction = Field(description="Current facing direction.")


class AdjacentState(str, Enum):
    """Categorisation for cells adjacent to the agent."""

    FREE = "FREE"
    WALL = "WALL"
    OUT_OF_BOUNDS = "OUT_OF_BOUNDS"
    AGENT = "AGENT"
    GOAL = "GOAL"
    CONTENDED = "CONTENDED"
    NO_GO = "NO_GO"


class AdjacentCell(BaseModel):
    """Cardinal neighbor state to remove ambiguity from ASCII patches."""

    dir: Literal["N", "E", "S", "W"] = Field(description="Cardinal direction from the current cell.")
    state: AdjacentState = Field(description="Occupancy classification for the neighbor.")


class MoveOutcome(str, Enum):
    """Execution result for the agent's previous action."""

    OK = "OK"
    BLOCK_WALL = "BLOCK_WALL"
    BLOCK_OOB = "BLOCK_OOB"
    BLOCK_AGENT = "BLOCK_AGENT"
    SWAP_CONFLICT = "SWAP_CONFLICT"
    YIELD = "YIELD"
    FINISHED = "FINISHED"


class Observation(BaseModel):
    """Full observation object passed to the LLM."""

    protocol_version: str = Field(description="Interface version for compatibility.")
    turn_index: int = Field(ge=0, description="Current turn, zero-based.")
    max_turns: int = Field(ge=1, description="Turn budget for the episode.")
    grid_size: GridSize = Field(description="Global grid dimensions.")
    self_state: AgentSelf = Field(description="Ego state of the agent.")
    local_patch: LocalPatch = Field(description="Dense egocentric patch.")
    neighbors_in_view: List[NeighborSummary] = Field(
        description="Neighbors observed within the local patch."
    )
    artifacts_in_view: List[PlacedArtifact] = Field(
        description="Artifacts detected within the patch."
    )
    inbox: List[ReceivedMessage] = Field(
        description="Messages delivered during this turn."
    )
    adjacent: List[AdjacentCell] = Field(description="Passability summary for the N/E/S/W neighboring cells.")
    recent_positions: List[Position] = Field(
        description="Most recent absolute positions occupied by the agent (newest first)."
    )
    comm_limits: CommLimits = Field(description="Communication constraints.")
    mark_limits: MarkLimits = Field(description="Artifact placement constraints.")
    goal_sensor: GoalSensorReading = Field(
        description="Sensing information pointing toward the goal."
    )
    last_move_outcome: MoveOutcome = Field(
        description="Outcome of the agent's previous action."
    )
    contended_neighbors: int = Field(
        ge=0,
        le=15,
        description="Bitmask (NESW) of neighbors targeted by multiple agents on the previous turn.",
    )
    history: List["TurnHistory"] = Field(
        default_factory=list,
        description="Chronological record (up to 5) of this agent's recent turns.",
    )


# ---------------------------------------------------------------------------
# Agent actions and decision
# ---------------------------------------------------------------------------


class MoveAction(BaseModel):
    """Move one step along a cardinal axis."""

    kind: Literal["MOVE"] = "MOVE"
    direction: Direction = Field(description="Direction of movement.")


class StayAction(BaseModel):
    """Remain stationary this turn."""

    kind: Literal["STAY"] = "STAY"


class CommunicateAction(BaseModel):
    """Broadcast exactly one structured message."""

    kind: Literal["COMMUNICATE"] = "COMMUNICATE"
    message: OutgoingMessage = Field(description="Message to broadcast.")


class MarkAction(BaseModel):
    """Drop an artifact at the current cell."""

    kind: Literal["MARK"] = "MARK"
    placement: PlacedArtifact = Field(description="Artifact definition.")


AgentAction = Union[MoveAction, StayAction, CommunicateAction, MarkAction]


class Decision(BaseModel):
    """Structured response expected from the LLM each turn."""

    action: AgentAction = Field(description="Action to execute this turn.")
    comment: Optional[str] = Field(
        default=None, description="Optional debugging note (ignored in scoring)."
    )


class MessageBrief(BaseModel):
    """Compact summary of a message for inclusion in turn history."""

    kind: str = Field(description="Message kind discriminator.")
    details: Optional[str] = Field(default=None, description="Optional short description.")
    sender: Optional[str] = Field(default=None, description="Sender id if relevant.")
    hop: Optional[int] = Field(default=None, description="Hop distance for received messages.")
    age: Optional[int] = Field(default=None, description="Age in turns for received messages.")


class TurnHistory(BaseModel):
    """Summary of the agent's own previous turns."""

    turn_index: int = Field(ge=0, description="Turn number this event represents.")
    intent: Literal["MOVE_N", "MOVE_E", "MOVE_S", "MOVE_W", "STAY", "COMMUNICATE", "MARK"] = Field(
        description="Intent chosen on that turn."
    )
    outcome: MoveOutcome = Field(description="Result of executing the intent.")
    delta: Literal["CLOSER", "SAME", "FARTHER"] = Field(description="Change in Manhattan distance to the goal.")
    loop: int = Field(ge=0, le=9, description="Consecutive turns without progress (capped at 9).")
    peer_bits: str = Field(
        description="Compact encoding of nearby agents/intents (e.g., 'N1E0S0W0|intent:E')."
    )
    note: Optional[str] = Field(
        default=None,
        max_length=12,
        description="Short status token (e.g., AVOID_LOOP, INTENT_SEEN, TRAFFIC_CONE).",
    )
