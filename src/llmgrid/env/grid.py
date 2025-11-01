"""Environment primitives for the partially observable grid world."""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Optional, Tuple

from llmgrid.schema import (
    AgentSelf,
    ArtifactClaim,
    ArtifactBearingSample,
    ArtifactGoalHint,
    ArtifactNoGo,
    ArtifactTrail,
    AdjacentCell,
    AdjacentState,
    BlockReason,
    CommLimits,
    Direction,
    GoalSensorBearing,
    GoalSensorReading,
    GridSize,
    LocalPatch,
    MarkLimits,
    MoveOutcome,
    MsgHere,
    MsgIntent,
    MsgMarkInfo,
    MsgSense,
    NeighborSummary,
    Observation,
    Octant,
    PlacedArtifact,
    Position,
    ReceivedMessage,
    RelativeOffset,
    StrengthBucket,
    TurnHistory,
)

TileChar = str  # ".", "#", "G", "A", "*"

TRAFFIC_CONE_TTL = 3


@dataclass
class MoveResult:
    final: Tuple[int, int]
    outcome: MoveOutcome
    target: Optional[Tuple[int, int]]
    opponents: List[str]
    cause_cell: Optional[Tuple[int, int]]


def _direction_delta(direction: Direction) -> Tuple[int, int]:
    return {
        Direction.N: (0, -1),
        Direction.E: (1, 0),
        Direction.S: (0, 1),
        Direction.W: (-1, 0),
    }[direction]


class GridWorld:
    """Grid-based environment with synchronous turns."""

    def __init__(
        self,
        width: int,
        height: int,
        obstacles: Iterable[Position],
        goal: Position,
        *,
        seed: int = 0,
        bearing_flip_p: float = 0.15,
        bearing_drop_p: float = 0.10,
        bearing_bias_seed: Optional[int] = None,
        bearing_bias_p: float = 0.0,
        bearing_bias_wall_bonus: float = 0.0,
        history_limit: int = 5,
    ) -> None:
        self.size = GridSize(width=width, height=height)
        self.goal = goal
        self.walls = {(p.x, p.y) for p in obstacles}
        self.rng = random.Random(seed)
        self.bearing_flip_p = bearing_flip_p
        self.bearing_drop_p = bearing_drop_p
        self.bearing_bias_seed = bearing_bias_seed
        self.bearing_bias_p = bearing_bias_p
        self.bearing_bias_wall_bonus = bearing_bias_wall_bonus
        self.history_limit = max(1, history_limit)

        self.occupancy: Dict[str, Tuple[int, int]] = {}
        self.orientation: Dict[str, Direction] = {}
        self.inboxes: Dict[str, List[ReceivedMessage]] = {}
        self.artifacts: Dict[Tuple[int, int], PlacedArtifact] = {}
        self.finished_agents: Dict[str, bool] = {}
        self.position_history: Dict[str, List[Tuple[int, int]]] = {}
        self.turn_history: Dict[str, Deque[dict]] = {}
        self.last_move_outcome: Dict[str, MoveOutcome] = {}
        self.loop_counters: Dict[str, int] = {}
        self.last_goal_distance: Dict[str, int] = {}
        self.last_intent_target: Dict[str, Optional[Tuple[int, int]]] = {}
        self.contended_neighbors: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Agent placement and utility helpers
    # ------------------------------------------------------------------

    def add_agent(self, agent_id: str, pos: Position, orientation: Direction) -> None:
        key = (pos.x, pos.y)
        if key in self.walls:
            raise ValueError("Cannot spawn agent on a wall.")
        if key in self.occupancy.values():
            raise ValueError("Spawn cell already occupied.")
        if not self._in_bounds(*key):
            raise ValueError("Spawn position out of bounds.")
        self.occupancy[agent_id] = key
        self.orientation[agent_id] = orientation
        self.inboxes[agent_id] = []
        self.finished_agents[agent_id] = False
        self.position_history[agent_id] = [key]
        self.turn_history[agent_id] = deque(maxlen=self.history_limit)
        self.last_move_outcome[agent_id] = MoveOutcome.OK
        self.loop_counters[agent_id] = 0
        self.last_goal_distance[agent_id] = abs(self.goal.x - pos.x) + abs(self.goal.y - pos.y)
        self.last_intent_target[agent_id] = None
        self.contended_neighbors[agent_id] = 0

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.size.width and 0 <= y < self.size.height

    def _passable(self, x: int, y: int) -> bool:
        return (x, y) not in self.walls

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def build_observation(
        self,
        agent_id: str,
        *,
        turn_index: int,
        max_turns: int,
        visibility_radius: int,
        radio_range: int,
    ) -> Observation:
        ax, ay = self.occupancy[agent_id]
        local_patch = self._render_patch(ax, ay, visibility_radius)
        neighbors = self._neighbors_in_view(agent_id, visibility_radius)
        artifacts = self._artifacts_in_view(ax, ay, visibility_radius)
        inbox = list(self.inboxes.get(agent_id, []))
        self.inboxes[agent_id] = []

        obs = Observation(
            protocol_version="1.0.0",
            turn_index=turn_index,
            max_turns=max_turns,
            grid_size=self.size,
            self_state=AgentSelf(
                agent_id=agent_id,
                abs_pos=Position(x=ax, y=ay),
                orientation=self.orientation[agent_id],
            ),
            local_patch=local_patch,
            neighbors_in_view=neighbors,
            artifacts_in_view=artifacts,
            inbox=inbox,
            adjacent=self._adjacent_summary(agent_id, ax, ay),
            recent_positions=[
                Position(x=px, y=py)
                for px, py in self.position_history.get(agent_id, [])[:5]
            ],
            comm_limits=CommLimits(
                range=radio_range,
                max_outbound_per_turn=1,
                max_payload_chars=96,
            ),
            mark_limits=MarkLimits(max_ttl=12, allow_mark_info_broadcast=True),
            goal_sensor=self._bearing_sensor(ax, ay),
            last_move_outcome=self.last_move_outcome.get(agent_id, MoveOutcome.OK),
            contended_neighbors=self.contended_neighbors.get(agent_id, 0),
            history=[
                TurnHistory.model_validate(item)
                for item in list(self.turn_history.get(agent_id, []))
            ],
        )
        return obs

    def _render_patch(self, cx: int, cy: int, radius: int) -> LocalPatch:
        rows: List[str] = []
        active_positions = {
            pos for aid, pos in self.occupancy.items() if not self.is_finished(aid)
        }
        for dy in range(-radius, radius + 1):
            line_chars: List[str] = []
            for dx in range(-radius, radius + 1):
                x, y = cx + dx, cy + dy
                if not self._in_bounds(x, y):
                    line_chars.append("#")
                    continue
                ch: TileChar = "."
                if (x, y) in self.walls:
                    ch = "#"
                elif (x, y) == (self.goal.x, self.goal.y):
                    ch = "G"
                elif (x, y) in active_positions:
                    ch = "A"
                elif (x, y) in self.artifacts:
                    ch = "*"
                line_chars.append(ch)
            rows.append("".join(line_chars))
        top_left = Position(x=max(0, cx - radius), y=max(0, cy - radius))
        return LocalPatch(radius=radius, top_left_abs=top_left, rows=rows)

    def _neighbors_in_view(self, agent_id: str, radius: int) -> List[NeighborSummary]:
        cx, cy = self.occupancy[agent_id]
        neighbors: List[NeighborSummary] = []
        for other_id, (ox, oy) in self.occupancy.items():
            if other_id == agent_id:
                continue
            if self.is_finished(other_id):
                continue
            if abs(ox - cx) <= radius and abs(oy - cy) <= radius:
                neighbors.append(
                    NeighborSummary(
                        agent_id=other_id,
                        abs_pos=Position(x=ox, y=oy),
                        rel=RelativeOffset(dx=ox - cx, dy=oy - cy),
                    )
                )
        return neighbors

    def _artifacts_in_view(
        self, cx: int, cy: int, radius: int
    ) -> List[PlacedArtifact]:
        results: List[PlacedArtifact] = []
        for (ax, ay), artifact in self.artifacts.items():
            if abs(ax - cx) <= radius and abs(ay - cy) <= radius:
                results.append(artifact)
        return results

    def _has_active_no_go(self, x: int, y: int) -> bool:
        artifact = self.artifacts.get((x, y))
        return isinstance(artifact, ArtifactNoGo) and artifact.ttl_remaining > 0

    # ------------------------------------------------------------------
    # Sensors
    # ------------------------------------------------------------------

    def _adjacent_summary(self, agent_id: str, ax: int, ay: int) -> List[AdjacentCell]:
        active_positions = {
            pos: other_id
            for other_id, pos in self.occupancy.items()
            if not self.is_finished(other_id)
        }
        mask = self.contended_neighbors.get(agent_id, 0)
        summary: List[AdjacentCell] = []
        for idx, (dir_name, delta) in enumerate({"N": (0, -1), "E": (1, 0), "S": (0, 1), "W": (-1, 0)}.items()):
            dx, dy = delta
            nx, ny = ax + dx, ay + dy
            if not self._in_bounds(nx, ny):
                state = AdjacentState.OUT_OF_BOUNDS
            elif (nx, ny) in self.walls:
                state = AdjacentState.WALL
            elif (nx, ny) == (self.goal.x, self.goal.y):
                state = AdjacentState.GOAL
            elif (nx, ny) in active_positions and active_positions[(nx, ny)] != agent_id:
                state = AdjacentState.AGENT
            else:
                state = AdjacentState.FREE

            if state == AdjacentState.FREE and mask & (1 << idx):
                state = AdjacentState.CONTENDED

            if state in (AdjacentState.FREE, AdjacentState.CONTENDED) and self._has_active_no_go(nx, ny):
                state = AdjacentState.NO_GO

            summary.append(AdjacentCell(dir=dir_name, state=state))
        return summary

    def _record_position(self, agent_id: str) -> None:
        history = self.position_history.setdefault(agent_id, [])
        current = self.occupancy.get(agent_id)
        if current is None:
            return
        if history and history[0] == current:
            return
        history.insert(0, current)
        if len(history) > self.history_limit:
            del history[self.history_limit :]

    def _bearing_sensor(self, x: int, y: int) -> GoalSensorReading:
        if self.rng.random() < self.bearing_drop_p:
            return GoalSensorBearing(bearing=None, strength=None, available=False)

        dx = self.goal.x - x
        dy = self.goal.y - y
        if dx == 0 and dy == 0:
            bearing = Octant.N
        else:
            angle = math.degrees(math.atan2(-dy, dx)) % 360.0
            bins = [
                Octant.E,
                Octant.NE,
                Octant.N,
                Octant.NW,
                Octant.W,
                Octant.SW,
                Octant.S,
                Octant.SE,
            ]
            idx = int((angle + 22.5) // 45) % 8
            bearing = bins[idx]
        if self.bearing_bias_seed is not None:
            steps = self._bias_steps(
                x,
                y,
                self.bearing_bias_seed,
                self.bearing_bias_p,
                self.bearing_bias_wall_bonus,
            )
            if steps != 0:
                bearing = self._rotate_octant(bearing, steps)
        if self.rng.random() < self.bearing_flip_p:
            order = [
                Octant.N,
                Octant.NE,
                Octant.E,
                Octant.SE,
                Octant.S,
                Octant.SW,
                Octant.W,
                Octant.NW,
            ]
            j = (order.index(bearing) + self.rng.choice([-1, 1])) % 8
            bearing = order[j]

        manhattan = abs(dx) + abs(dy)
        if manhattan <= 4:
            strength = StrengthBucket.NEAR
        elif manhattan <= 10:
            strength = StrengthBucket.MID
        else:
            strength = StrengthBucket.FAR
        return GoalSensorBearing(bearing=bearing, strength=strength, available=True)

    def _neighbor_has_wall(self, x: int, y: int) -> bool:
        for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
            nx, ny = x + dx, y + dy
            if (nx, ny) in self.walls:
                return True
        return False

    def _bias_steps(
        self,
        x: int,
        y: int,
        seed: int,
        base_prob: float,
        wall_bonus: float,
    ) -> int:
        if base_prob <= 0 and wall_bonus <= 0:
            return 0
        h = ((x * 73856093) ^ (y * 19349663) ^ (seed * 83492791)) & 0xFFFFFFFF
        primary = ((h >> 8) & 0xFFFF) / 65535.0
        secondary = (h & 0xFF) / 255.0
        bias_p = base_prob + (wall_bonus if self._neighbor_has_wall(x, y) else 0.0)
        bias_p = max(0.0, min(bias_p, 0.49))
        if primary < bias_p:
            return 1 if secondary < 0.5 else -1
        return 0

    @staticmethod
    def _rotate_octant(bearing: Octant, steps: int) -> Octant:
        order = [
            Octant.N,
            Octant.NE,
            Octant.E,
            Octant.SE,
            Octant.S,
            Octant.SW,
            Octant.W,
            Octant.NW,
        ]
        idx = order.index(bearing)
        return order[(idx + steps) % len(order)]

    # ------------------------------------------------------------------
    # Messaging
    # ------------------------------------------------------------------

    def deliver_message(self, recipient_id: str, message: ReceivedMessage) -> None:
        if recipient_id not in self.inboxes:
            self.inboxes[recipient_id] = []
        self.inboxes[recipient_id].append(message)

    def is_finished(self, agent_id: str) -> bool:
        return self.finished_agents.get(agent_id, False)

    def mark_finished(self, agent_id: str) -> None:
        self.finished_agents[agent_id] = True
        history = self.position_history.get(agent_id)
        if history is not None and (self.goal.x, self.goal.y) not in history[:1]:
            history.insert(0, (self.goal.x, self.goal.y))
            if len(history) > self.history_limit:
                del history[self.history_limit :]
        self.last_move_outcome[agent_id] = MoveOutcome.FINISHED
        self.loop_counters[agent_id] = 0
        self.contended_neighbors[agent_id] = 0

    def record_history(self, agent_id: str, payload: dict) -> None:
        if agent_id not in self.turn_history:
            self.turn_history[agent_id] = deque(maxlen=self.history_limit)
        self.turn_history[agent_id].append(payload)

    # ------------------------------------------------------------------
    # Movement and artifacts
    # ------------------------------------------------------------------

    def resolve_moves(self, intents: Dict[str, Optional[Direction]]) -> Dict[str, MoveResult]:
        start_positions = {aid: self.occupancy[aid] for aid in self.occupancy.keys()}
        targets: Dict[str, Optional[Tuple[int, int]]] = {}
        proposed: Dict[str, Tuple[int, int]] = {}
        results: Dict[str, MoveResult] = {}

        for aid in self.occupancy.keys():
            sx, sy = start_positions[aid]
            if self.is_finished(aid):
                proposed[aid] = (sx, sy)
                targets[aid] = None
                results[aid] = MoveResult(final=(sx, sy), outcome=MoveOutcome.FINISHED, target=None, opponents=[], cause_cell=None)

        for agent_id, direction in intents.items():
            sx, sy = start_positions[agent_id]
            if self.is_finished(agent_id):
                continue
            if direction is None:
                proposed[agent_id] = (sx, sy)
                targets[agent_id] = None
                results[agent_id] = MoveResult(final=(sx, sy), outcome=MoveOutcome.YIELD, target=None, opponents=[], cause_cell=None)
                continue
            dx, dy = _direction_delta(direction)
            tx, ty = sx + dx, sy + dy
            targets[agent_id] = (tx, ty)
            if not self._in_bounds(tx, ty):
                proposed[agent_id] = (sx, sy)
                results[agent_id] = MoveResult(
                    final=(sx, sy),
                    outcome=MoveOutcome.BLOCK_OOB,
                    target=(tx, ty),
                    opponents=[],
                    cause_cell=(tx, ty),
                )
            elif not self._passable(tx, ty):
                proposed[agent_id] = (sx, sy)
                results[agent_id] = MoveResult(
                    final=(sx, sy),
                    outcome=MoveOutcome.BLOCK_WALL,
                    target=(tx, ty),
                    opponents=[],
                    cause_cell=(tx, ty),
                )
            else:
                proposed[agent_id] = (tx, ty)
                results[agent_id] = MoveResult(
                    final=(tx, ty),
                    outcome=MoveOutcome.OK,
                    target=(tx, ty),
                    opponents=[],
                    cause_cell=None,
                )

        for aid in self.occupancy.keys():
            if aid not in proposed:
                proposed[aid] = start_positions[aid]
                targets.setdefault(aid, None)
                results.setdefault(
                    aid,
                    MoveResult(final=start_positions[aid], outcome=MoveOutcome.OK, target=None, opponents=[], cause_cell=None),
                )

        occupants: Dict[Tuple[int, int], List[str]] = {}
        for aid, cell in proposed.items():
            occupants.setdefault(cell, []).append(aid)

        swap_lookup: Dict[str, List[str]] = {}
        for aid, target in targets.items():
            if target is None or self.is_finished(aid):
                continue
            for other, other_target in targets.items():
                if other <= aid or self.is_finished(other):
                    continue
                if other_target is None:
                    continue
                if target == start_positions.get(other) and other_target == start_positions.get(aid):
                    swap_lookup.setdefault(aid, []).append(other)
                    swap_lookup.setdefault(other, []).append(aid)

        contested_cells: List[Tuple[int, int]] = []
        for cell, ids in occupants.items():
            if len(ids) == 1:
                aid = ids[0]
                if aid in swap_lookup:
                    self.occupancy[aid] = start_positions[aid]
                    opponents = swap_lookup[aid]
                    results[aid] = MoveResult(
                        final=start_positions[aid],
                        outcome=MoveOutcome.SWAP_CONFLICT,
                        target=targets.get(aid),
                        opponents=opponents,
                        cause_cell=targets.get(aid),
                    )
                    if targets.get(aid) is not None:
                        contested_cells.append(targets[aid])
                    continue

                self.occupancy[aid] = cell
                result = results[aid]
                if result.outcome == MoveOutcome.OK and cell == (self.goal.x, self.goal.y):
                    results[aid] = MoveResult(
                        final=cell,
                        outcome=MoveOutcome.FINISHED,
                        target=result.target,
                        opponents=result.opponents,
                        cause_cell=result.cause_cell,
                    )
                continue

            swap = False
            if len(ids) == 2:
                a, b = ids
                if targets.get(a) == start_positions.get(b) and targets.get(b) == start_positions.get(a):
                    swap = True

            for aid in ids:
                self.occupancy[aid] = start_positions[aid]
                opponents = [other for other in ids if other != aid]
                outcome = MoveOutcome.SWAP_CONFLICT if swap else MoveOutcome.BLOCK_AGENT
                results[aid] = MoveResult(
                    final=start_positions[aid],
                    outcome=outcome,
                    target=targets.get(aid),
                    opponents=opponents,
                    cause_cell=cell,
                )
            contested_cells.append(cell)

        for aid in intents.keys():
            self._record_position(aid)

        for cell in contested_cells:
            if self._in_bounds(*cell) and cell != (self.goal.x, self.goal.y) and cell not in self.walls:
                self._place_congestion_marker(cell)

        return results

    def place_artifact(self, agent_id: str, artifact: PlacedArtifact) -> None:
        ax, ay = self.occupancy[agent_id]
        allowed_kinds = {"TRAIL", "BEARING_SAMPLE", "NO_GO", "CLAIM", "GOAL_HINT"}
        if artifact.kind not in allowed_kinds:
            raise ValueError(f"Unsupported artifact kind: {artifact.kind}")

        located = artifact
        if isinstance(artifact, ArtifactTrail):
            located = ArtifactTrail(
                kind="TRAIL", dir_entered=artifact.dir_entered, ttl_remaining=artifact.ttl_remaining
            )
        elif isinstance(artifact, ArtifactBearingSample):
            located = ArtifactBearingSample(
                kind="BEARING_SAMPLE",
                bearing=artifact.bearing,
                strength=artifact.strength,
                measured_at=Position(x=ax, y=ay),
                turn_index=artifact.turn_index,
                ttl_remaining=artifact.ttl_remaining,
            )
        elif isinstance(artifact, ArtifactNoGo):
            located = artifact
        elif isinstance(artifact, ArtifactClaim):
            located = artifact
        elif isinstance(artifact, ArtifactGoalHint):
            located = artifact
        self.artifacts[(ax, ay)] = located

    def _place_congestion_marker(self, cell: Tuple[int, int]) -> None:
        existing = self.artifacts.get(cell)
        ttl = TRAFFIC_CONE_TTL
        if isinstance(existing, ArtifactNoGo):
            ttl = max(existing.ttl_remaining, ttl)
        self.artifacts[cell] = ArtifactNoGo(kind="NO_GO", reason=BlockReason.CONGESTION, ttl_remaining=ttl)

    def decay_artifacts(self) -> None:
        expired: List[Tuple[int, int]] = []
        for key, artifact in self.artifacts.items():
            ttl = artifact.ttl_remaining - 1
            if ttl <= 0:
                expired.append(key)
            else:
                if isinstance(artifact, ArtifactTrail):
                    self.artifacts[key] = ArtifactTrail(
                        kind="TRAIL", dir_entered=artifact.dir_entered, ttl_remaining=ttl
                    )
                elif isinstance(artifact, ArtifactBearingSample):
                    self.artifacts[key] = ArtifactBearingSample(
                        kind="BEARING_SAMPLE",
                        bearing=artifact.bearing,
                        strength=artifact.strength,
                        measured_at=artifact.measured_at,
                        turn_index=artifact.turn_index,
                        ttl_remaining=ttl,
                    )
                elif isinstance(artifact, ArtifactNoGo):
                    self.artifacts[key] = ArtifactNoGo(
                        kind="NO_GO", reason=artifact.reason, ttl_remaining=ttl
                    )
                elif isinstance(artifact, ArtifactClaim):
                    self.artifacts[key] = ArtifactClaim(
                        kind="CLAIM",
                        next_turn_index=artifact.next_turn_index,
                        ttl_remaining=ttl,
                    )
                elif isinstance(artifact, ArtifactGoalHint):
                    self.artifacts[key] = ArtifactGoalHint(
                        kind="GOAL_HINT", confidence=artifact.confidence, ttl_remaining=ttl
                    )
        for key in expired:
            self.artifacts.pop(key, None)

    # ------------------------------------------------------------------
    # Helpers for checking progress
    # ------------------------------------------------------------------

    def agent_on_goal(self, agent_id: str) -> bool:
        return self.occupancy[agent_id] == (self.goal.x, self.goal.y)

    def all_agents_on_goal(self, agent_ids: Iterable[str]) -> bool:
        return all(self.agent_on_goal(aid) for aid in agent_ids)

    # ------------------------------------------------------------------
    # Message constructors to keep schema usage centralised
    # ------------------------------------------------------------------

    @staticmethod
    def message_here(sender_id: str, seq: int, pos: Position, orientation: Direction) -> MsgHere:
        return MsgHere(kind="HERE", sender_id=sender_id, seq=seq, pos=pos, orientation=orientation)

    @staticmethod
    def message_intent(sender_id: str, seq: int, intent: str) -> MsgIntent:
        return MsgIntent(kind="INTENT", sender_id=sender_id, seq=seq, next_action=intent)

    @staticmethod
    def message_sense(
        sender_id: str,
        seq: int,
        at: Position,
        bearing: Optional[Octant],
        strength: Optional[StrengthBucket],
    ) -> MsgSense:
        return MsgSense(
            kind="SENSE",
            sender_id=sender_id,
            seq=seq,
            at=at,
            mode="BEARING",
            bearing=bearing,
            strength=strength,
        )

    @staticmethod
    def message_mark_info(sender_id: str, seq: int, artifact: PlacedArtifact) -> MsgMarkInfo:
        return MsgMarkInfo(kind="MARK_INFO", sender_id=sender_id, seq=seq, placed=artifact)
