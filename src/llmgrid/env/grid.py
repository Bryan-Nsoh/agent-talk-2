"""Environment primitives for the partially observable grid world."""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Optional, Tuple

from llmgrid.agent_map import AgentMap
from llmgrid.schema import (
    AgentSelf,
    AdjacentCell,
    AdjacentState,
    BlockReason,
    CommLimits,
    Direction,
    GoalSensorBearing,
    GoalSensorReading,
    GridSize,
    LocalPatch,
    MoveOutcome,
    MessageBrief,
    MsgHere,
    MsgIntent,
    NeighborSummary,
    Observation,
    Octant,
    Position,
    ReceivedMessage,
    RelativeOffset,
    StrengthBucket,
    TurnHistory,
    WorldMapMeta,
)

TileChar = str  # ".", "#", "G", "A", "*"

TRAFFIC_CONE_TTL = 0  # artifacts removed


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

    _ICON_SEQUENCE = list("123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")

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
        self.message_history: Dict[str, Deque[MessageBrief]] = {}
        self.artifacts: Dict[Tuple[int, int], dict] = {}
        self.finished_agents: Dict[str, bool] = {}
        self.position_history: Dict[str, List[Tuple[int, int]]] = {}
        self.turn_history: Dict[str, Deque[dict]] = {}
        self.last_move_outcome: Dict[str, MoveOutcome] = {}
        self.last_goal_distance: Dict[str, int] = {}
        self.last_intent_target: Dict[str, Optional[Tuple[int, int]]] = {}
        self.contended_neighbors: Dict[str, int] = {}
        self.message_seq: Dict[str, int] = {}
        self.agent_maps: Dict[str, AgentMap] = {}
        self.agent_icons: Dict[str, str] = {}

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
        self.message_history[agent_id] = deque(maxlen=10)
        self.last_move_outcome[agent_id] = MoveOutcome.OK
        self.last_goal_distance[agent_id] = abs(self.goal.x - pos.x) + abs(self.goal.y - pos.y)
        self.last_intent_target[agent_id] = None
        self.contended_neighbors[agent_id] = 0
        self.message_seq[agent_id] = 0
        self.agent_maps[agent_id] = AgentMap(self.size.width, self.size.height)
        self.agent_icons[agent_id] = self._assign_icon(agent_id)

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
        max_outbound_per_turn: int = 1,
    ) -> Observation:
        ax, ay = self.occupancy[agent_id]
        local_patch = self._render_patch(ax, ay, visibility_radius)
        neighbors = self._neighbors_in_view(agent_id, visibility_radius)
        artifacts: List[dict] = []
        inbox = list(self.inboxes.get(agent_id, []))
        # Append received messages to brief history before clearing inbox
        for msg in inbox:
            self._record_message_brief(agent_id, msg)
        # Clear inbox for next turn (messages are now reflected in history)
        self.inboxes[agent_id] = []

        # compute radio proximity using Manhattan distance within radio_range
        radio_count = 0
        for other_id, (ox, oy) in self.occupancy.items():
            if other_id == agent_id or self.is_finished(other_id):
                continue
            if abs(ox - ax) + abs(oy - ay) <= radio_range:
                radio_count += 1

        agent_map = self._update_agent_map(
            agent_id=agent_id,
            patch=local_patch,
            turn_index=turn_index,
        )
        world_map_ascii = agent_map.render_ascii(
            icon_lookup=self.agent_icons,
            orientations={aid: self.orientation.get(aid, Direction.N).name for aid in self.agent_maps.keys()},
        )
        frontiers = agent_map.find_frontiers()
        adjacent_frontiers = [
            Position(x=fx, y=fy)
            for fx, fy in frontiers
            if abs(fx - ax) + abs(fy - ay) == 1
        ]
        nearest_frontier = None
        if frontiers:
            fx, fy = min(frontiers, key=lambda coord: abs(coord[0] - ax) + abs(coord[1] - ay))
            nearest_frontier = Position(x=fx, y=fy)

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
            any_peer_in_range=(radio_count > 0),
            radio_peers_count=radio_count,
            artifacts_in_view=artifacts,
            inbox=inbox,
            recent_messages=list(self.message_history.get(agent_id, [])),
            adjacent=self._adjacent_summary(agent_id, ax, ay),
            recent_positions=[
                Position(x=px, y=py)
                for px, py in self.position_history.get(agent_id, [])[:10]
            ],
            comm_limits=CommLimits(
                range=radio_range,
                max_outbound_per_turn=max(0, max_outbound_per_turn),
                max_payload_chars=96,
            ),
            goal_sensor=self._bearing_sensor(ax, ay),
            world_map_meta=WorldMapMeta(
                x_right=True,
                y_up=True,
                origin=Position(x=0, y=0),
                width=self.size.width,
                height=self.size.height,
            ),
            adjacent_frontiers=adjacent_frontiers,
            nearest_frontier=nearest_frontier,
            last_move_outcome=self.last_move_outcome.get(agent_id, MoveOutcome.OK),
            contended_neighbors=self.contended_neighbors.get(agent_id, 0),
            history=[
                TurnHistory.model_validate(item)
                for item in list(self.turn_history.get(agent_id, []))
            ],
            world_map_ascii=world_map_ascii,
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
                # artifacts removed
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

    def _update_agent_map(
        self,
        *,
        agent_id: str,
        patch: LocalPatch,
        turn_index: int,
    ) -> AgentMap:
        """Update the persistent egocentric map for an agent and return it."""

        agent_map = self.agent_maps[agent_id]
        occupancy_lookup = {
            other_id: pos
            for other_id, pos in self.occupancy.items()
            if not self.is_finished(other_id)
        }
        agent_map.stamp_patch(
            top_left=patch.top_left_abs,
            rows=patch.rows,
            occupancy_lookup=occupancy_lookup,
            turn_index=turn_index,
        )
        recent_positions = list(self.position_history.get(agent_id, [])[:3])
        agent_map.mark_self(
            agent_id,
            self.occupancy[agent_id],
            turn_index=turn_index,
            recent_positions=recent_positions,
        )
        return agent_map

    def _assign_icon(self, agent_id: str) -> str:
        """Return a concise single-character icon to represent an agent on the map."""

        if agent_id in self.agent_icons:
            return self.agent_icons[agent_id]
        idx = len(self.agent_icons)
        if idx < len(self._ICON_SEQUENCE):
            return self._ICON_SEQUENCE[idx]
        # Fallback once the pool is exhausted; still deterministic.
        return "@"

    def _empty_mark_limits(self):
        class _M:
            max_ttl = 0
            allow_mark_info_broadcast = False
        return _M()

    def _has_active_no_go(self, x: int, y: int) -> bool:
        return False

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
        envelope = message.envelope
        sender = getattr(envelope, "sender_id", None)
        if sender is not None and envelope.seq is None:
            envelope.seq = self.next_message_seq(sender)
        if recipient_id not in self.inboxes:
            self.inboxes[recipient_id] = []
        self.inboxes[recipient_id].append(message)

    def is_finished(self, agent_id: str) -> bool:
        return self.finished_agents.get(agent_id, False)

    def next_message_seq(self, sender_id: str) -> int:
        current = self.message_seq.get(sender_id, 0)
        self.message_seq[sender_id] = current + 1
        return current

    def increment_inbox_ages(self) -> None:
        for messages in self.inboxes.values():
            for msg in messages:
                msg.age += 1
        # Also age the brief history summaries
        for dq in self.message_history.values():
            for i in range(len(dq)):
                brief = dq[i]
                new_age = (brief.age or 0) + 1
                dq[i] = MessageBrief(kind=brief.kind, details=brief.details, sender=brief.sender, hop=brief.hop, age=new_age)

    def mark_finished(self, agent_id: str) -> None:
        self.finished_agents[agent_id] = True
        history = self.position_history.get(agent_id)
        if history is not None and (self.goal.x, self.goal.y) not in history[:1]:
            history.insert(0, (self.goal.x, self.goal.y))
            if len(history) > self.history_limit:
                del history[self.history_limit :]
        self.last_move_outcome[agent_id] = MoveOutcome.FINISHED
        self.contended_neighbors[agent_id] = 0

    # ------------------------------------------------------------------
    # Message brief helpers
    # ------------------------------------------------------------------

    def _record_message_brief(self, agent_id: str, received: ReceivedMessage) -> None:
        env = received.envelope
        kind = getattr(env, "kind", "")
        details = None
        if kind == "HERE":
            try:
                details = f"pos=({env.pos.x},{env.pos.y})"
            except Exception:
                details = None
        elif kind == "INTENT":
            details = getattr(env, "next_action", None)
        elif kind == "REQUEST":
            parts = [getattr(env, "req", None)]
            tgt = getattr(env, "target", None)
            if tgt is not None:
                parts.append(f"target=({tgt.x},{tgt.y})")
            details = " ".join([p for p in parts if p]) or None
        elif kind == "MAP_REQUEST":
            try:
                details = f"origin=({env.origin.x},{env.origin.y}) r={getattr(env, 'radius', '?')}"
            except Exception:
                details = None
        elif kind == "MAP_PATCH":
            try:
                details = f"origin=({env.origin.x},{env.origin.y}) rows={len(env.rows)}"
            except Exception:
                details = None
        elif kind == "MAP_NO_PATCH":
            try:
                details = f"origin=({env.origin.x},{env.origin.y})"
            except Exception:
                details = None
        elif kind == "CHAT":
            txt = getattr(env, "text", None)
            if isinstance(txt, str):
                details = txt[:96]
        brief = MessageBrief(kind=kind, details=details, sender=getattr(env, "sender_id", None), hop=received.hop_distance, age=received.age)
        self.message_history.setdefault(agent_id, deque(maxlen=10)).append(brief)

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
            self.orientation[agent_id] = direction
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
            if self.is_finished(aid):
                continue
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

        # artifacts removed; no congestion markers are placed

        return results

    def place_artifact(self, agent_id: str, artifact) -> None:
        return None

    def _place_congestion_marker(self, cell: Tuple[int, int]) -> None:
        return None

    def decay_artifacts(self) -> None:
        return None

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

    # Removed legacy radio helper constructors (SENSE, MARK_INFO) in this branch.
