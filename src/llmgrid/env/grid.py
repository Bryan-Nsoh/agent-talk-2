"""Environment primitives for the map-sharing single-grid observation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Optional, Tuple

from llmgrid.agent_map import AgentMap
from llmgrid.schema import (
    AdjacentCell,
    AdjacentState,
    AgentSelf,
    Direction,
    Grid,
    LastResult,
    MoveOutcome,
    NeighborInView,
    Observation,
    Position,
)

Coordinate = Tuple[int, int]


def _direction_delta(direction: Direction) -> Coordinate:
    return {
        Direction.N: (0, -1),
        Direction.E: (1, 0),
        Direction.S: (0, 1),
        Direction.W: (-1, 0),
    }[direction]


@dataclass
class MoveResult:
    final: Coordinate
    outcome: MoveOutcome
    target: Optional[Coordinate]
    opponents: List[str]


class GridWorld:
    """Grid-based environment with synchronous turns, no comms."""

    _ICON_SEQUENCE = list("123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    def __init__(
        self,
        width: int,
        height: int,
        obstacles: Iterable[Position],
        goal: Position,
        *,
        seed: int = 0,
        history_limit: int = 10,
    ) -> None:
        self.size = (width, height)
        self.goal = (goal.x, goal.y)
        self.walls = {(p.x, p.y) for p in obstacles}
        self.history_limit = max(1, history_limit)
        self.occupancy: Dict[str, Coordinate] = {}
        self.finished: Dict[str, bool] = {}
        self.finished_positions: Dict[str, Coordinate] = {}
        self.position_history: Dict[str, List[Coordinate]] = {}
        self.last_result: Dict[str, LastResult] = {}
        self.agent_maps: Dict[str, AgentMap] = {}
        self.agent_icons: Dict[str, str] = {}

    # ------------------------------------------------------------------ #
    # Agent placement
    # ------------------------------------------------------------------ #
    def add_agent(self, agent_id: str, pos: Position) -> None:
        key = (pos.x, pos.y)
        if key in self.walls:
            raise ValueError("Cannot spawn agent on a wall.")
        if not self._in_bounds(*key):
            raise ValueError("Spawn position out of bounds.")
        self.occupancy[agent_id] = key
        self.finished[agent_id] = False
        self.position_history[agent_id] = [key]
        self.last_result[agent_id] = LastResult(kind=MoveOutcome.OK, cell=None, opponents=[])
        self.agent_maps[agent_id] = AgentMap(*self.size)
        self.agent_icons[agent_id] = self._assign_icon(agent_id)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _assign_icon(self, agent_id: str) -> str:
        if agent_id in self.agent_icons:
            return self.agent_icons[agent_id]
        idx = len(self.agent_icons)
        if idx < len(self._ICON_SEQUENCE):
            return self._ICON_SEQUENCE[idx]
        return "@"

    def _in_bounds(self, x: int, y: int) -> bool:
        w, h = self.size
        return 0 <= x < w and 0 <= y < h

    def _passable(self, x: int, y: int) -> bool:
        return (x, y) not in self.walls and self._in_bounds(x, y)

    def is_finished(self, agent_id: str) -> bool:
        return self.finished.get(agent_id, False)

    def agent_on_goal(self, agent_id: str) -> bool:
        return self.occupancy.get(agent_id) == self.goal

    def all_agents_on_goal(self, agent_ids: Iterable[str]) -> bool:
        return all(self.agent_on_goal(aid) for aid in agent_ids)

    # ------------------------------------------------------------------ #
    # Observation construction
    # ------------------------------------------------------------------ #
    def build_observation(
        self,
        agent_id: str,
        *,
        turn_index: int,
        max_turns: int,
        visibility_radius: int,
        map_sharing: str,
    ) -> Observation:
        ax, ay = self.occupancy[agent_id]
        neighbors = self._neighbors_in_view(agent_id, visibility_radius)
        neighbor_positions = {(n.pos.x, n.pos.y): self.agent_icons.get(n.agent_id, "?") for n in neighbors}

        visible_cells = self._visible_cells(ax, ay, visibility_radius)
        amap = self.agent_maps[agent_id]
        amap.update_visible(visible_cells)

        recent_positions = list(self.position_history.get(agent_id, [])[:3])
        amap.set_recent(recent_positions)
        amap.set_last_collision(self._last_collision_cell(agent_id))
        amap.mark_visit((ax, ay))

        grid_rows = amap.render_grid(
            self_pos=(ax, ay),
            neighbors=neighbor_positions,
            goal_pos=self.goal if self.goal else None,
        )

        frontiers = amap.find_frontiers()
        adjacent_frontiers = [
            Position(x=fx, y=fy) for fx, fy in frontiers if abs(fx - ax) + abs(fy - ay) == 1
        ]

        goal_known = any("G" in row for row in amap._base)
        goal_pos = Position(x=self.goal[0], y=self.goal[1]) if goal_known else None

        adj = self._adjacent_summary(agent_id, ax, ay)

        obs = Observation(
            protocol_version="3.0.0",
            turn_index=turn_index,
            max_turns=max_turns,
            grid=Grid(width=self.size[0], height=self.size[1], rows=grid_rows),
            legend=self._legend(),
            self=AgentSelf(agent_id=agent_id, pos=Position(x=ax, y=ay)),
            neighbors_in_view=neighbors,
            adjacent=adj,
            adjacent_frontiers=adjacent_frontiers,
            goal_known=goal_known,
            goal_pos=goal_pos,
            last_result=self.last_result.get(agent_id, LastResult(kind=MoveOutcome.OK, cell=None, opponents=[])),
            map_sharing=map_sharing,
        )
        return obs

    def _legend(self) -> Dict[str, str]:
        return {
            "#": "WALL (impassable)",
            "G": "GOAL (reach to finish)",
            "X": "UNKNOWN (unseen)",
            ".": "FREE (seen, not visited)",
            "~": "FREE (visited 2+ turns ago)",
            "*": "FREE (in your last 3 positions)",
            "!": "FREE (your last collision target)",
            "@": "SELF (you are here)",
            "1,2,3...": "OTHER AGENTS (visible in neighbors_in_view)",
        }

    def _visible_cells(self, cx: int, cy: int, radius: int) -> List[Tuple[int, int, str]]:
        cells: List[Tuple[int, int, str]] = []
        for x in range(cx - radius, cx + radius + 1):
            for y in range(cy - radius, cy + radius + 1):
                if abs(x - cx) + abs(y - cy) > radius:
                    continue
                if not self._in_bounds(x, y):
                    continue
                if (x, y) in self.walls:
                    ch = "#"
                elif (x, y) == self.goal:
                    ch = "G"
                else:
                    ch = "."
                cells.append((x, y, ch))
        return cells

    def _neighbors_in_view(self, agent_id: str, radius: int) -> List[NeighborInView]:
        ax, ay = self.occupancy[agent_id]
        neighbors: List[NeighborInView] = []
        for other_id, (ox, oy) in self.occupancy.items():
            if other_id == agent_id or self.is_finished(other_id):
                continue
            if abs(ox - ax) + abs(oy - ay) <= radius:
                neighbors.append(NeighborInView(agent_id=other_id, pos=Position(x=ox, y=oy)))
        return neighbors

    def _adjacent_summary(self, agent_id: str, ax: int, ay: int) -> List[AdjacentCell]:
        summary: List[AdjacentCell] = []
        occupied = {pos for aid, pos in self.occupancy.items() if aid != agent_id and not self.is_finished(aid)}
        for dir_name, delta in {
            Direction.N: (0, -1),
            Direction.E: (1, 0),
            Direction.S: (0, 1),
            Direction.W: (-1, 0),
        }.items():
            dx, dy = delta
            nx, ny = ax + dx, ay + dy
            if not self._in_bounds(nx, ny):
                state = AdjacentState.OUT_OF_BOUNDS
            elif (nx, ny) in self.walls:
                state = AdjacentState.WALL
            elif (nx, ny) == self.goal:
                state = AdjacentState.GOAL
            elif (nx, ny) in occupied:
                state = AdjacentState.AGENT
            else:
                state = AdjacentState.FREE
            summary.append(AdjacentCell(dir=dir_name, state=state))
        return summary

    def _last_collision_cell(self, agent_id: str) -> Optional[Coordinate]:
        lr = self.last_result.get(agent_id)
        if lr and lr.kind in {MoveOutcome.BLOCK_AGENT, MoveOutcome.SWAP_CONFLICT, MoveOutcome.BLOCK_WALL}:
            if lr.cell is not None:
                return (lr.cell.x, lr.cell.y)
        return None

    # ------------------------------------------------------------------ #
    # Movement
    # ------------------------------------------------------------------ #
    def resolve_moves(self, intents: Dict[str, Optional[Direction]]) -> Dict[str, MoveResult]:
        start_positions = dict(self.occupancy)
        targets: Dict[str, Optional[Coordinate]] = {}
        results: Dict[str, MoveResult] = {}

        # Proposed targets
        for aid, pos in start_positions.items():
            if self.is_finished(aid):
                results[aid] = MoveResult(final=pos, outcome=MoveOutcome.FINISHED, target=None, opponents=[])
                targets[aid] = None
                continue
            direction = intents.get(aid)
            if direction is None:
                targets[aid] = None
                results[aid] = MoveResult(final=pos, outcome=MoveOutcome.OK, target=None, opponents=[])
            else:
                dx, dy = _direction_delta(direction)
                targets[aid] = (pos[0] + dx, pos[1] + dy)

        # Resolve walls/out-of-bounds
        for aid, target in targets.items():
            if self.is_finished(aid):
                continue
            if target is None:
                continue
            tx, ty = target
            if not self._in_bounds(tx, ty) or (tx, ty) in self.walls:
                results[aid] = MoveResult(
                    final=start_positions[aid],
                    outcome=MoveOutcome.BLOCK_WALL,
                    target=target,
                    opponents=[],
                )

        # Track proposed entries
        occupants: Dict[Coordinate, List[str]] = {}
        for aid, target in targets.items():
            if self.is_finished(aid):
                continue
            if target is None:
                continue
            if aid in results and results[aid].outcome == MoveOutcome.BLOCK_WALL:
                continue
            occupants.setdefault(target, []).append(aid)

        # Detect swaps
        swap_pairs: Dict[str, str] = {}
        for aid, target in targets.items():
            if target is None or self.is_finished(aid):
                continue
            for bid, btarget in targets.items():
                if bid == aid or btarget is None or self.is_finished(bid):
                    continue
                if target == start_positions.get(bid) and btarget == start_positions.get(aid):
                    swap_pairs[aid] = bid

        # Handle conflicts and moves
        for cell, ids in occupants.items():
            if len(ids) == 1:
                aid = ids[0]
                if aid in swap_pairs:
                    results[aid] = MoveResult(
                        final=start_positions[aid],
                        outcome=MoveOutcome.SWAP_CONFLICT,
                        target=targets.get(aid),
                        opponents=[swap_pairs[aid]],
                    )
                    continue
                self.occupancy[aid] = cell
                outcome = MoveOutcome.FINISHED if cell == self.goal else MoveOutcome.OK
                results[aid] = MoveResult(final=cell, outcome=outcome, target=cell, opponents=[])
                continue

            # collision
            for aid in ids:
                results[aid] = MoveResult(
                    final=start_positions[aid],
                    outcome=MoveOutcome.BLOCK_AGENT,
                    target=cell,
                    opponents=[o for o in ids if o != aid],
                )

        # Fill results for agents that had OK stay
        for aid in start_positions:
            if aid in results:
                continue
            self.occupancy[aid] = start_positions[aid]
            results[aid] = MoveResult(final=start_positions[aid], outcome=MoveOutcome.OK, target=None, opponents=[])

        # Mark finished agents and remove them from blocking
        finished_now: List[str] = []
        for aid, res in results.items():
            if res.outcome == MoveOutcome.FINISHED:
                finished_now.append(aid)
        for aid in finished_now:
            self.finished[aid] = True
            self.finished_positions[aid] = self.occupancy.get(aid, self.goal)
            self.occupancy.pop(aid, None)

        # Record position histories
        for aid, res in results.items():
            hist = self.position_history.setdefault(aid, deque(maxlen=self.history_limit))
            if isinstance(hist, deque):
                if not hist or hist[0] != res.final:
                    hist.appendleft(res.final)
                self.position_history[aid] = list(hist)
            else:
                if not hist or hist[0] != res.final:
                    hist.insert(0, res.final)
                if len(hist) > self.history_limit:
                    del hist[self.history_limit :]

        # Update last_result
        for aid, res in results.items():
            cell_pos = Position(x=res.target[0], y=res.target[1]) if res.target is not None else None
            self.last_result[aid] = LastResult(kind=res.outcome, cell=cell_pos, opponents=res.opponents)

        return results

    # ------------------------------------------------------------------ #
    # Map sharing
    # ------------------------------------------------------------------ #
    def merge_base_maps(self, aid_src: str, aid_dst: str) -> None:
        self.agent_maps[aid_dst].merge_base_from(self.agent_maps[aid_src])
