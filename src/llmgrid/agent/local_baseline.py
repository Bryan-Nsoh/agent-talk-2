"""Local heuristic baseline for dry runs without LLM calls."""

from __future__ import annotations

import random
from typing import Dict, Optional

from llmgrid.schema import AdjacentCell, AdjacentState, Decision, Direction, MoveAction, Observation, StayAction


class GreedyBaseline:
    """Rule-based fallback: explore frontiers, otherwise drift toward goal if known."""

    def __init__(self, *, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def decide(self, observation: Observation) -> Decision:
        ax, ay = observation.self.pos.x, observation.self.pos.y
        goal = observation.goal_pos if observation.goal_known else None
        adj: Dict[Direction, AdjacentState] = {
            cell.dir: cell.state for cell in observation.adjacent
        }

        # Prefer immediate frontiers
        if observation.adjacent_frontiers:
            target = observation.adjacent_frontiers[0]
            dir_choice = self._direction_toward(ax, ay, target.x, target.y)
            if dir_choice and adj.get(dir_choice) in {AdjacentState.FREE, AdjacentState.GOAL}:
                return Decision(
                    action=MoveAction(direction=dir_choice),
                    comment=f"Frontier {target.x},{target.y}",
                )

        # Move toward goal if known
        if goal:
            best_dir = self._best_goal_dir(ax, ay, goal.x, goal.y, adj)
            if best_dir:
                return Decision(
                    action=MoveAction(direction=best_dir),
                    comment=f"Toward goal {goal.x},{goal.y}",
                )

        # Otherwise pick any free neighbor
        free_dirs = [d for d, st in adj.items() if st in {AdjacentState.FREE, AdjacentState.GOAL}]
        if free_dirs:
            d = self.rng.choice(free_dirs)
            return Decision(action=MoveAction(direction=d), comment="Exploring free cell")

        return Decision(action=StayAction(), comment="No safe move")

    def _direction_toward(self, ax: int, ay: int, tx: int, ty: int) -> Optional[Direction]:
        if tx > ax:
            return Direction.E
        if tx < ax:
            return Direction.W
        if ty > ay:
            return Direction.S
        if ty < ay:
            return Direction.N
        return None

    def _best_goal_dir(
        self,
        ax: int,
        ay: int,
        gx: int,
        gy: int,
        adj: Dict[Direction, AdjacentState],
    ) -> Optional[Direction]:
        best = None
        best_dist = None
        for dir_name, (dx, dy) in {
            Direction.N: (0, -1),
            Direction.E: (1, 0),
            Direction.S: (0, 1),
            Direction.W: (-1, 0),
        }.items():
            if adj.get(dir_name) not in {AdjacentState.FREE, AdjacentState.GOAL}:
                continue
            nx, ny = ax + dx, ay + dy
            dist = abs(nx - gx) + abs(ny - gy)
            if best_dist is None or dist < best_dist:
                best = dir_name
                best_dist = dist
        return best

    async def decide_async(self, observation: Observation) -> Decision:
        return self.decide(observation)

    def get_state(self) -> tuple:
        return self.rng.getstate()

    def set_state(self, state: tuple) -> None:
        self.rng.setstate(state)
