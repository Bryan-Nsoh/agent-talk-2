"""Local heuristic baseline for dry runs without LLM calls."""

from __future__ import annotations

import random
from typing import Optional

from llmgrid.schema import Decision, Direction, MoveAction, Observation, StayAction


class GreedyBaseline:
    """Tiny rule-based agent for debugging the environment without API calls."""

    def __init__(self, *, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def decide(self, observation: Observation) -> Decision:
        # Try moving toward the goal if bearing is available; otherwise stay put.
        bearing = None
        if observation.goal_sensor.mode == "BEARING" and observation.goal_sensor.available:
            bearing = observation.goal_sensor.bearing

        direction = self._bearing_to_direction(bearing)
        if direction is None:
            return Decision(action=StayAction())
        return Decision(action=MoveAction(direction=direction))

    def _bearing_to_direction(self, bearing: Optional[str]) -> Optional[Direction]:
        if bearing is None:
            return None
        mapping = {
            "N": Direction.N,
            "NE": self.rng.choice([Direction.N, Direction.E]),
            "E": Direction.E,
            "SE": self.rng.choice([Direction.S, Direction.E]),
            "S": Direction.S,
            "SW": self.rng.choice([Direction.S, Direction.W]),
            "W": Direction.W,
            "NW": self.rng.choice([Direction.N, Direction.W]),
        }
        return mapping.get(bearing)

    async def decide_async(self, observation: Observation) -> Decision:
        return self.decide(observation)

    def get_state(self) -> tuple:
        """Expose RNG state so simulations can checkpoint baselines."""

        return self.rng.getstate()

    def set_state(self, state: tuple) -> None:
        """Restore RNG state when resuming a simulation."""

        self.rng.setstate(state)
