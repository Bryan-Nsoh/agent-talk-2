"""Episode driver that wires the environment and agent policies together."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Any, TextIO

from llmgrid.agent.llm_agent import DecisionTrace, LlmPolicy
from llmgrid.agent.local_baseline import GreedyBaseline
from llmgrid.env.grid import GridWorld
from llmgrid.schema import (
    Decision,
    Direction,
    Observation,
    PlacedArtifact,
    Position,
    ReceivedMessage,
)


@dataclass
class EpisodeMetrics:
    turns: int
    success: bool
    messages_sent: int
    marks_placed: int
    collisions: int
    reasoning_log: List[Dict[str, Any]]


def _manhattan(a: Position, b: Position) -> int:
    return abs(a.x - b.x) + abs(a.y - b.y)


def _resolve_policy(
    use_llm: bool, model_id: str, seed: int
) -> "PolicyProtocol":
    if use_llm:
        return LlmPolicy(model_id)
    return GreedyBaseline(seed=seed)


class PolicyProtocol:
    """Protocol shim for type checking."""

    def decide(self, observation: Observation) -> Decision:  # pragma: no cover - protocol stub
        raise NotImplementedError


def run_episode(
    *,
    use_llm: bool,
    model_id: str,
    width: int,
    height: int,
    obstacles: Iterable[Position],
    start_positions: Dict[str, Position],
    goal: Position,
    turns: int,
    visibility: int,
    radio_range: int,
    seed: int = 0,
    transcript: Optional[List[dict]] = None,
    movement: Optional[List[dict]] = None,
    transcript_writer: Optional[TextIO] = None,
    movement_writer: Optional[TextIO] = None,
) -> EpisodeMetrics:
    """Simulate a single episode and return aggregate metrics."""

    world = GridWorld(width, height, obstacles, goal, seed=seed)
    for idx, (agent_id, pos) in enumerate(start_positions.items()):
        orientation = [Direction.N, Direction.S, Direction.E, Direction.W][idx % 4]
        world.add_agent(agent_id, pos, orientation)

    policy = _resolve_policy(use_llm, model_id, seed)
    agent_ids = list(start_positions.keys())

    messages_sent = 0
    marks_placed = 0
    collisions = 0
    reasoning_log: List[Dict[str, Any]] = []

    if movement is not None:
        _append_snapshot(movement, world, 0, agent_ids, actions=None)
        if movement_writer is not None:
            movement_writer.write(json.dumps(movement[-1]))
            movement_writer.write("\n")
            movement_writer.flush()

    for turn in range(turns):
        active_agents = [aid for aid in agent_ids if not world.is_finished(aid)]

        observations: Dict[str, Observation] = {}
        for aid in active_agents:
            observations[aid] = world.build_observation(
                aid,
                turn_index=turn,
                max_turns=turns,
                visibility_radius=visibility,
                radio_range=radio_range,
            )

        decisions: Dict[str, Decision] = {}
        for aid in active_agents:
            if transcript is not None and hasattr(policy, "decide_with_trace"):
                trace: DecisionTrace = policy.decide_with_trace(observations[aid])  # type: ignore[attr-defined]
                decisions[aid] = trace.decision
                record = {
                    "turn": turn,
                    "agent_id": aid,
                    "prompt": trace.prompt,
                    "observation": observations[aid].model_dump(mode="json"),
                    "decision": trace.decision.model_dump(mode="json"),
                    "trace_messages": trace.trace_messages,
                }
                if transcript is not None:
                    transcript.append(record)
                if transcript_writer is not None:
                    transcript_writer.write(json.dumps(record))
                    transcript_writer.write("\n")
                    transcript_writer.flush()
            else:
                decisions[aid] = policy.decide(observations[aid])
            comment = decisions[aid].comment
            reasoning_log.append(
                {
                    "turn": turn,
                    "agent_id": aid,
                    "comment": comment if comment is not None else "",
                }
            )

        # Broadcast messages
        for aid, decision in decisions.items():
            if decision.action.kind != "COMMUNICATE":
                continue
            message = decision.action.message
            sender_pos = world.occupancy[aid]
            recipients = _recipients_in_range(world, aid, radio_range)
            for rid in recipients:
                rx, ry = world.occupancy[rid]
                rm = ReceivedMessage(
                    envelope=message,
                    hop_distance=abs(rx - sender_pos[0]) + abs(ry - sender_pos[1]),
                    age=0,
                )
                world.deliver_message(rid, rm)
            messages_sent += 1

        # Place artifacts
        for aid, decision in decisions.items():
            if decision.action.kind != "MARK":
                continue
            placed: PlacedArtifact = decision.action.placement
            world.place_artifact(aid, placed)
            marks_placed += 1

        # Resolve moves and collisions
        intents: Dict[str, Optional[Direction]] = {}
        before = dict(world.occupancy)
        for aid, decision in decisions.items():
            if decision.action.kind == "MOVE":
                intents[aid] = decision.action.direction
            else:
                intents[aid] = None

        world.resolve_moves(intents)
        after = dict(world.occupancy)
        for aid, direction in intents.items():
            if direction is None:
                continue
            if before[aid] == after[aid]:
                collisions += 1

        world.decay_artifacts()

        newly_finished = []
        for aid in active_agents:
            if world.agent_on_goal(aid):
                world.mark_finished(aid)
                newly_finished.append(aid)

        for aid in newly_finished:
            reasoning_log.append(
                {
                    "turn": turn,
                    "agent_id": aid,
                    "comment": "FINISHED",
                }
            )

        if movement is not None:
            action_map: Dict[str, Optional[str]] = {}
            for aid in agent_ids:
                if aid in decisions:
                    action_map[aid] = _decision_action_label(decisions.get(aid))
                else:
                    action_map[aid] = None
            _append_snapshot(movement, world, turn + 1, agent_ids, actions=action_map)
            if movement_writer is not None and movement is not None:
                movement_writer.write(json.dumps(movement[-1]))
                movement_writer.write("\n")
                movement_writer.flush()

        if world.all_agents_on_goal(agent_ids):
            return EpisodeMetrics(
                turns=turn + 1,
                success=True,
                messages_sent=messages_sent,
                marks_placed=marks_placed,
                collisions=collisions,
                reasoning_log=reasoning_log,
            )

    return EpisodeMetrics(
        turns=turns,
        success=False,
        messages_sent=messages_sent,
        marks_placed=marks_placed,
        collisions=collisions,
        reasoning_log=reasoning_log,
    )


def _recipients_in_range(world: GridWorld, sender_id: str, radio_range: int) -> List[str]:
    sx, sy = world.occupancy[sender_id]
    recipients: List[str] = []
    for aid, (x, y) in world.occupancy.items():
        if aid == sender_id:
            continue
        if abs(x - sx) + abs(y - sy) <= radio_range:
            recipients.append(aid)
    return recipients


def _decision_action_label(decision: Optional[Decision]) -> Optional[str]:
    if decision is None:
        return None
    action = decision.action
    kind = action.kind
    if kind == "MOVE":
        return f"MOVE_{action.direction.value}"
    if kind == "STAY":
        return "STAY"
    if kind == "COMMUNICATE":
        return "COMMUNICATE"
    if kind == "MARK":
        return "MARK"
    return None


def _append_snapshot(
    store: List[dict],
    world: GridWorld,
    turn: int,
    agent_ids: List[str],
    actions: Optional[Dict[str, Optional[str]]],
) -> None:
    agents_payload = {}
    for aid in agent_ids:
        x, y = world.occupancy[aid]
        orientation = world.orientation.get(aid)
        status = "FINISHED" if world.is_finished(aid) else "ACTIVE"
        agents_payload[aid] = {
            "x": x,
            "y": y,
            "orientation": orientation.value if orientation is not None else None,
            "action": actions.get(aid) if actions is not None else None,
            "status": status,
        }

    store.append(
        {
            "turn": turn,
            "agents": agents_payload,
        }
    )
