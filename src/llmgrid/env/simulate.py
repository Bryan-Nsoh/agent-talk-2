"""Episode driver for the map-sharing, no-comm environment."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, TextIO, Tuple

from llmgrid.agent.llm_agent import DecisionTrace, LlmPolicy
from llmgrid.agent.local_baseline import GreedyBaseline
from llmgrid.env.grid import GridWorld
from llmgrid.schema import Decision, Direction, MoveOutcome, Observation, Position


@dataclass
class EpisodeMetrics:
    turns: int
    success: bool
    collisions: int
    reasoning_log: List[Dict[str, Any]]
    collision_causes: Dict[str, int]
    # Backwards-compat/CLI fields (not used in this spec)
    messages_sent: int = 0
    marks_placed: int = 0
    hazard_events: int = 0
    comments_clamped: int = 0
    comments_autofilled: int = 0
    no_go_exposures: int = 0
    contended_exposures: int = 0
    history_limit: int = 0
    loop_guidance: str = "passive"


@dataclass
class DecisionOutcome:
    decision: Decision
    record: Optional[dict]
    attempts: int


class PolicyProtocol:
    def decide(self, observation: Observation) -> Decision:  # pragma: no cover - baseline stub
        raise NotImplementedError

    async def decide_async(self, observation: Observation) -> Decision:  # pragma: no cover - LLM stub
        raise NotImplementedError

    async def decide_with_trace_async(self, observation: Observation) -> DecisionTrace:  # pragma: no cover - LLM stub
        raise NotImplementedError


async def _call_policy_once_async(
    policy: PolicyProtocol,
    observation: Observation,
    capture_trace: bool,
    agent_id: str,
    turn: int,
) -> DecisionOutcome:
    if hasattr(policy, "decide_async"):
        if capture_trace and hasattr(policy, "decide_with_trace_async"):
            trace: DecisionTrace = await getattr(policy, "decide_with_trace_async")(observation)  # type: ignore[attr-defined]
            record = {
                "turn": turn,
                "agent_id": agent_id,
                "prompt": trace.prompt,
                "observation": observation.model_dump(mode="json"),
                "decision": trace.decision.model_dump(mode="json"),
                "trace_messages": trace.trace_messages,
            }
            return DecisionOutcome(decision=trace.decision, record=record, attempts=1)
        decision = await getattr(policy, "decide_async")(observation)  # type: ignore[attr-defined]
        return DecisionOutcome(decision=decision, record=None, attempts=1)

    decision = policy.decide(observation)
    return DecisionOutcome(decision=decision, record=None, attempts=1)


async def _decide_with_retry_async(
    policy: PolicyProtocol,
    observation: Observation,
    capture_trace: bool,
    agent_id: str,
    turn: int,
    max_attempts: int,
    base_delay: float,
    jitter: float,
) -> DecisionOutcome:
    delay = base_delay
    attempts = 0
    last_exc: Optional[Exception] = None
    while attempts < max_attempts:
        attempts += 1
        try:
            outcome = await _call_policy_once_async(
                policy,
                observation,
                capture_trace,
                agent_id,
                turn,
            )
            outcome.attempts = attempts
            return outcome
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempts >= max_attempts:
                raise
            sleep_for = delay + jitter
            await asyncio.sleep(max(0.0, sleep_for))
            delay *= 2
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("unreachable")


async def _gather_decisions_async(
    active_agents: List[str],
    policy: PolicyProtocol,
    observations: Dict[str, Observation],
    capture_trace: bool,
    turn: int,
    concurrency_window: int,
    max_attempts: int,
    base_delay: float,
    jitter: float,
) -> tuple[Dict[str, DecisionOutcome], bool]:
    if not active_agents:
        return {}, False

    semaphore = asyncio.Semaphore(max(1, concurrency_window))
    outcomes: Dict[str, DecisionOutcome] = {}
    any_retry = False

    async def worker(aid: str) -> tuple[str, DecisionOutcome]:
        async with semaphore:
            outcome = await _decide_with_retry_async(
                policy,
                observations[aid],
                capture_trace,
                aid,
                turn,
                max_attempts,
                base_delay,
                jitter,
            )
            return aid, outcome

    tasks = [asyncio.create_task(worker(aid)) for aid in active_agents]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for result in results:
        if isinstance(result, Exception):
            raise result
        aid, outcome = result
        outcomes[aid] = outcome
        if outcome.attempts > 1:
            any_retry = True
    return outcomes, any_retry


def _resolve_policy(
    use_llm: bool,
    model_id: str,
    seed: int,
    *,
    history_limit: int,
    radio_range: int,
) -> PolicyProtocol:
    if use_llm:
        return LlmPolicy(
            model_id,
            strategy="none",
            loop_guidance="passive",
            history_limit=history_limit,
            radio_range=radio_range,
        )
    return GreedyBaseline(seed=seed)


def _direction_from_move(decision: Decision) -> Optional[Direction]:
    if decision.action.kind == "MOVE":  # type: ignore[attr-defined]
        return decision.action.direction  # type: ignore[attr-defined]
    return None


def _intents_from_decisions(decisions: Dict[str, Decision]) -> Dict[str, Optional[Direction]]:
    intents: Dict[str, Optional[Direction]] = {}
    for aid, dec in decisions.items():
        intents[aid] = _direction_from_move(dec)
    return intents


async def run_episode_async(
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
    map_sharing: str,
    comm_strategy: str = "none",
    history_limit: int = 10,
    loop_guidance: str = "passive",
    agent_order: Optional[List[str]] = None,
    seed: int = 0,
    transcript: Optional[List[dict]] = None,
    transcript_writer: Optional[TextIO] = None,
    movement: Optional[List[dict]] = None,
    movement_writer: Optional[TextIO] = None,
    checkpoint_path: Optional[Path] = None,
    concurrency_start: Optional[int] = None,
    concurrency_max: Optional[int] = None,
    retry_max_attempts: int = 1,
    retry_base_delay: float = 0.25,
    retry_jitter: float = 0.05,
) -> EpisodeMetrics:
    radio_range = max(0, radio_range)
    map_sharing = map_sharing.lower()
    if map_sharing not in {"none", "radio_sync", "global"}:
        raise ValueError(f"Unsupported map_sharing: {map_sharing}")

    world = GridWorld(
        width,
        height,
        obstacles,
        goal,
        seed=seed,
        history_limit=history_limit,
    )
    agent_ids = list(agent_order) if agent_order else list(start_positions.keys())
    for aid, pos in start_positions.items():
        world.add_agent(aid, pos)

    policy = _resolve_policy(
        use_llm,
        model_id,
        seed,
        history_limit=history_limit,
        radio_range=radio_range,
    )

    if concurrency_max is None:
        concurrency_max = len(agent_ids) if agent_ids else 1
    concurrency_max = max(1, concurrency_max)
    if concurrency_start is None:
        concurrency_start = 1
    concurrency_window = max(1, min(concurrency_start, concurrency_max))
    if not use_llm:
        concurrency_window = 1
        concurrency_max = 1

    collisions = 0
    reasoning_log: List[Dict[str, Any]] = []
    collision_cause_counts: Dict[str, int] = {}

    for turn in range(0, turns):
        active_agents = [aid for aid in agent_ids if not world.is_finished(aid)]
        observations: Dict[str, Observation] = {}
        for aid in active_agents:
            observations[aid] = world.build_observation(
                aid,
                turn_index=turn,
                max_turns=turns,
                visibility_radius=visibility,
                map_sharing=map_sharing,
            )

        # Map sharing after building obs? we want shared base before policy.
        if map_sharing != "none":
            if map_sharing == "global":
                _merge_global(world)
            elif map_sharing == "radio_sync":
                _merge_radio(world, active_agents, radio_range)
            # rebuild observations with merged maps
            for aid in active_agents:
                observations[aid] = world.build_observation(
                    aid,
                    turn_index=turn,
                    max_turns=turns,
                    visibility_radius=visibility,
                    map_sharing=map_sharing,
                )

        capture_trace = transcript is not None and hasattr(policy, "decide_with_trace_async")
        if use_llm:
            outcomes, _ = await _gather_decisions_async(
                active_agents,
                policy,
                observations,
                capture_trace,
                turn,
                concurrency_window,
                retry_max_attempts,
                retry_base_delay,
                retry_jitter,
            )
        else:
            outcomes = {
                aid: await _call_policy_once_async(policy, observations[aid], capture_trace, aid, turn)
                for aid in active_agents
            }

        decisions = {aid: outcome.decision for aid, outcome in outcomes.items()}

        intents = _intents_from_decisions(decisions)
        move_results = world.resolve_moves(intents)

        for aid, res in move_results.items():
            if res.outcome in {MoveOutcome.BLOCK_AGENT}:
                collisions += 1
                collision_cause_counts[res.outcome.value] = collision_cause_counts.get(res.outcome.value, 0) + 1

        if transcript is not None:
            for aid in active_agents:
                rec = outcomes[aid].record
                if rec is not None:
                    transcript.append(rec)
                    if transcript_writer:
                        transcript_writer.write(json.dumps(rec))
                        transcript_writer.write("\n")
                        transcript_writer.flush()

        if movement is not None:
            pos_snapshot = worldsnapshot_positions(world)
            goal_hits = [
                aid
                for aid, pos in pos_snapshot.items()
                if pos == world.goal
            ]
            # per-agent manhattan distance to goal
            dist_to_goal = {
                aid: abs(pos[0] - world.goal[0]) + abs(pos[1] - world.goal[1])
                for aid, pos in pos_snapshot.items()
            }
            snapshot = {
                "turn": turn,
                "positions": {aid: {"x": pos[0], "y": pos[1]} for aid, pos in pos_snapshot.items()},
                "finished": [aid for aid, done in world.finished.items() if done],
                "goal_hits": goal_hits,
                "dist_to_goal": dist_to_goal,
            }
            movement.append(snapshot)
            if movement_writer:
                movement_writer.write(json.dumps(snapshot))
                movement_writer.write("\n")
                movement_writer.flush()

        if world.all_agents_on_goal(agent_ids):
            return EpisodeMetrics(
                turns=turn + 1,
                success=True,
                collisions=collisions,
                reasoning_log=reasoning_log,
                collision_causes=collision_cause_counts,
                history_limit=history_limit,
            )

    return EpisodeMetrics(
        turns=turns,
        success=world.all_agents_on_goal(agent_ids),
        collisions=collisions,
        reasoning_log=reasoning_log,
        collision_causes=collision_cause_counts,
        history_limit=history_limit,
    )


def _merge_radio(world: GridWorld, agents: List[str], radio_range: int) -> None:
    for i, aid in enumerate(agents):
        ax, ay = world.occupancy.get(aid, (None, None))
        if ax is None:
            continue
        for bid in agents[i + 1 :]:
            bx, by = world.occupancy.get(bid, (None, None))
            if bx is None:
                continue
            if abs(ax - bx) + abs(ay - by) <= radio_range:
                world.merge_base_maps(aid, bid)
                world.merge_base_maps(bid, aid)


def _merge_global(world: GridWorld) -> None:
    # pick first agent map as accumulator
    agents = list(world.agent_maps.keys())
    if not agents:
        return
    acc = world.agent_maps[agents[0]]
    for aid in agents[1:]:
        acc.merge_base_from(world.agent_maps[aid])
    for aid in agents[1:]:
        world.agent_maps[aid].merge_base_from(acc)


# --------------------------------------------------------------------------- #
# Convenience sync wrapper
# --------------------------------------------------------------------------- #


def run_episode(**kwargs) -> EpisodeMetrics:
    return asyncio.run(run_episode_async(**kwargs))


# backwards compatibility name used in tests/logs
_run_episode_async = run_episode_async


# --------------------------------------------------------------------------- #
# Checkpoint stub (compatibility)
# --------------------------------------------------------------------------- #


class EpisodeCheckpoint:  # pragma: no cover - placeholder
    @classmethod
    def load(cls, path: Path) -> "EpisodeCheckpoint":
        raise NotImplementedError("Checkpointing is not supported in this refactor.")

    def write(self, path: Path) -> None:
        raise NotImplementedError("Checkpointing is not supported in this refactor.")


# Helper to include finished agents in movement snapshots
def worldsnapshot_positions(world: GridWorld) -> dict[str, tuple[int, int]]:
    positions = dict(world.occupancy)
    positions.update(world.finished_positions)
    return positions
