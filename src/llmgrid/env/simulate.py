"""Episode driver that wires the environment and agent policies together."""

from __future__ import annotations

import asyncio
import json
import random
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, TextIO, Tuple

from pydantic import BaseModel, Field

from llmgrid.agent.llm_agent import DecisionTrace, LlmPolicy
from llmgrid.agent.local_baseline import GreedyBaseline
from llmgrid.env.grid import GridWorld
from llmgrid.schema import (
    AdjacentState,
    Decision,
    Direction,
    MessageBrief,
    MoveOutcome,
    Observation,
    OutgoingMessage,
    PlacedArtifact,
    Position,
    ReceivedMessage,
    StayAction,
    TurnHistory,
)


@dataclass
class EpisodeMetrics:
    turns: int
    success: bool
    messages_sent: int
    marks_placed: int
    collisions: int
    reasoning_log: List[Dict[str, Any]]
    collision_causes: Dict[str, int] = field(default_factory=dict)
    hazard_events: int = 0
    comments_clamped: int = 0
    comments_autofilled: int = 0
    no_go_exposures: int = 0
    contended_exposures: int = 0
    history_limit: int = 5
    loop_guidance: str = "passive"


@dataclass
class DecisionOutcome:
    decision: Decision
    record: Optional[dict]
    attempts: int


async def _call_policy_once_async(
    policy: "PolicyProtocol",
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
    policy: "PolicyProtocol",
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
            sleep_for = delay + random.uniform(0.0, jitter)
            await asyncio.sleep(max(0.0, sleep_for))
            delay *= 2
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("unreachable: retry loop exited without result")


async def _gather_decisions_async(
    active_agents: List[str],
    policy: "PolicyProtocol",
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


class ArtifactRecord(BaseModel):
    pos: Position
    artifact: PlacedArtifact


class GridWorldState(BaseModel):
    width: int
    height: int
    goal: Position
    walls: List[Position]
    occupancy: Dict[str, Position]
    orientation: Dict[str, Direction]
    inboxes: Dict[str, List[ReceivedMessage]] = Field(default_factory=dict)
    artifacts: List[ArtifactRecord] = Field(default_factory=list)
    finished_agents: Dict[str, bool] = Field(default_factory=dict)
    position_history: Dict[str, List[Position]] = Field(default_factory=dict)
    turn_history: Dict[str, List[TurnHistory]] = Field(default_factory=dict)
    rng_state: List[Any]
    bearing_flip_p: float
    bearing_drop_p: float
    bearing_bias_seed: Optional[int] = None
    bearing_bias_p: float = 0.0
    bearing_bias_wall_bonus: float = 0.0
    history_limit: int = 5

    @classmethod
    def capture(cls, world: GridWorld) -> "GridWorldState":
        return cls(
            width=world.size.width,
            height=world.size.height,
            goal=Position(x=world.goal.x, y=world.goal.y),
            walls=[Position(x=x, y=y) for x, y in sorted(world.walls)],
            occupancy={aid: Position(x=pos[0], y=pos[1]) for aid, pos in world.occupancy.items()},
            orientation=dict(world.orientation),
            inboxes={aid: list(messages) for aid, messages in world.inboxes.items()},
            artifacts=[
                ArtifactRecord(pos=Position(x=x, y=y), artifact=artifact)
                for (x, y), artifact in world.artifacts.items()
            ],
            finished_agents=dict(world.finished_agents),
            position_history={
                aid: [Position(x=px, y=py) for px, py in history]
                for aid, history in world.position_history.items()
            },
            turn_history={
                aid: [TurnHistory.model_validate(item) for item in history]
                for aid, history in world.turn_history.items()
            },
            rng_state=_freeze_random_state(world.rng.getstate()),
            bearing_flip_p=world.bearing_flip_p,
            bearing_drop_p=world.bearing_drop_p,
            bearing_bias_seed=world.bearing_bias_seed,
            bearing_bias_p=world.bearing_bias_p,
            bearing_bias_wall_bonus=world.bearing_bias_wall_bonus,
            history_limit=world.history_limit,
        )

    def restore(self) -> GridWorld:
        world = GridWorld(
            self.width,
            self.height,
            self.walls,
            self.goal,
            seed=0,
            bearing_flip_p=self.bearing_flip_p,
            bearing_drop_p=self.bearing_drop_p,
            bearing_bias_seed=self.bearing_bias_seed,
            bearing_bias_p=self.bearing_bias_p,
            bearing_bias_wall_bonus=self.bearing_bias_wall_bonus,
            history_limit=self.history_limit,
        )
        world.walls = {(p.x, p.y) for p in self.walls}
        world.occupancy = {aid: (pos.x, pos.y) for aid, pos in self.occupancy.items()}
        world.orientation = dict(self.orientation)
        world.inboxes = {aid: list(messages) for aid, messages in self.inboxes.items()}
        world.artifacts = {(rec.pos.x, rec.pos.y): rec.artifact for rec in self.artifacts}
        world.finished_agents = dict(self.finished_agents)
        world.position_history = {
            aid: [(pos.x, pos.y) for pos in history]
            for aid, history in self.position_history.items()
        }
        world.turn_history = {
            aid: deque(
                [history.model_dump() if isinstance(history, TurnHistory) else history for history in histories],
                maxlen=self.history_limit,
            )
            for aid, histories in self.turn_history.items()
        }
        world.rng.setstate(_thaw_random_state(self.rng_state))
        return world


class EpisodeCheckpoint(BaseModel):
    version: Literal["1.0"] = "1.0"
    use_llm: bool
    model_id: str
    turns_total: int
    turn_next: int
    visibility: int
    radio_range: int
    seed: int
    maze_metadata: Dict[str, Any] = Field(default_factory=dict)
    agent_ids: List[str]
    start_positions: Dict[str, Position]
    goal: Position
    world: GridWorldState
    messages_sent: int
    marks_placed: int
    collisions: int
    reasoning_log: List[Dict[str, Any]] = Field(default_factory=list)
    transcript: Optional[List[dict]] = None
    movement: Optional[List[dict]] = None
    baseline_rng_state: Optional[List[Any]] = None
    transcript_path: Optional[str] = None
    movement_stream_path: Optional[str] = None
    episode_path: Optional[str] = None
    concurrency_window: int = 1
    comm_strategy: str = "none"
    history_limit: int = 5
    loop_guidance: str = "passive"

    @classmethod
    def load(cls, path: Path) -> "EpisodeCheckpoint":
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")


def _resolve_policy(
    use_llm: bool,
    model_id: str,
    seed: int,
    strategy: str,
    loop_guidance: str,
    history_limit: int,
) -> "PolicyProtocol":
    if use_llm:
        return LlmPolicy(model_id, strategy=strategy, loop_guidance=loop_guidance, history_limit=history_limit)
    return GreedyBaseline(seed=seed)


class PolicyProtocol:
    """Protocol shim for type checking."""

    def decide(self, observation: Observation) -> Decision:  # pragma: no cover - baseline stub
        raise NotImplementedError

    async def decide_async(self, observation: Observation) -> Decision:  # pragma: no cover - LLM stub
        raise NotImplementedError

    async def decide_with_trace_async(self, observation: Observation) -> DecisionTrace:  # pragma: no cover - LLM stub
        raise NotImplementedError


async def _run_episode_async(
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
    agent_order: Optional[List[str]] = None,
    resume: Optional[EpisodeCheckpoint] = None,
    checkpoint_path: Optional[Path] = None,
    checkpoint_interval: int = 1,
    transcript_path: Optional[str] = None,
    movement_stream_path: Optional[str] = None,
    episode_path: Optional[str] = None,
    maze_metadata: Optional[Dict[str, Any]] = None,
    bearing_bias_seed: Optional[int] = None,
    bearing_bias_p: float = 0.0,
    bearing_bias_wall_bonus: float = 0.0,
    concurrency_start: Optional[int] = None,
    concurrency_max: Optional[int] = None,
    retry_max_attempts: int = 5,
    retry_base_delay: float = 1.0,
    retry_jitter: float = 0.5,
    comm_strategy: str = "none",
    history_limit: int = 5,
    loop_guidance: str = "passive",
) -> EpisodeMetrics:
    """Simulate a single episode and return aggregate metrics."""

    history_limit = max(1, history_limit)
    loop_guidance = loop_guidance.lower()
    comm_strategy = comm_strategy.lower()
    collision_cause_counts: Counter[str] = Counter()
    hazard_events = 0
    comments_clamped = 0
    comments_autofilled = 0
    no_go_exposures = 0
    contended_exposures = 0
    maze_meta = maze_metadata or {}

    if resume is not None:
        start_positions = resume.start_positions

    agent_ids = list(agent_order) if agent_order is not None else list(start_positions.keys())

    if resume is not None:
        world = resume.world.restore()
        if history_limit != world.history_limit:
            raise ValueError(
                f"Checkpoint recorded with history_limit={world.history_limit}; got history_limit={history_limit}."
            )
        history_limit = world.history_limit
        resume_loop_guidance = getattr(resume, "loop_guidance", "passive")
        if loop_guidance != resume_loop_guidance:
            raise ValueError(
                f"Checkpoint recorded with loop_guidance={resume_loop_guidance!r}; got loop_guidance={loop_guidance!r}."
            )
        loop_guidance = resume_loop_guidance
        messages_sent = resume.messages_sent
        marks_placed = resume.marks_placed
        collisions = resume.collisions
        reasoning_log = list(resume.reasoning_log)
        agent_ids = resume.agent_ids
        start_turn = resume.turn_next
        if transcript is not None and resume.transcript:
            if not transcript:
                transcript.extend(resume.transcript)
        if movement is not None and resume.movement:
            if not movement:
                movement.extend(resume.movement)
    else:
        world = GridWorld(
            width,
            height,
            obstacles,
            goal,
            seed=seed,
            bearing_bias_seed=bearing_bias_seed,
            bearing_bias_p=bearing_bias_p,
            bearing_bias_wall_bonus=bearing_bias_wall_bonus,
            history_limit=history_limit,
        )
        for idx, (agent_id, pos) in enumerate(start_positions.items()):
            orientation = [Direction.N, Direction.S, Direction.E, Direction.W][idx % 4]
            world.add_agent(agent_id, pos, orientation)
        messages_sent = 0
        marks_placed = 0
        collisions = 0
        reasoning_log: List[Dict[str, Any]] = []
        start_turn = 0
        if movement is not None:
            _append_snapshot(movement, world, 0, agent_ids, actions=None)
            if movement_writer is not None:
                movement_writer.write(json.dumps(movement[-1]))
                movement_writer.write("\n")
                movement_writer.flush()

    policy = _resolve_policy(use_llm, model_id, seed, comm_strategy, loop_guidance, history_limit)
    if resume is not None and isinstance(policy, GreedyBaseline) and resume.baseline_rng_state is not None:
        policy.set_state(_thaw_random_state(resume.baseline_rng_state))

    if concurrency_max is None:
        concurrency_max = len(agent_ids) if agent_ids else 1
    concurrency_max = max(1, concurrency_max)

    if resume is not None and resume.concurrency_window:
        concurrency_window = max(1, min(resume.concurrency_window, concurrency_max))
    else:
        if concurrency_start is None:
            # Default to 1 for safety - higher concurrency can cause connection pool issues
            # with Azure and other providers when using asyncio.run() in threads
            concurrency_start = 1
        concurrency_window = max(1, min(concurrency_start, concurrency_max))

    if not use_llm:
        concurrency_max = 1
        concurrency_window = 1

    if start_turn >= turns:
        success_now = world.all_agents_on_goal(agent_ids)
        if checkpoint_path is not None:
            checkpoint = EpisodeCheckpoint(
                use_llm=use_llm,
                model_id=model_id,
                turns_total=turns,
                turn_next=start_turn,
                visibility=visibility,
                radio_range=radio_range,
                seed=seed,
                maze_metadata=maze_meta,
                agent_ids=agent_ids,
                start_positions=start_positions,
                goal=goal,
                world=GridWorldState.capture(world),
                messages_sent=messages_sent,
                marks_placed=marks_placed,
                collisions=collisions,
                reasoning_log=reasoning_log,
                transcript=transcript,
                movement=movement,
                baseline_rng_state=_freeze_random_state(policy.get_state()) if hasattr(policy, "get_state") else None,
                transcript_path=transcript_path,
                movement_stream_path=movement_stream_path,
                episode_path=episode_path,
                concurrency_window=concurrency_window,
                comm_strategy=comm_strategy,
                history_limit=history_limit,
                loop_guidance=loop_guidance,
            )
            checkpoint.write(checkpoint_path)
        cause_dict = dict(collision_cause_counts)
        return EpisodeMetrics(
            turns=start_turn,
            success=success_now,
            messages_sent=messages_sent,
            marks_placed=marks_placed,
            collisions=collisions,
            reasoning_log=reasoning_log,
            collision_causes=cause_dict,
            hazard_events=hazard_events,
            comments_clamped=comments_clamped,
            comments_autofilled=comments_autofilled,
            no_go_exposures=no_go_exposures,
            contended_exposures=contended_exposures,
            history_limit=history_limit,
            loop_guidance=loop_guidance,
        )

    for turn in range(start_turn, turns):
        goal_distances_before: Dict[str, int] = {
            aid: abs(world.occupancy[aid][0] - world.goal.x) + abs(world.occupancy[aid][1] - world.goal.y)
            for aid in agent_ids
        }
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

        for obs in observations.values():
            for adjacent in obs.adjacent:
                if adjacent.state == AdjacentState.NO_GO:
                    no_go_exposures += 1
                if adjacent.state == AdjacentState.CONTENDED:
                    contended_exposures += 1

        capture_trace = transcript is not None and hasattr(policy, "decide_with_trace_async")
        if use_llm:
            try:
                outcomes, had_retry = await _gather_decisions_async(
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
            except Exception:
                concurrency_window = max(1, concurrency_window // 2)
                raise
        else:
            outcomes = {
                aid: await _call_policy_once_async(policy, observations[aid], capture_trace, aid, turn)
                for aid in active_agents
            }
            had_retry = False

        decisions: Dict[str, Decision] = {}
        for aid in active_agents:
            outcome = outcomes[aid]
            decisions[aid] = outcome.decision
            if outcome.record is not None:
                if transcript is not None:
                    transcript.append(outcome.record)
                if transcript_writer is not None:
                    transcript_writer.write(json.dumps(outcome.record))
                    transcript_writer.write("\n")
                    transcript_writer.flush()
            comment = _truncate(decisions[aid].comment)
            reasoning_log.append(
                {
                    "turn": turn,
                    "agent_id": aid,
                    "comment": comment if comment is not None else "",
                }
            )

        if use_llm:
            if had_retry:
                concurrency_window = max(1, concurrency_window // 2)
            elif concurrency_window < concurrency_max:
                concurrency_window = min(concurrency_window + 1, concurrency_max)

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

        intents: Dict[str, Optional[Direction]] = {}
        before = dict(world.occupancy)
        for aid, decision in decisions.items():
            intents[aid] = decision.action.direction if decision.action.kind == "MOVE" else None

        move_results = world.resolve_moves(intents)
        after = dict(world.occupancy)
        for aid, direction in intents.items():
            if direction is None:
                continue
            if before[aid] == after[aid]:
                collisions += 1

        contested_cells = {
            result.cause_cell
            for result in move_results.values()
            if result.cause_cell is not None and result.outcome in (MoveOutcome.BLOCK_AGENT, MoveOutcome.SWAP_CONFLICT)
        }
        for result in move_results.values():
            if result.outcome not in (MoveOutcome.OK, MoveOutcome.FINISHED, MoveOutcome.YIELD):
                collision_cause_counts[result.outcome.value] += 1
        hazard_events += len(contested_cells)
        world.contended_neighbors = {
            aid: _compute_contended_mask(world.occupancy[aid], contested_cells)
            for aid in world.occupancy.keys()
        }

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

        for aid in active_agents:
            result = move_results[aid]
            observation = observations[aid]
            distance_before = goal_distances_before.get(aid, world.last_goal_distance.get(aid, 0))
            pos = world.occupancy[aid]
            distance_after = abs(pos[0] - world.goal.x) + abs(pos[1] - world.goal.y)
            delta = _goal_delta(distance_before, distance_after)
            loop_val = world.loop_counters.get(aid, 0)
            if delta == "CLOSER":
                loop_val = 0
            else:
                loop_val = min(9, loop_val + 1)
            world.loop_counters[aid] = loop_val
            world.last_goal_distance[aid] = distance_after
            world.last_move_outcome[aid] = result.outcome
            world.last_intent_target[aid] = result.target
            peer_bits = _peer_bits(observation)
            note = _derive_note(result, loop_val)
            history_payload = _make_history_entry(
                turn=turn,
                decision=decisions[aid],
                observation=observation,
                result=result,
                delta=delta,
                loop_value=loop_val,
                peer_bits=peer_bits,
                note=note,
            )
            world.record_history(aid, history_payload)

        if movement is not None:
            action_map: Dict[str, Optional[str]] = {}
            for aid in agent_ids:
                if aid in decisions:
                    action_map[aid] = _decision_action_label(decisions.get(aid))
                else:
                    action_map[aid] = None
            _append_snapshot(movement, world, turn + 1, agent_ids, actions=action_map)
            if movement_writer is not None:
                movement_writer.write(json.dumps(movement[-1]))
                movement_writer.write("\n")
                movement_writer.flush()

        should_checkpoint = checkpoint_path is not None and (
            (turn + 1) % max(1, checkpoint_interval) == 0
            or world.all_agents_on_goal(agent_ids)
            or turn + 1 == turns
        )

        if should_checkpoint and checkpoint_path is not None:
            baseline_state = None
            if hasattr(policy, "get_state"):
                baseline_state = _freeze_random_state(policy.get_state())  # type: ignore[attr-defined]
            checkpoint = EpisodeCheckpoint(
                use_llm=use_llm,
                model_id=model_id,
                turns_total=turns,
                turn_next=turn + 1,
                visibility=visibility,
                radio_range=radio_range,
                seed=seed,
                maze_metadata=maze_meta,
                agent_ids=agent_ids,
                start_positions=start_positions,
                goal=goal,
                world=GridWorldState.capture(world),
                messages_sent=messages_sent,
                marks_placed=marks_placed,
                collisions=collisions,
                reasoning_log=list(reasoning_log),
                transcript=transcript,
                movement=movement,
                baseline_rng_state=baseline_state,
                transcript_path=transcript_path,
                movement_stream_path=movement_stream_path,
                episode_path=episode_path,
                concurrency_window=concurrency_window,
                comm_strategy=comm_strategy,
                history_limit=history_limit,
                loop_guidance=loop_guidance,
            )
            checkpoint.write(checkpoint_path)

        if world.all_agents_on_goal(agent_ids):
            cause_dict = dict(collision_cause_counts)
            return EpisodeMetrics(
                turns=turn + 1,
                success=True,
                messages_sent=messages_sent,
                marks_placed=marks_placed,
                collisions=collisions,
                reasoning_log=reasoning_log,
                collision_causes=cause_dict,
                hazard_events=hazard_events,
                comments_clamped=comments_clamped,
                comments_autofilled=comments_autofilled,
                no_go_exposures=no_go_exposures,
                contended_exposures=contended_exposures,
                history_limit=history_limit,
                loop_guidance=loop_guidance,
            )

    cause_dict = dict(collision_cause_counts)
    return EpisodeMetrics(
        turns=turns,
        success=False,
        messages_sent=messages_sent,
        marks_placed=marks_placed,
        collisions=collisions,
        reasoning_log=reasoning_log,
        collision_causes=cause_dict,
        hazard_events=hazard_events,
        comments_clamped=comments_clamped,
        comments_autofilled=comments_autofilled,
        no_go_exposures=no_go_exposures,
        contended_exposures=contended_exposures,
        history_limit=history_limit,
        loop_guidance=loop_guidance,
    )


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
    agent_order: Optional[List[str]] = None,
    resume: Optional[EpisodeCheckpoint] = None,
    checkpoint_path: Optional[Path] = None,
    checkpoint_interval: int = 1,
    transcript_path: Optional[str] = None,
    movement_stream_path: Optional[str] = None,
    episode_path: Optional[str] = None,
    maze_metadata: Optional[Dict[str, Any]] = None,
    bearing_bias_seed: Optional[int] = None,
    bearing_bias_p: float = 0.0,
    bearing_bias_wall_bonus: float = 0.0,
    concurrency_start: Optional[int] = None,
    concurrency_max: Optional[int] = None,
    retry_max_attempts: int = 5,
    retry_base_delay: float = 1.0,
    retry_jitter: float = 0.5,
    comm_strategy: str = "none",
    history_limit: int = 5,
    loop_guidance: str = "passive",
) -> EpisodeMetrics:
    """Synchronously run the async driver in a fresh event loop."""
    return asyncio.run(
        _run_episode_async(
            use_llm=use_llm,
            model_id=model_id,
            width=width,
            height=height,
            obstacles=obstacles,
            start_positions=start_positions,
            goal=goal,
            turns=turns,
            visibility=visibility,
            radio_range=radio_range,
            seed=seed,
            transcript=transcript,
            movement=movement,
            transcript_writer=transcript_writer,
            movement_writer=movement_writer,
            agent_order=agent_order,
            resume=resume,
            checkpoint_path=checkpoint_path,
            checkpoint_interval=checkpoint_interval,
            transcript_path=transcript_path,
            movement_stream_path=movement_stream_path,
            episode_path=episode_path,
            maze_metadata=maze_metadata,
            bearing_bias_seed=bearing_bias_seed,
            bearing_bias_p=bearing_bias_p,
            bearing_bias_wall_bonus=bearing_bias_wall_bonus,
            concurrency_start=concurrency_start,
            concurrency_max=concurrency_max,
            retry_max_attempts=retry_max_attempts,
            retry_base_delay=retry_base_delay,
            retry_jitter=retry_jitter,
            comm_strategy=comm_strategy,
            history_limit=history_limit,
            loop_guidance=loop_guidance,
        )
    )


def _freeze_random_state(state: Any) -> List[Any]:
    if isinstance(state, tuple):
        return [_freeze_random_state(item) for item in state]
    return state


def _thaw_random_state(state: Any) -> Any:
    if isinstance(state, list):
        return tuple(_thaw_random_state(item) for item in state)
    return state


def _recipients_in_range(world: GridWorld, sender_id: str, radio_range: int) -> List[str]:
    sx, sy = world.occupancy[sender_id]
    recipients: List[str] = []
    for aid, (x, y) in world.occupancy.items():
        if aid == sender_id:
            continue
        if abs(x - sx) + abs(y - sy) <= radio_range:
            recipients.append(aid)
    return recipients


def _compute_contended_mask(position: Tuple[int, int], contested_cells: Iterable[Optional[Tuple[int, int]]]) -> int:
    """Return a NESW bitmask for contested neighbour cells relative to `position`."""

    if not contested_cells:
        return 0

    x, y = position
    mask = 0
    for cell in contested_cells:
        if cell is None:
            continue
        cx, cy = cell
        dx = cx - x
        dy = cy - y
        if dx == 0 and dy == -1:
            mask |= 0b0001  # north
        elif dx == 1 and dy == 0:
            mask |= 0b0010  # east
        elif dx == 0 and dy == 1:
            mask |= 0b0100  # south
        elif dx == -1 and dy == 0:
            mask |= 0b1000  # west
    return mask


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


def _truncate(text: Optional[str], limit: int = 160) -> Optional[str]:
    if not text:
        return None
    trimmed = text.strip()
    if not trimmed:
        return None
    if len(trimmed) <= limit:
        return trimmed
    return trimmed[: limit - 3].rstrip() + "..."


def _summarise_outgoing(message: OutgoingMessage) -> MessageBrief:
    details = None
    kind = getattr(message, "kind", "")
    if kind == "HERE":
        details = f"pos=({message.pos.x},{message.pos.y})"
    elif kind == "INTENT":
        details = message.next_action
    elif kind == "SENSE":
        parts = []
        if message.mode:
            parts.append(message.mode)
        if getattr(message, "bearing", None):
            parts.append(f"bearing={message.bearing}")
        if getattr(message, "strength", None):
            parts.append(f"strength={message.strength}")
        if getattr(message, "value_bin", None):
            parts.append(f"value={message.value_bin}")
        if getattr(message, "approx_bearing", None):
            parts.append(f"approx={message.approx_bearing}")
        details = " ".join(parts) if parts else None
    elif kind == "BLOCKED":
        details = f"({message.where.x},{message.where.y}) {message.reason}"
    elif kind == "REQUEST":
        parts = [message.req]
        if message.target is not None:
            parts.append(f"target=({message.target.x},{message.target.y})")
        details = " ".join(parts)
    elif kind == "MARK_INFO":
        details = getattr(message.placed, "kind", None)
    return MessageBrief(kind=kind, details=details)


def _summarise_received(received: ReceivedMessage) -> MessageBrief:
    brief = _summarise_outgoing(received.envelope)
    brief.sender = received.envelope.sender_id
    brief.hop = received.hop_distance
    brief.age = received.age
    return brief


def _intent_token(action_label: str) -> str:
    valid = {"MOVE_N", "MOVE_E", "MOVE_S", "MOVE_W", "STAY", "COMMUNICATE", "MARK"}
    if action_label in valid:
        return action_label
    if action_label.startswith("MOVE_") and len(action_label) == 6 and action_label[-1] in {"N", "E", "S", "W"}:
        return action_label
    return "STAY"


def _goal_delta(before: int, after: int) -> str:
    if after < before:
        return "CLOSER"
    if after > before:
        return "FARTHER"
    return "SAME"


def _peer_bits(observation: Observation) -> str:
    flags = {"N": 0, "E": 0, "S": 0, "W": 0}
    for cell in observation.adjacent:
        if cell.state == AdjacentState.AGENT:
            flags[cell.dir] = 1
    bits = f"N{flags['N']}E{flags['E']}S{flags['S']}W{flags['W']}"
    recent_intent = observation.history[0].intent if observation.history else "-"
    return f"{bits}|intent:{recent_intent}"


def _derive_note(result: "MoveResult", loop_value: int) -> Optional[str]:
    if loop_value >= 3:
        return "AVOID_LOOP"
    if result.outcome in (MoveOutcome.BLOCK_AGENT, MoveOutcome.SWAP_CONFLICT):
        return "TRAFFIC_CONE"
    return None


def _make_history_entry(
    *,
    turn: int,
    decision: Decision,
    observation: Observation,
    result: "MoveResult",
    delta: str,
    loop_value: int,
    peer_bits: str,
    note: Optional[str],
) -> dict:
    action_label = _decision_action_label(decision) or decision.action.kind.upper()
    intent_token = _intent_token(action_label)
    comment = _truncate(decision.comment)
    sent_brief = None
    if decision.action.kind == "COMMUNICATE":
        sent_brief = _summarise_outgoing(decision.action.message)
    received = [_summarise_received(msg) for msg in observation.inbox]
    entry = TurnHistory(
        turn_index=turn,
        intent=intent_token,
        outcome=result.outcome,
        delta=delta,
        loop=min(loop_value, 9),
        peer_bits=peer_bits,
        note=note,
    )
    payload = entry.model_dump()
    payload.update(
        {
            "action": action_label,
            "comment": comment,
            "sent_message": sent_brief,
            "received_messages": received,
        }
    )
    return payload


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

    store.append({"turn": turn, "agents": agents_payload})
