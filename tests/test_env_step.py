import pytest

from llmgrid.env import simulate as simulate_module
from llmgrid.env.grid import GridWorld, TRAFFIC_CONE_TTL
from llmgrid.env.simulate import EpisodeCheckpoint, run_episode
from llmgrid.schema import (
    CommunicateAction,
    Decision,
    Direction,
    MoveOutcome,
    MsgHere,
    MsgIntent,
    Position,
    ReceivedMessage,
    TurnHistory,
)


def test_gridworld_resolves_simple_move():
    world = GridWorld(
        width=5,
        height=5,
        obstacles=[],
        goal=Position(x=4, y=4),
        seed=0,
    )
    world.add_agent("a1", Position(x=1, y=1), Direction.E)
    world.add_agent("a2", Position(x=3, y=1), Direction.W)

    intents = {"a1": Direction.E, "a2": Direction.W}
    results = world.resolve_moves(intents)
    assert world.occupancy["a1"] == (1, 1)
    assert world.occupancy["a2"] == (3, 1)
    assert results["a1"].outcome == MoveOutcome.BLOCK_AGENT
    assert results["a2"].outcome == MoveOutcome.BLOCK_AGENT


class InterruptingBaseline(simulate_module.GreedyBaseline):
    def __init__(self, *, seed: int, fail_after: int) -> None:
        super().__init__(seed=seed)
        self.fail_after = fail_after
        self.calls = 0

    def decide(self, observation):  # type: ignore[override]
        if self.calls >= self.fail_after:
            raise RuntimeError("simulated failure")
        self.calls += 1
        return super().decide(observation)


def test_run_episode_resume(tmp_path, monkeypatch):
    width = 6
    height = 6
    seed = 7
    turns = 6
    visibility = 1
    radio_range = 2
    start_positions = {
        "a1": Position(x=0, y=0),
        "a2": Position(x=width - 1, y=height - 1),
    }
    goal = Position(x=width - 1, y=0)
    obstacles: list[Position] = []
    checkpoint_path = tmp_path / "episode_checkpoint.json"

    expected_metrics = run_episode(
        use_llm=False,
        model_id="openrouter:test",
        width=width,
        height=height,
        obstacles=obstacles,
        start_positions=start_positions,
        goal=goal,
        turns=turns,
        visibility=visibility,
        radio_range=radio_range,
        seed=seed,
        agent_order=list(start_positions.keys()),
    )

    original_resolve = simulate_module._resolve_policy

    def patched_resolve(use_llm: bool, model_id: str, seed: int, strategy: str, loop_guidance: str, history_limit: int):
        assert not use_llm
        assert strategy == "none"
        assert loop_guidance == "passive"
        assert history_limit == 5
        return InterruptingBaseline(seed=seed, fail_after=4)

    monkeypatch.setattr(simulate_module, "_resolve_policy", patched_resolve)

    with pytest.raises(RuntimeError):
        run_episode(
            use_llm=False,
            model_id="openrouter:test",
            width=width,
            height=height,
            obstacles=obstacles,
            start_positions=start_positions,
            goal=goal,
            turns=turns,
            visibility=visibility,
            radio_range=radio_range,
            seed=seed,
            transcript=[],
            movement=[],
            checkpoint_path=checkpoint_path,
            checkpoint_interval=1,
            agent_order=list(start_positions.keys()),
        )

    monkeypatch.setattr(simulate_module, "_resolve_policy", original_resolve)

    checkpoint = EpisodeCheckpoint.load(checkpoint_path)
    assert checkpoint.turn_next == 2
    assert checkpoint.history_limit == 5
    assert checkpoint.loop_guidance == "passive"

    for agent_id, history in checkpoint.world.turn_history.items():
        assert history, f"expected turn history for {agent_id}"
        assert len(history) <= checkpoint.history_limit
        assert all(isinstance(entry, TurnHistory) for entry in history)

    transcript_records = list(checkpoint.transcript or [])
    movement_records = list(checkpoint.movement or [])

    resume_metrics = run_episode(
        use_llm=False,
        model_id="openrouter:test",
        width=checkpoint.world.width,
        height=checkpoint.world.height,
        obstacles=[Position(x=p.x, y=p.y) for p in checkpoint.world.walls],
        start_positions=checkpoint.start_positions,
        goal=checkpoint.goal,
        turns=turns,
        visibility=checkpoint.visibility,
        radio_range=checkpoint.radio_range,
        seed=checkpoint.seed,
        transcript=transcript_records,
        movement=movement_records,
        agent_order=checkpoint.agent_ids,
        resume=checkpoint,
        checkpoint_path=checkpoint_path,
        checkpoint_interval=1,
        maze_metadata=checkpoint.maze_metadata,
    )

    final_checkpoint = EpisodeCheckpoint.load(checkpoint_path)
    assert final_checkpoint.history_limit == 5
    assert final_checkpoint.loop_guidance == "passive"
    for history in final_checkpoint.world.turn_history.values():
        assert history
        assert len(history) <= final_checkpoint.history_limit
        assert all(isinstance(entry, TurnHistory) for entry in history)

    assert resume_metrics.turns == expected_metrics.turns
    assert resume_metrics.success == expected_metrics.success
    assert resume_metrics.messages_sent == expected_metrics.messages_sent
    assert resume_metrics.marks_placed == expected_metrics.marks_placed
    assert resume_metrics.collisions == expected_metrics.collisions


def test_observation_history_includes_turn_summary():
    world = GridWorld(
        width=5,
        height=5,
        obstacles=[],
        goal=Position(x=4, y=4),
        seed=11,
    )
    world.add_agent("a1", Position(x=2, y=2), Direction.N)
    world.add_agent("a2", Position(x=2, y=3), Direction.S)

    incoming = MsgIntent(sender_id="a2", seq=1, next_action="MOVE_N")
    world.inboxes.setdefault("a1", []).append(
        ReceivedMessage(envelope=incoming, hop_distance=1, age=0)
    )

    obs0 = world.build_observation(
        "a1",
        turn_index=0,
        max_turns=10,
        visibility_radius=1,
        radio_range=2,
    )
    assert obs0.history == []

    decision = Decision(
        action=CommunicateAction(
            message=MsgHere(
                sender_id="a1",
                seq=0,
                pos=Position(x=2, y=2),
                orientation=Direction.N,
            ),
        ),
        comment="Acknowledged",
    )

    history_entry = TurnHistory(
        turn_index=0,
        intent="COMMUNICATE",
        outcome=MoveOutcome.OK,
        delta="SAME",
        loop=0,
        peer_bits="N0E0S0W0|intent:MOVE_N",
        note="TEST",
    )
    world.record_history("a1", history_entry.model_dump())

    obs1 = world.build_observation(
        "a1",
        turn_index=1,
        max_turns=10,
        visibility_radius=1,
        radio_range=2,
    )

    assert len(obs1.history) == 1
    entry = obs1.history[0]
    assert entry.turn_index == 0
    assert entry.intent == "COMMUNICATE"
    assert entry.outcome == MoveOutcome.OK
    assert entry.delta == "SAME"
    assert entry.loop == 0
    assert entry.peer_bits.startswith("N")
    assert entry.note == "TEST"
    assert obs1.last_move_outcome == MoveOutcome.OK
    assert obs1.contended_neighbors == 0


def test_agent_conflict_sets_cone_and_outcomes():
    world = GridWorld(
        width=4,
        height=4,
        obstacles=[],
        goal=Position(x=3, y=3),
        seed=3,
    )
    world.add_agent("a1", Position(x=1, y=1), Direction.E)
    world.add_agent("a2", Position(x=2, y=2), Direction.N)

    intents = {"a1": Direction.E, "a2": Direction.N}
    results = world.resolve_moves(intents)

    assert results["a1"].outcome == MoveOutcome.BLOCK_AGENT
    assert results["a2"].outcome == MoveOutcome.BLOCK_AGENT
    assert results["a1"].cause_cell == (2, 1)
    assert results["a2"].cause_cell == (2, 1)

    cone = world.artifacts.get((2, 1))
    assert cone is not None
    assert cone.kind == "NO_GO"
    assert cone.ttl_remaining == TRAFFIC_CONE_TTL

    contested = {
        result.cause_cell
        for result in results.values()
        if result.cause_cell is not None and result.outcome in (MoveOutcome.BLOCK_AGENT, MoveOutcome.SWAP_CONFLICT)
    }
    mask_a1 = simulate_module._compute_contended_mask(world.occupancy["a1"], contested)
    assert mask_a1 & 0b0010  # east bit set


def test_block_wall_outcome():
    world = GridWorld(
        width=3,
        height=3,
        obstacles=[Position(x=2, y=1)],
        goal=Position(x=2, y=2),
        seed=7,
    )
    world.add_agent("a1", Position(x=1, y=1), Direction.E)

    intents = {"a1": Direction.E}
    results = world.resolve_moves(intents)
    assert results["a1"].outcome == MoveOutcome.BLOCK_WALL
    assert results["a1"].cause_cell == (2, 1)
