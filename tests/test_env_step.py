import pytest

from llmgrid.agent_map import AgentMap
from llmgrid.env import simulate as simulate_module
from llmgrid.env.grid import GridWorld, TRAFFIC_CONE_TTL
from llmgrid.env.simulate import EpisodeCheckpoint, run_episode, sync_maps
from llmgrid.schema import (
    CommunicateAction,
    Decision,
    Direction,
    MoveOutcome,
    MsgIntent,
    MsgRequest,
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

    def patched_resolve(
        use_llm: bool,
        model_id: str,
        seed: int,
        strategy: str,
        loop_guidance: str,
        history_limit: int,
        **kwargs,
    ):
        assert not use_llm
        assert strategy == "none"
        assert loop_guidance == "passive"
        assert history_limit == 5
        assert kwargs.get("radio_range", 0) == 0
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
            message=MsgRequest(
                sender_id="a1",
                seq=0,
                req="GUIDE",
                target=Position(x=2, y=2),
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


def test_build_observation_respects_comm_limits():
    world = GridWorld(
        width=3,
        height=3,
        obstacles=[],
        goal=Position(x=2, y=2),
        seed=5,
    )
    world.add_agent("a1", Position(x=1, y=1), Direction.N)

    obs = world.build_observation(
        "a1",
        turn_index=0,
        max_turns=10,
        visibility_radius=1,
        radio_range=0,
        max_outbound_per_turn=0,
    )

    assert obs.comm_limits.range == 0
    assert obs.comm_limits.max_outbound_per_turn == 0


def test_agent_conflict_sets_contended_mask_and_outcomes():
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

    # Artifacts/NO_GO purged; verify contended mask captures the hotspot instead of a cone

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


def test_finished_agents_do_not_block_goal():
    world = GridWorld(
        width=3,
        height=1,
        obstacles=[],
        goal=Position(x=2, y=0),
        seed=0,
    )
    world.add_agent("a1", Position(x=1, y=0), Direction.E)
    world.add_agent("a2", Position(x=0, y=0), Direction.E)

    # First agent reaches the goal and finishes.
    results = world.resolve_moves({"a1": Direction.E, "a2": None})
    assert results["a1"].outcome == MoveOutcome.FINISHED
    world.mark_finished("a1")
    assert world.is_finished("a1")

    # Second agent moves into the vacated cell.
    results = world.resolve_moves({"a1": None, "a2": Direction.E})
    assert results["a2"].outcome == MoveOutcome.OK
    assert world.occupancy["a2"] == (1, 0)

    # Second agent should now be able to enter the goal cell and finish.
    results = world.resolve_moves({"a1": None, "a2": Direction.E})
    assert results["a2"].outcome == MoveOutcome.FINISHED
    world.mark_finished("a2")
    assert world.is_finished("a2")


def test_agent_map_merge_from():
    m1 = AgentMap(3, 3)
    m2 = AgentMap(3, 3)

    m1.load_state(
        {
            "tiles": ["XXX", ".X.", "XXX"],
            "agents": {"a": {"x": 0, "y": 0, "turn": 1}},
        }
    )
    m2.load_state(
        {
            "tiles": ["XGX", "X#X", "XXX"],
            "agents": {"a": {"x": 1, "y": 1, "turn": 3}, "b": {"x": 2, "y": 2, "turn": 2}},
        }
    )

    m1.merge_from(m2)
    state = m1.export_state()

    assert state["tiles"][0][1] == "G"  # filled unknown
    assert state["tiles"][1][0] == "."  # preserved known tile
    assert state["agents"]["a"]["turn"] == 3  # newer turn kept
    assert state["agents"]["b"]["x"] == 2  # new agent added


def test_sync_maps_radio_range_merges_tiles():
    world = GridWorld(
        width=4,
        height=3,
        obstacles=[Position(x=3, y=1)],
        goal=Position(x=3, y=2),
        seed=0,
    )
    world.add_agent("a1", Position(x=0, y=1), Direction.E)
    world.add_agent("a2", Position(x=2, y=1), Direction.W)

    # Populate individual maps
    world.build_observation("a1", turn_index=0, max_turns=5, visibility_radius=1, radio_range=2)
    world.build_observation("a2", turn_index=0, max_turns=5, visibility_radius=1, radio_range=2)

    # Only a2 sees the wall at (3,1) before sync
    before = world.agent_maps["a1"].export_state()
    assert before["tiles"][1][3] == "X"

    sync_maps(world, ["a1", "a2"], "radio_sync", radio_range=2)

    after = world.agent_maps["a1"].export_state()
    assert after["tiles"][1][3] == "#"  # learned from a2


def test_sync_maps_global_shares_all():
    world = GridWorld(
        width=6,
        height=5,
        obstacles=[Position(x=1, y=1), Position(x=4, y=3)],
        goal=Position(x=5, y=4),
        seed=1,
    )
    world.add_agent("a1", Position(x=1, y=2), Direction.E)  # sees wall at (1,1)
    world.add_agent("a2", Position(x=4, y=2), Direction.W)  # sees wall at (4,3)
    world.add_agent("a3", Position(x=3, y=2), Direction.N)  # central observer

    for aid in ["a1", "a2", "a3"]:
        world.build_observation(aid, turn_index=0, max_turns=5, visibility_radius=1, radio_range=0)

    sync_maps(world, ["a1", "a2", "a3"], "global", radio_range=0)

    for aid in ["a1", "a2", "a3"]:
        tiles = world.agent_maps[aid].export_state()["tiles"]
        assert tiles[1][1] == "#"  # wall from a1
        assert tiles[3][4] == "#"  # wall from a2
