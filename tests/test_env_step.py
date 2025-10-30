import pytest

from llmgrid.env import simulate as simulate_module
from llmgrid.env.grid import GridWorld
from llmgrid.env.simulate import EpisodeCheckpoint, run_episode
from llmgrid.schema import Direction, Position


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
    world.resolve_moves(intents)
    assert world.occupancy["a1"] == (1, 1)
    assert world.occupancy["a2"] == (3, 1)


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

    def patched_resolve(use_llm: bool, model_id: str, seed: int):
        assert not use_llm
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

    assert resume_metrics.turns == expected_metrics.turns
    assert resume_metrics.success == expected_metrics.success
    assert resume_metrics.messages_sent == expected_metrics.messages_sent
    assert resume_metrics.marks_placed == expected_metrics.marks_placed
    assert resume_metrics.collisions == expected_metrics.collisions
