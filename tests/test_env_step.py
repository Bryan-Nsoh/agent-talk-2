from llmgrid.env.grid import GridWorld
from llmgrid.schema import Direction, MoveOutcome, Position


def test_gridworld_resolves_simple_move():
    world = GridWorld(
        width=5,
        height=5,
        obstacles=[],
        goal=Position(x=4, y=4),
        seed=0,
    )
    world.add_agent("a1", Position(x=1, y=1))
    world.add_agent("a2", Position(x=2, y=1))

    intents = {"a1": Direction.E, "a2": Direction.W}
    results = world.resolve_moves(intents)
    assert results["a1"].outcome == MoveOutcome.SWAP_CONFLICT
    assert results["a2"].outcome == MoveOutcome.SWAP_CONFLICT


def test_build_observation_has_grid_and_legend():
    world = GridWorld(
        width=3,
        height=3,
        obstacles=[],
        goal=Position(x=2, y=0),
        seed=1,
    )
    world.add_agent("a1", Position(x=1, y=1))
    obs = world.build_observation(
        "a1",
        turn_index=0,
        max_turns=10,
        visibility_radius=2,
        map_sharing="none",
    )
    assert obs.grid.width == 3
    assert obs.grid.height == 3
    assert obs.legend["#"] == "WALL (impassable)"
    assert obs.self.agent_id == "a1"
