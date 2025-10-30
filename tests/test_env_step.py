from llmgrid.env.grid import GridWorld
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

    # Collision rule keeps both agents in place
    intents = {"a1": Direction.E, "a2": Direction.W}
    world.resolve_moves(intents)
