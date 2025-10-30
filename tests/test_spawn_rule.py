from llmgrid.cli import poc_two_agents
from llmgrid.schema import Position


def test_default_start_positions_half_distance_rule():
    width, height = 20, 12
    goal = poc_two_agents._default_goal(width, height)
    positions = poc_two_agents._default_start_positions(width, height, goal, agent_count=5)
    assert len(positions) == 5
    min_distance = (width + height + 1) // 2
    for pos in positions.values():
        dist = abs(goal.x - pos.x) + abs(goal.y - pos.y)
        assert dist >= min_distance
        assert (pos.x, pos.y) != (goal.x, goal.y)
