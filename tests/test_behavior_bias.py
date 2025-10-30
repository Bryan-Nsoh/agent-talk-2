from collections import deque

from llmgrid.env.grid import GridWorld
from llmgrid.schema import Octant, Position


def make_corridor_world(*, bias_seed: int | None) -> GridWorld:
    walls = [Position(x=7, y=y) for y in range(1, 14) if y not in {4, 5}]
    walls += [Position(x=x, y=9) for x in range(3, 12)]
    return GridWorld(
        width=16,
        height=15,
        obstacles=walls,
        goal=Position(x=14, y=2),
        seed=3,
        bearing_flip_p=0.0,
        bearing_drop_p=0.0,
        bearing_bias_seed=bias_seed,
        bearing_bias_p=0.2,
        bearing_bias_wall_bonus=0.12,
    )


def shortest_path(world: GridWorld, start: Position) -> list[tuple[int, int]]:
    goal = (world.goal.x, world.goal.y)
    queue = deque([(start.x, start.y)])
    parents: dict[tuple[int, int], tuple[int, int] | None] = {(start.x, start.y): None}
    while queue:
        x, y = queue.popleft()
        if (x, y) == goal:
            break
        for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
            nx, ny = x + dx, y + dy
            if not world._in_bounds(nx, ny) or not world._passable(nx, ny):
                continue
            if (nx, ny) in parents:
                continue
            parents[(nx, ny)] = (x, y)
            queue.append((nx, ny))

    path = []
    node = goal
    if node not in parents:
        return path
    while node is not None:
        path.append(node)
        node = parents[node]
    path.reverse()
    return path


def octant_index(bearing: Octant | None) -> int:
    order = [
        Octant.N,
        Octant.NE,
        Octant.E,
        Octant.SE,
        Octant.S,
        Octant.SW,
        Octant.W,
        Octant.NW,
    ]
    if bearing is None:
        return -1
    return order.index(bearing)


def test_bias_rotates_bearings_along_shortest_path():
    start = Position(x=1, y=13)
    base_world = make_corridor_world(bias_seed=None)
    drift_world = make_corridor_world(bias_seed=42)

    path = shortest_path(base_world, start)
    assert path, "Shortest path should exist"

    differences = 0
    for (x, y) in path:
        base_bearing = base_world._bearing_sensor(x, y).bearing
        drift_bearing = drift_world._bearing_sensor(x, y).bearing
        if base_bearing == drift_bearing:
            continue
        base_idx = octant_index(base_bearing)
        drift_idx = octant_index(drift_bearing)
        assert base_idx != -1 and drift_idx != -1
        delta = (drift_idx - base_idx) % 8
        assert delta in {1, 7}, "Bias must rotate bearing by ±45°"
        differences += 1

    assert differences >= 4, "Gold Drift should perturb several cells along the optimal path"
