import random

from llmgrid.env.grid import GridWorld
from llmgrid.schema import Position, StrengthBucket


def make_world(
    width: int = 20,
    height: int = 15,
    *,
    walls: list[Position] | None = None,
    goal: Position | None = None,
    seed: int = 123,
    bias_seed: int | None = 42,
    bias_p: float = 0.18,
    bias_bonus: float = 0.12,
):
    walls = walls or []
    goal = goal or Position(x=width - 2, y=2)
    return GridWorld(
        width=width,
        height=height,
        obstacles=walls,
        goal=goal,
        seed=seed,
        bearing_flip_p=0.0,
        bearing_drop_p=0.0,
        bearing_bias_seed=bias_seed,
        bearing_bias_p=bias_p,
        bearing_bias_wall_bonus=bias_bonus,
    )


def biased_fraction(world: GridWorld, *, samples: int | None = None) -> float:
    coords = [
        (x, y)
        for y in range(world.size.height)
        for x in range(world.size.width)
    ]
    if samples is not None and samples < len(coords):
        coords = random.Random(0).sample(coords, samples)

    biased = 0
    for x, y in coords:
        steps = world._bias_steps(
            x,
            y,
            world.bearing_bias_seed or 0,
            world.bearing_bias_p,
            world.bearing_bias_wall_bonus,
        )
        if steps != 0:
            biased += 1
    return biased / len(coords)


def test_bias_deterministic_seed():
    world1 = make_world(bias_seed=77)
    world2 = make_world(bias_seed=77)
    vals1 = [
        [
            world1._bias_steps(x, y, 77, world1.bearing_bias_p, world1.bearing_bias_wall_bonus)
            for x in range(world1.size.width)
        ]
        for y in range(world1.size.height)
    ]
    vals2 = [
        [
            world2._bias_steps(x, y, 77, world2.bearing_bias_p, world2.bearing_bias_wall_bonus)
            for x in range(world2.size.width)
        ]
        for y in range(world2.size.height)
    ]
    assert vals1 == vals2


def test_bias_changes_with_seed():
    world = make_world()
    vals_seed_a = [
        [world._bias_steps(x, y, 11, world.bearing_bias_p, world.bearing_bias_wall_bonus) for x in range(world.size.width)]
        for y in range(world.size.height)
    ]
    vals_seed_b = [
        [world._bias_steps(x, y, 99, world.bearing_bias_p, world.bearing_bias_wall_bonus) for x in range(world.size.width)]
        for y in range(world.size.height)
    ]
    assert vals_seed_a != vals_seed_b


def test_bias_fraction_matches_expected_band():
    world = make_world(bias_bonus=0.0)
    frac = biased_fraction(world)
    assert 0.12 <= frac <= 0.28, f"Fraction {frac:.3f} outside expected band"


def test_bias_near_wall_amplification():
    wall_positions = [Position(x=10, y=y) for y in range(2, 13)]
    world = make_world(walls=wall_positions)
    wall_set = {(p.x, p.y) for p in wall_positions}
    adj_cells: list[tuple[int, int]] = []
    interior_cells: list[tuple[int, int]] = []
    for y in range(world.size.height):
        for x in range(world.size.width):
            if (x, y) in wall_set:
                continue
            if world._neighbor_has_wall(x, y):
                adj_cells.append((x, y))
            else:
                interior_cells.append((x, y))

    def fraction(cells: list[tuple[int, int]]) -> float:
        hits = 0
        for x, y in cells:
            if world._bias_steps(x, y, world.bearing_bias_seed or 0, world.bearing_bias_p, world.bearing_bias_wall_bonus):
                hits += 1
        return hits / len(cells) if cells else 0.0

    adj_frac = fraction(adj_cells)
    interior_frac = fraction(interior_cells)
    assert adj_frac > interior_frac + 0.05


def test_sensor_always_available_and_strength_monotonic():
    world = make_world()
    samples = [
        Position(x=1, y=1),
        Position(x=5, y=5),
        Position(x=10, y=8),
    ]
    for pos in samples:
        reading = world._bearing_sensor(pos.x, pos.y)
        assert reading.available is True

    goal = world.goal
    near = Position(x=max(0, goal.x - 1), y=goal.y)
    mid = Position(x=max(0, goal.x - 5), y=min(world.size.height - 1, goal.y + 3))
    far = Position(x=1, y=world.size.height - 2)

    near_strength = world._bearing_sensor(near.x, near.y).strength
    mid_strength = world._bearing_sensor(mid.x, mid.y).strength
    far_strength = world._bearing_sensor(far.x, far.y).strength

    assert near_strength == StrengthBucket.NEAR
    assert far_strength == StrengthBucket.FAR
    assert mid_strength in {StrengthBucket.MID, StrengthBucket.FAR}
