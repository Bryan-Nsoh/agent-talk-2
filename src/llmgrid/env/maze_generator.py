from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

from llmgrid.schema import Position


Direction = Tuple[int, int]

CARDINALS: Sequence[Direction] = [(0, -1), (1, 0), (0, 1), (-1, 0)]


@dataclass
class MazeConfig:
    width: int
    height: int
    seed: int
    extra_connection_prob: float = 0.0

    def __post_init__(self) -> None:
        if self.width < 3 or self.height < 3:
            raise ValueError("Maze dimensions must be at least 3x3.")
        if not (0.0 <= self.extra_connection_prob <= 1.0):
            raise ValueError("extra_connection_prob must be between 0 and 1.")


class MazeGenerator:
    """Generate carved mazes with guaranteed connectivity between supply points."""

    def __init__(self, config: MazeConfig) -> None:
        self.config = config
        self.rng = random.Random(config.seed)

    def generate(
        self,
        required_open_cells: Iterable[Tuple[int, int]],
    ) -> List[Position]:
        """
        Generate a maze and return obstacle positions.

        The generator ensures that all `required_open_cells` are free and
        mutually reachable.
        """
        width, height = self.config.width, self.config.height
        grid = [[1 for _ in range(width)] for _ in range(height)]

        start_x, start_y = self._initial_cell()
        grid[start_y][start_x] = 0
        stack: List[Tuple[int, int]] = [(start_x, start_y)]

        while stack:
            cx, cy = stack[-1]
            neighbors = self._uncarved_neighbors(cx, cy, grid)
            if not neighbors:
                stack.pop()
                continue
            nx, ny, wx, wy = self.rng.choice(neighbors)
            grid[wy][wx] = 0  # carve hallway between cells
            grid[ny][nx] = 0  # carve the neighbor cell
            stack.append((nx, ny))

        if self.config.extra_connection_prob > 0:
            self._add_extra_connections(grid)

        for cell in required_open_cells:
            x, y = cell
            if 0 <= x < width and 0 <= y < height:
                grid[y][x] = 0

        self._ensure_connection(grid, required_open_cells)

        obstacles = [
            Position(x=x, y=y)
            for y in range(height)
                for x in range(width)
                    if grid[y][x] == 1
        ]
        return obstacles

    def render_ascii(self, obstacles: Sequence[Position]) -> str:
        width, height = self.config.width, self.config.height
        obstacle_set = {(p.x, p.y) for p in obstacles}
        rows: List[str] = []
        for y in range(height):
            line = []
            for x in range(width):
                line.append("#" if (x, y) in obstacle_set else ".")
            rows.append("".join(line))
        return "\n".join(rows)

    def _initial_cell(self) -> Tuple[int, int]:
        # use odd coordinates to maximise uniformity
        x = 1 if self.config.width > 1 else 0
        y = 1 if self.config.height > 1 else 0
        return x, y

    def _uncarved_neighbors(
        self,
        x: int,
        y: int,
        grid: List[List[int]],
    ) -> List[Tuple[int, int, int, int]]:
        neighbors: List[Tuple[int, int, int, int]] = []
        width, height = self.config.width, self.config.height
        for dx, dy in CARDINALS:
            nx = x + dx * 2
            ny = y + dy * 2
            if (
                0 <= nx < width
                and 0 <= ny < height
                and grid[ny][nx] == 1
            ):
                wx = x + dx
                wy = y + dy
                neighbors.append((nx, ny, wx, wy))
        return neighbors

    def _add_extra_connections(self, grid: List[List[int]]) -> None:
        width, height = self.config.width, self.config.height
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                if grid[y][x] == 1 and self.rng.random() < self.config.extra_connection_prob:
                    # Only carve if it connects two distinct corridors
                    passages = sum(
                        grid[y + dy][x + dx] == 0
                        for dx, dy in CARDINALS
                    )
                    if passages >= 2:
                        grid[y][x] = 0

    def _ensure_connection(
        self,
        grid: List[List[int]],
        required_open_cells: Iterable[Tuple[int, int]],
    ) -> None:
        cells = list(required_open_cells)
        if not cells:
            return
        base = cells[0]
        reachable = self._flood_fill(grid, base)
        for cell in cells[1:]:
            if cell not in reachable:
                self._carve_path(grid, base, cell)
                reachable = self._flood_fill(grid, base)

    def _flood_fill(
        self,
        grid: List[List[int]],
        start: Tuple[int, int],
    ) -> set[Tuple[int, int]]:
        width, height = self.config.width, self.config.height
        sx, sy = start
        if not (0 <= sx < width and 0 <= sy < height):
            return set()
        if grid[sy][sx] == 1:
            return set()
        visited: set[Tuple[int, int]] = {start}
        queue: List[Tuple[int, int]] = [start]
        while queue:
            x, y = queue.pop()
            for dx, dy in CARDINALS:
                nx, ny = x + dx, y + dy
                if (
                    0 <= nx < width
                    and 0 <= ny < height
                    and grid[ny][nx] == 0
                    and (nx, ny) not in visited
                ):
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        return visited

    def _carve_path(
        self,
        grid: List[List[int]],
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> None:
        width, height = self.config.width, self.config.height
        sx, sy = start
        gx, gy = goal
        sx = min(max(sx, 0), width - 1)
        sy = min(max(sy, 0), height - 1)
        gx = min(max(gx, 0), width - 1)
        gy = min(max(gy, 0), height - 1)

        grid[sy][sx] = 0
        grid[gy][gx] = 0

        cx, cy = sx, sy
        while (cx, cy) != (gx, gy):
            if cx < gx:
                cx += 1
            elif cx > gx:
                cx -= 1
            elif cy < gy:
                cy += 1
            else:
                cy -= 1
            grid[cy][cx] = 0

