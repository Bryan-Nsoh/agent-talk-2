"""Agent-specific persistent map with single-grid rendering."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

Coordinate = Tuple[int, int]


@dataclass
class BaseTile:
    """Base tile knowledge: '#', 'G', '.', or 'X' (unknown)."""

    ch: str = "X"


class AgentMap:
    UNKNOWN = "X"

    def __init__(self, width: int, height: int, *, recent_len: int = 3) -> None:
        self.width = width
        self.height = height
        self._base: List[List[str]] = [
            [self.UNKNOWN for _ in range(width)] for _ in range(height)
        ]
        self.visited: Set[Coordinate] = set()
        self.recent = deque(maxlen=recent_len)
        self.last_collision: Optional[Coordinate] = None

    # ------------------------------------------------------------------ #
    # Updates
    # ------------------------------------------------------------------ #
    def update_visible(self, cells: Iterable[Tuple[int, int, str]]) -> None:
        """Integrate currently visible base tiles (no overlays)."""
        for x, y, ch in cells:
            if 0 <= x < self.width and 0 <= y < self.height:
                if ch in {"#", "G"}:
                    self._base[y][x] = ch
                else:
                    # Any non-wall/goal seen cell is free
                    self._base[y][x] = "."

    def mark_visit(self, pos: Coordinate) -> None:
        if 0 <= pos[0] < self.width and 0 <= pos[1] < self.height:
            self.visited.add(pos)

    def set_recent(self, positions: List[Coordinate]) -> None:
        self.recent.clear()
        for pos in positions:
            self.recent.append(pos)
            self.mark_visit(pos)

    def set_last_collision(self, cell: Optional[Coordinate]) -> None:
        self.last_collision = cell

    def merge_base_from(self, other: "AgentMap") -> None:
        """Merge base tiles only; overlays (visited/trails/collisions) stay local."""
        if self.width != other.width or self.height != other.height:
            raise ValueError("Map dimensions must match to merge.")
        for y in range(self.height):
            for x in range(self.width):
                if self._base[y][x] == self.UNKNOWN and other._base[y][x] != self.UNKNOWN:
                    self._base[y][x] = other._base[y][x]

    # ------------------------------------------------------------------ #
    # Queries / rendering
    # ------------------------------------------------------------------ #
    def base_tile(self, x: int, y: int) -> str:
        return self._base[y][x]

    def find_frontiers(self) -> List[Coordinate]:
        """Unknown cells with at least one known free/goal neighbor."""
        frontiers: List[Coordinate] = []
        for y in range(self.height):
            for x in range(self.width):
                if self._base[y][x] != self.UNKNOWN:
                    continue
                for nx, ny in _neighbors4(x, y):
                    if 0 <= nx < self.width and 0 <= ny < self.height:
                        if self._base[ny][nx] in {".", "G"}:
                            frontiers.append((x, y))
                            break
        return frontiers

    def render_grid(
        self,
        *,
        self_pos: Coordinate,
        neighbors: Dict[Coordinate, str],
        goal_pos: Optional[Coordinate],
    ) -> List[List[str]]:
        rows: List[List[str]] = []
        recent_set = set(self.recent)
        visited_set = set(self.visited)
        lc = self.last_collision
        for y in range(self.height):
            row: List[str] = []
            for x in range(self.width):
                base = self._base[y][x]
                ch = base
                if base == self.UNKNOWN:
                    ch = "X"
                elif base == "#":
                    ch = "#"
                elif (x, y) == self_pos:
                    ch = "@"
                elif (x, y) in neighbors:
                    ch = neighbors[(x, y)]
                elif lc is not None and (x, y) == lc and base != "#":
                    ch = "!"
                elif (x, y) in recent_set:
                    ch = "*"
                elif (x, y) in visited_set:
                    ch = "~"
                else:
                    ch = "." if base != "G" else "G"
                row.append(ch)
            rows.append(row)
        return rows


def _neighbors4(x: int, y: int):
    yield x + 1, y
    yield x - 1, y
    yield x, y + 1
    yield x, y - 1
