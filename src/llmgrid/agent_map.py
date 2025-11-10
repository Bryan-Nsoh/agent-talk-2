"""Persistent egocentric ASCII map that agents update as they explore."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple, Any

from llmgrid.schema import Position

Coordinate = Tuple[int, int]


@dataclass
class SeenAgent:
    """Snapshot of where a peer was last observed."""

    pos: Coordinate
    turn_index: int


class AgentMap:
    """Sparse world map that remembers revealed tiles and last-seen agents."""

    UNKNOWN = "X"

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self._tiles: List[List[str]] = [
            [self.UNKNOWN for _ in range(width)] for _ in range(height)
        ]
        self._agents_last_seen: Dict[str, SeenAgent] = {}
        self._recent_positions: List[Coordinate] = []

    def reset(self) -> None:
        """Clear all knowledge (used rarely in tests)."""

        for row in self._tiles:
            for x in range(len(row)):
                row[x] = self.UNKNOWN
        self._agents_last_seen.clear()

    def stamp_patch(
        self,
        *,
        top_left: Position,
        rows: List[str],
        occupancy_lookup: Dict[str, Coordinate],
        turn_index: int,
    ) -> None:
        """Integrate the visible local patch into the persistent map."""

        pos_to_agent = {pos: aid for aid, pos in occupancy_lookup.items()}

        for dy, line in enumerate(rows):
            y = top_left.y + dy
            if not (0 <= y < self.height):
                continue
            for dx, char in enumerate(line):
                x = top_left.x + dx
                if not (0 <= x < self.width):
                    continue

                base_char = self._translate_tile_char(char)
                self._tiles[y][x] = base_char

                agent_id = pos_to_agent.get((x, y))
                if agent_id:
                    self._agents_last_seen[agent_id] = SeenAgent(
                        pos=(x, y),
                        turn_index=turn_index,
                    )

    def mark_self(
        self,
        agent_id: str,
        pos: Coordinate,
        turn_index: int,
        recent_positions: List[Coordinate],
    ) -> None:
        """Record the agent's current absolute position and recent trail."""

        self._agents_last_seen[agent_id] = SeenAgent(pos=pos, turn_index=turn_index)
        x, y = pos
        if 0 <= x < self.width and 0 <= y < self.height:
            if self._tiles[y][x] == self.UNKNOWN:
                self._tiles[y][x] = "."
        self._recent_positions = recent_positions[:3]

    def render_ascii(
        self,
        *,
        icon_lookup: Dict[str, str],
        orientations: Dict[str, str],
    ) -> str:
        """Return a labeled ASCII representation of the known world (north-up)."""

        tens_chars = [" "] * self.width
        for x in range(0, self.width, 10):
            tens_chars[x] = str(x // 10)
        tens_line = "".join(tens_chars)
        ones_line = "".join(str(x % 10) for x in range(self.width))

        lines: List[str] = []
        lines.append(f"x(tens): {tens_line}")
        lines.append(f"x(units): {ones_line}")

        overlay = {(seen.pos): icon_lookup.get(aid, "?") for aid, seen in self._agents_last_seen.items()}

        for y in range(self.height - 1, -1, -1):
            row = list(self._tiles[y])
            for x in range(self.width):
                icon = overlay.get((x, y))
                if icon:
                    row[x] = icon
                elif (x, y) in self._recent_positions:
                    row[x] = "~"
            lines.append(f"y={y:02d} | {''.join(row)} |")

        lines.append(f"x(units): {ones_line}")
        lines.append("")
        lines.append("Legend")
        lines.append(" - Compass: N↑ E→ S↓ W←")
        agent_entries: List[str] = []
        for aid, seen in sorted(self._agents_last_seen.items()):
            arrow = {"N": "^", "E": ">", "S": "v", "W": "<"}.get(orientations.get(aid, ""), "?")
            agent_entries.append(f"{aid}:{arrow}")
        if agent_entries:
            lines.append(" - Agents: " + "  ".join(agent_entries))
        return "\n".join(lines)

    @staticmethod
    def _translate_tile_char(char: str) -> str:
        """Map local patch glyphs to persistent map glyphs."""

        if char == "#":
            return "#"
        if char == "G":
            return "G"
        # treat agents and empty cells as traversable space
        return "."

    def export_state(self) -> Dict[str, Any]:
        """Return a JSON-serialisable snapshot."""

        return {
            "tiles": ["".join(row) for row in self._tiles],
            "agents": {
                agent_id: {
                    "x": seen.pos[0],
                    "y": seen.pos[1],
                    "turn": seen.turn_index,
                }
                for agent_id, seen in self._agents_last_seen.items()
            },
        }

    def load_state(self, payload: Dict[str, Any]) -> None:
        """Restore from a snapshot produced by export_state."""

        tiles = payload.get("tiles", [])
        for y, line in enumerate(tiles):
            if y >= self.height:
                break
            for x, char in enumerate(line):
                if x >= self.width:
                    break
                self._tiles[y][x] = char
        agents_payload = payload.get("agents", {})
        self._agents_last_seen = {
            agent_id: SeenAgent(
                pos=(data["x"], data["y"]),
                turn_index=data.get("turn", 0),
            )
            for agent_id, data in agents_payload.items()
        }
        self._recent_positions = []

    def find_frontiers(self) -> List[Coordinate]:
        """Unknown cells that border a discovered free/goal tile."""

        frontiers: List[Coordinate] = []
        for y in range(self.height):
            for x in range(self.width):
                if self._tiles[y][x] != self.UNKNOWN:
                    continue
                for nx, ny in _neighbors4(x, y):
                    if 0 <= nx < self.width and 0 <= ny < self.height:
                        if self._tiles[ny][nx] in {".", "G"}:
                            frontiers.append((x, y))
                            break
        return frontiers

    def extract_window(self, center: Coordinate, radius: int) -> Tuple[Position, List[str]]:
        """Return the top-left (min x, max y) coordinate and rows for a window."""

        cx, cy = center
        left = max(0, cx - radius)
        right = min(self.width - 1, cx + radius)
        top = min(self.height - 1, cy + radius)
        bottom = max(0, cy - radius)

        rows: List[str] = []
        for y in range(top, bottom - 1, -1):
            row_chars: List[str] = []
            for x in range(left, right + 1):
                row_chars.append(self._tiles[y][x])
            rows.append("".join(row_chars))

        return Position(x=left, y=top), rows

    def apply_patch(self, top_left: Position, rows: List[str]) -> None:
        """Merge a rectangular patch (row 0 == highest y) into the map."""

        for dy, line in enumerate(rows):
            y = top_left.y - dy
            if not (0 <= y < self.height):
                continue
            for dx, char in enumerate(line):
                x = top_left.x + dx
                if not (0 <= x < self.width):
                    continue
                if char == self.UNKNOWN:
                    continue
                self._tiles[y][x] = char

    def get_tile(self, x: int, y: int) -> str:
        if 0 <= x < self.width and 0 <= y < self.height:
            return self._tiles[y][x]
        return "#"


def _neighbors4(x: int, y: int) -> Iterable[Coordinate]:
    yield x + 1, y
    yield x - 1, y
    yield x, y + 1
    yield x, y - 1
