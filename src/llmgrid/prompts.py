"""Prompt template for the map-sharing no-comm baseline."""

from __future__ import annotations

CORE_HEADER = """OBJECTIVE:
Reach the goal yourself and help your teammates reach it too. Agents should be cooperative and avoid blocking each other. Progress means reducing everyone’s distance to the goal, exploring new corridors, and guiding the whole team toward completion.

GRID REPRESENTATION:
You receive a single grid as JSON:
  "grid": {
    "width": int,
    "height": int,
    "rows": [
      ["#",".",".","X",...],  // y = 0 (TOP row)
      ["#","@","~",".",...],  // y = 1
      ...
      ["#","#","#","#",...]   // y = height-1 (BOTTOM row)
    ]
  }

INDEXING:
- grid.rows[y][x] is the symbol at (x,y).
- y: 0 at TOP, increases DOWNWARD. x: 0 at LEFT, increases RIGHTWARD.
- Your position self.pos = {"x": X, "y": Y} corresponds to grid.rows[Y][X].

SYMBOLS (see legend in the observation):
  # = WALL
  G = GOAL
  X = UNKNOWN
  @ = YOU (current position)
  1,2,3... = OTHER AGENTS (visible now)
  ! = last collision target
  * = one of your last 3 positions
  ~ = visited (not in last 3)
  . = free, never visited

SYMBOL PRIORITY:
Walls/goal/unknown are base. On free cells the highest applicable overlay shows:
  @ > digits > ! > * > ~ > .

MAP SHARING:
map_sharing = none | radio_sync | global.
- radio_sync: when within radio_range, base tiles (wall/free/goal/unknown) are merged; overlays stay personal.
- global: all base tiles are shared globally; overlays stay personal.
- Treat any non-'X' tile as reliable even if you didn’t see it yourself.

VISIBILITY:
- Visibility radius R_vis = 2 (Manhattan). Only currently visible agents appear as digits.

LAST RESULT:
last_result = {kind, cell?, opponents[]}
- kind: OK | FINISHED | BLOCK_WALL | BLOCK_AGENT | SWAP_CONFLICT
- cell: target you tried (goal cell for FINISHED, null for OK)
- opponents: agents collided with (empty for wall/goal/OK)
- The last collision cell is also shown as '!' for one turn if on-grid.

DECISION HIERARCHY:
1) Avoid walls/out of bounds; do not step into '#'.
2) Respect last collision: avoid repeating the '!' cell if possible.
3) Break immediate backtracks: avoid bouncing between recent '*' cells; STAY if needed.
4) Explore unknown first: prefer adjacent_frontiers (unknown neighbors).
5) Use goal when known: move toward goal_pos using free cells; avoid obvious collisions.
6) When all options look bad: STAY or pick the least-recent '~' to unwind loops.
7) If adjacent to goal, move into it unless it would clearly collide.

OUTPUT CONTRACT:
Return ONE JSON object (no prose):
{"action":{"kind":"MOVE"|"STAY","direction":"N"|"E"|"S"|"W" (for MOVE)},"comment":"<=25 words"}

=== INPUTS ===
<OBSERVATION_JSON>
"""


def build_prompt_header() -> str:
    return CORE_HEADER
