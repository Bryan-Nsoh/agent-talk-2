"""Prompt templates that follow the cache-friendly header rules."""

STATIC_HEADER = """OBJECTIVE:
Choose exactly one action that maximises team progress toward the goal under partial observability and range-limited communication.

OUTPUT CONTRACT:
Return a single structured object that conforms to the `Decision` model. Do not return any other text.

EXECUTION RULES:
1. Read <OBSERVATION_JSON>.
2. Respect walls, bounds, and agent collisions visible in the patch.
3. Consult the `adjacent` list; it labels each of the N/E/S/W cells as FREE, WALL, OUT_OF_BOUNDS, AGENT, or GOAL.
4. Use `recent_positions` (last 5 positions, newest first) to avoid immediate backtracking unless it is the only safe option.
5. Use `goal_sensor`, `neighbors_in_view`, `artifacts_in_view`, and `inbox` to decide.
6. If you communicate, send the minimal valid message.
7. If no safe move exists, prefer STAY over collisions.
8. Populate `comment` with exactly one paragraph (1-3 sentences) that explains your reasoning for this turn.

DECISION RULES:
- Precedence: 1) OUTPUT CONTRACT, 2) EXECUTION RULES.
- Ambiguity: choose the safest valid action. Do not invent ids or positions.

QUALITY GATE:
Return exactly one `Decision` object that validates.

FINAL INSTRUCTIONS:
Emit only the structured `Decision` object.

=== INPUTS ===
<OBSERVATION_JSON>
"""
