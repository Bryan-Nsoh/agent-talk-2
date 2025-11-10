"""Prompt templates assembled from capability-specific blocks."""

from __future__ import annotations

from typing import List

CORE_HEADER_TEMPLATE = """OBJECTIVE:
Keep the entire team safe and moving. Agents should be cooperative and do their best to help each other. "Progress" means avoiding stalemates, exploring new corridors, and guiding everyone to the goal—sometimes by stepping away from it temporarily. The episode ends after 60 turns; a timeout is a failure even if some agents reach the goal.

MISSION BRIEF:
- Grid awareness comes from the JSON: `grid_size`, `local_patch`, and `adjacent` describe nearby tiles; `self_state` gives your orientation.
- `world_map_ascii` is your stitched map (X=unknown, `~` marks your recent trail). `adjacent_frontiers` lists unknown cells you can reveal immediately; `nearest_frontier` points to the closest X tile overall.
- Actions: {actions_sentence}
- Collisions (BLOCK_AGENT or SWAP_CONFLICT) reset you and waste a turn. `contended_neighbors` tells you which adjacent directions collided last turn—treat them as hotspots and coordinate before retrying.
- History: `history` captures your recent intents/outcomes and notes; `recent_positions` lists the cells you just visited.
- Goal sensor (`goal_sensor`) is a noisy hint. Treat it as guidance, not a command.
- Teammates do not see your thoughts—announce reroutes, hazards, or intents when relevant.
- {message_behavior_line}

TOOL ARSENAL (with quick cues):
- MOVE_N/E/S/W — default travel. Example: `adjacent.E = FREE`, frontier east → MOVE_E, comment `Scouting unknown at (x+1,y)`.
- STAY — hold position when moving would collide or you need to coordinate first. Example: all sides blocked, teammate approaching → STAY + explain the pause.
{comm_tool_line}
- HISTORY — glance at prior intents/outcomes to avoid repeating the same blockage; `recent_positions` exposes short back-and-forth patterns so you can change tactics.
{oracle_tool_line}

DECISION HIERARCHY (apply in order every turn):
1. PREVENT COLLISIONS: Respect WALL / contended cells. Yield or coordinate before entering tight corridors.
2. EXPLORE UNKNOWN: If `adjacent_frontiers` is non-empty, MOVE into one of those cells to reveal it. Otherwise head toward `nearest_frontier`; pick a direction that leaves the `~` trail (prefer tiles not visited in the last few turns) before reusing older paths.
3. ADVANCE TOWARD GOAL: After nearby frontiers are mapped and hazards cleared, follow the goal bearing or Manhattan gradient.

REASONING NOTE (comment field, ≤25 words):
- Use the comment purely as reasoning: explain why you chose this action.
- Reference absolute coords or map features (e.g., “scouting X at (3,2)”). If you pick from `adjacent_frontiers` or `nearest_frontier`, cite the coordinate explicitly.
- If you enter a tile marked `~` (recent trail), explain why revisiting it is necessary.
- If `contended_neighbors` flags a direction, mention how you’ll avoid or coordinate around it.
{comment_extra_line}

OUTPUT CONTRACT:
Return ONE JSON object with this exact shape (no code fences, no prose):
{{"action":{{"kind":"{action_contract}","direction":"N|E|S|W" (only for MOVE),"payload":null}},"comment":"OK; <=25 words"}}.
Example: {{"action":{{"kind":"MOVE","direction":"N","payload":null}},"comment":"OK; advancing north to scout"}}.
Do NOT emit explanations outside that JSON blob.

EXECUTION RULES:
1. Read <OBSERVATION_JSON>.
2. Respect walls, bounds, and agent collisions visible in the patch.
3. Consult `adjacent` (NESW labels) plus `recent_positions` to avoid immediate backtracking unless it is the only safe option.
4. Use `history`, `goal_sensor`, `neighbors_in_view`, and `inbox` to inform your choice.
5. {communication_execution_line}
6. Populate `comment` with one concise paragraph (1–3 sentences) explaining your reasoning for this turn.
7. {execution_rule_line}

QUALITY GATE:
Return exactly one `Decision` object that validates.

FINAL INSTRUCTIONS:
Emit only the structured `Decision` object.

=== INPUTS ===
<OBSERVATION_JSON>
"""


def _optional_line(text: str) -> str:
    return text + "\n" if text else ""


def _actions_sentence(action_kinds: List[str], radio_range: int) -> str:
    parts = ["MOVE_N/E/S/W, STAY"]
    if "COMMUNICATE" in action_kinds:
        parts.append(f"COMMUNICATE (one structured radio message per turn, range {radio_range}).")
    if "ASK_ORACLE" in action_kinds:
        parts.append("ASK_ORACLE (request global guidance; peer radio is disabled while oracle is available).")
    return " ".join(parts)


def _message_behavior_line(action_kinds: List[str]) -> str:
    if "COMMUNICATE" in action_kinds:
        return "Messages you send this turn arrive in teammate inboxes at the start of the next turn; broadcast before stepping back into a contested direction."
    return "Peer radio is disabled for this baseline; plan routes assuming you will not receive teammate messages."


def build_prompt_header(
    *,
    radio_range: int,
    action_kinds: List[str],
) -> str:
    allow_comm = "COMMUNICATE" in action_kinds
    allow_oracle = False

    actions_sentence = _actions_sentence(action_kinds, radio_range)
    message_behavior_line = _message_behavior_line(action_kinds)

    comm_tool_line = ""
    if allow_comm:
        comm_tool_line = (
            "- COMMUNICATE — one structured message to share intent, hazards, or reroutes (range "
            f"{radio_range}). Share new information, not repeats.\n"
        )

    oracle_tool_line = ""

    if not allow_comm:
        comm_comment_line = "- Radio is disabled; rely on MOVE/STAY to coordinate implicitly."
        communication_execution_line = "Radio is disabled this run; skip peer communication actions."
    else:
        comm_comment_line = "- Use COMMUNICATE to broadcast reroutes, hazards, or intent when helpful."
        communication_execution_line = "If you COMMUNICATE, send the minimal helpful message."

    comment_extra_line = comm_comment_line + "\n"

    execution_rule_line = "Stay focused on MOVE/STAY actions per the rules above."
    # remove trailing newline if no extra line appended
    comment_extra_line = comment_extra_line.rstrip("\n") + "\n"

    action_contract = "|".join(action_kinds)

    return CORE_HEADER_TEMPLATE.format(
        actions_sentence=actions_sentence,
        message_behavior_line=message_behavior_line,
        comm_tool_line=comm_tool_line,
        oracle_tool_line=oracle_tool_line,
        comment_extra_line=comment_extra_line,
        action_contract=action_contract,
        communication_execution_line=communication_execution_line,
        execution_rule_line=execution_rule_line,
    )
