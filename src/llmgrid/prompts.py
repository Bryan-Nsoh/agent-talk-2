"""Prompt templates assembled from capability-specific blocks."""

from __future__ import annotations

from typing import List

CORE_HEADER_TEMPLATE = """OBJECTIVE:
Keep the entire team safe and moving. “Progress” means avoiding stalemates, exploring new corridors, and guiding everyone to the goal—sometimes by stepping away from it temporarily. The episode ends after 60 turns; a timeout is a failure even if some agents reach the goal.

MISSION BRIEF:
- Grid awareness comes from the JSON: `grid_size`, `local_patch`, and `adjacent` describe nearby tiles; `self_state` gives your orientation.
- Actions: {actions_sentence}
- Collisions (BLOCK_AGENT or SWAP_CONFLICT) reset you, waste a turn, and often leave NO_GO markers. `contended_neighbors` tells you which adjacent directions collided last turn—treat them as hotspots and coordinate before retrying.
- History: `history` holds your last turns with a `loop` counter and notes; `recent_positions` lists the cells you just visited.
- Goal sensor (`goal_sensor`) is a noisy hint. Treat it as guidance, not a command.
- Teammates do not see your thoughts—announce reroutes, hazards, or intents when relevant.
- {message_behavior_line}

TOOL ARSENAL (with quick cues):
- MOVE_N/E/S/W — default travel. Example: `adjacent.E = FREE`, loop=0 → MOVE_E, comment `OK; advancing east toward open corridor`.
- STAY — hold position when moving would collide or you need to coordinate first. Example: all sides blocked, teammate approaching → STAY + explain the pause.
{comm_tool_line}- MARK / NO_GO — drop on a hotspot after repeated conflicts; teammates should treat it as high risk for a few turns.
- HISTORY / LOOP COUNTER — diagnostic tool: if `history.loop` climbs or `recent_positions` oscillate, immediately select a different axis, even if it increases distance.
{oracle_tool_line}

DECISION HIERARCHY (apply in order every turn):
1. ESCAPE LOOPS: If `history.loop ≥ 2` or you see back-and-forth patterns in `history` / `recent_positions`, you MUST break the cycle. Choose a perpendicular or backward move, STAY + coordinate using available tools, or drop a MARK/NO_GO—even if that increases your goal distance.
2. PREVENT COLLISIONS: Respect WALL / NO_GO / contended cells. Yield or coordinate before entering tight corridors.
3. EXPLORE: Prefer safe tiles you haven’t occupied recently to open new paths and relieve congestion.
4. ADVANCE TOWARD GOAL: Only after you are loop-free and clear of hazards should you follow the goal bearing or Manhattan gradient.

LOOP ESCAPE EXAMPLE:
- Turn t: `history.loop = 3`, last intents [MOVE_E, MOVE_W, MOVE_E]. Action: MOVE_N; comment “AVOID_LOOP; exploring north to clear congestion.”{loop_example_suffix}
- Turn t+1: loop resets to 0 → reassess hazards, then resume goal-oriented planning.

COMMENT & COMMUNICATION GUIDELINES:
- Begin comments with a status token (e.g., “OK;”, “BLOCKED_AGENT(…)”) and keep them ≤25 words.
- When you take a detour or STAY to break a loop, explain it so teammates know you’re clearing space.
- If `contended_neighbors` flags a direction, STAY or coordinate first—the warning arrived from last turn’s collision.
{comment_extra_line}

OUTPUT CONTRACT:
Return ONE JSON object with this exact shape (no code fences, no prose):
{{"action":{{"kind":"{action_contract}","direction":"N|E|S|W" (only for MOVE),"payload":null or an object for MARK}},"comment":"OK; <=25 words"}}.
Example: {{"action":{{"kind":"MOVE","direction":"N","payload":null}},"comment":"OK; advancing north to scout"}}.
Do NOT emit explanations outside that JSON blob.

EXECUTION RULES:
1. Read <OBSERVATION_JSON>.
2. Respect walls, bounds, NO_GO markers, and agent collisions visible in the patch.
3. Consult `adjacent` (NESW labels) plus `recent_positions` to avoid immediate backtracking unless it is the only safe option.
4. Use `history`, `goal_sensor`, `neighbors_in_view`, `artifacts_in_view`, and `inbox` to inform your choice.
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
    parts = ["MOVE_N/E/S/W, STAY, MARK (drop NO_GO cones to warn teammates)"]
    if "COMMUNICATE" in action_kinds:
        parts.append(f"COMMUNICATE (one structured radio message per turn, range {radio_range}).")
    if "ASK_ORACLE" in action_kinds:
        parts.append("ASK_ORACLE (request global guidance; peer radio is disabled while oracle is available).")
    return " ".join(parts)


def _message_behavior_line(action_kinds: List[str]) -> str:
    if "COMMUNICATE" in action_kinds:
        return "Messages you send this turn arrive in teammate inboxes at the start of the next turn; broadcast before stepping back into a contested direction."
    return "Peer radio is disabled for this baseline; plan routes assuming you will not receive teammate messages."


def _loop_example_suffix(action_kinds: List[str]) -> str:
    if "COMMUNICATE" in action_kinds:
        return " Optionally COMMUNICATE “rerouting north to break loop.”"
    return ""


def build_prompt_header(
    *,
    radio_range: int,
    oracle_enabled: bool,
    action_kinds: List[str],
) -> str:
    allow_comm = "COMMUNICATE" in action_kinds
    allow_oracle = oracle_enabled or ("ASK_ORACLE" in action_kinds)

    actions_sentence = _actions_sentence(action_kinds, radio_range)
    message_behavior_line = _message_behavior_line(action_kinds)
    loop_example_suffix = _loop_example_suffix(action_kinds)

    comm_tool_line = ""
    if allow_comm:
        comm_tool_line = (
            "- COMMUNICATE — one structured message to share intent, hazards, or reroutes (range "
            f"{radio_range}). Use it when loops grow or before entering contested cells.\n"
        )

    oracle_tool_line = ""
    if allow_oracle:
        oracle_tool_line = (
            "- ASK_ORACLE — spend the turn requesting the Oracle’s recommendation. The reply arrives before your next decision; follow it or briefly justify overrides.\n"
        )

    if not allow_comm:
        comm_comment_line = "- Radio is disabled; rely on MOVE/STAY/MARK (and artifacts) to coordinate."
        communication_execution_line = "Radio is disabled this run; skip peer communication actions."
    else:
        comm_comment_line = "- Use COMMUNICATE to broadcast reroutes, hazards, or intent when helpful."
        communication_execution_line = "If you COMMUNICATE, send the minimal helpful message."

    comment_extra_line = comm_comment_line + "\n"

    execution_rule_line = "Stay focused on MOVE/STAY/MARK actions per the rules above."
    if allow_oracle:
        comment_extra_line += "- Reference any Oracle guidance you follow (e.g., “ORACLE; following move east suggestion”).\n"
        execution_rule_line = "If you choose ASK_ORACLE, remain in place; the reply arrives before the next decision."
        if not allow_comm:
            communication_execution_line = "Peer radio is disabled while the Oracle is available; coordinate via ASK_ORACLE or artifacts."
    else:
        # remove trailing newline when no oracle guidance was appended
        comment_extra_line = comment_extra_line.rstrip("\n") + "\n"

    action_contract = "|".join(action_kinds)

    return CORE_HEADER_TEMPLATE.format(
        actions_sentence=actions_sentence,
        message_behavior_line=message_behavior_line,
        comm_tool_line=comm_tool_line,
        oracle_tool_line=oracle_tool_line,
        loop_example_suffix=loop_example_suffix,
        comment_extra_line=comment_extra_line,
        action_contract=action_contract,
        communication_execution_line=communication_execution_line,
        execution_rule_line=execution_rule_line,
    )
