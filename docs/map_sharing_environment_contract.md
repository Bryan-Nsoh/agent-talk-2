# Map-Sharing Environment Contract (LLM Observation & Prompt Spec)

This document defines the environment representation and prompt contract for the no-communication + map-sharing baselines. It is intended as the source of truth for how the grid, symbols, and observation JSON are structured.

---

## Prompt Header (to be used verbatim)

```text
OBJECTIVE:
Reach the goal yourself and help your teammates reach it too. Agents should be cooperative and avoid blocking each other. Progress means reducing everyone’s distance to the goal, exploring new corridors, and guiding the whole team toward completion.

GRID REPRESENTATION:

The world is a 2D grid provided as:

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
- grid.rows[y][x] gives the symbol at position (x, y).
- y is the row index: 0 = TOP, increases DOWNWARD.
- x is the column index: 0 = LEFT, increases RIGHTWARD.
- Your position self.pos = {"x": X, "y": Y} corresponds to grid.rows[Y][X].

COORDINATE SYSTEM:
- X increases rightward (west to east).
- Y increases downward (north to south).
- Moving SOUTH increases Y, moving NORTH decreases Y.
- Moving EAST increases X, moving WEST decreases X.

SYMBOLS (SEE LEGEND IN THE OBSERVATION FOR FULL DESCRIPTIONS):

Base tiles (these define the environment itself):
  # = WALL (impassable)
  G = GOAL (reach this tile to finish for that agent)
  X = UNKNOWN (unexplored in your current map)

Free cell states (these encode your relationship to the cell, by priority):

  @ = SELF (you are currently standing here)
  1,2,3... = OTHER AGENTS (visible teammates located here)
  ! = COLLISION (cell you tried to enter last turn and failed)
  * = RECENT TRAIL (one of your last 3 positions)
  ~ = VISITED (you stood here before, but not in your last 3 positions)
  . = FREE (known traversable, never visited by you)

SYMBOL PRIORITY (WHAT YOU SEE WHEN STATES OVERLAP):

For each cell (x,y), the environment applies this logic:

  if base tile is WALL      → '#'
  else if base tile is GOAL → 'G'
  else if base tile is UNKNOWN → 'X'
  else (base is FREE):
    if self.pos == (x,y) → '@'
    else if a visible agent (from neighbors_in_view) is at (x,y) → '1','2','3',...
    else if this is the last collision target cell → '!'
    else if this cell is in your last 3 positions → '*'
    else if you have ever stood on this cell before → '~'
    else → '.'

So every cell is exactly one character. Free cells receive the highest-priority overlay symbol that applies.

SELF AND NEIGHBORS:

- "self": {"agent_id": string, "pos": {"x": int, "y": int}} gives your current position.
- "neighbors_in_view": list of other agents you currently see within a small radius (e.g. up to 2 cells in any direction). Each entry is:
    {"agent_id": string, "pos": {"x": int, "y": int}}

  Only agents currently in view are listed here and drawn as digits in the grid. There is no “last seen” memory for other agents in the grid: if they leave your view, their symbol disappears.

ADJACENT CELLS AND FRONTIERS:

- "adjacent": list of 4 entries, summarizing the four cardinal neighbors:

    {"dir": "N"|"E"|"S"|"W", "state": "FREE"|"WALL"|"GOAL"|"AGENT"|"OUT_OF_BOUNDS"}

  This is derived from the grid using your self.pos and neighbors_in_view.

- "adjacent_frontiers": list of positions {"x": int, "y": int} for UNKNOWN neighbors that you can reach in one step (frontiers of explored space):
  - A frontier is an 'X' cell that borders a known FREE or GOAL cell.
  - adjacent_frontiers are such cells at distance 1 from your current position.

GOAL:

- The goal is the unique tile where grid.rows[y][x] == 'G'.
- "goal_known": boolean indicating whether the environment has discovered the goal’s location (via any agent).
- "goal_pos": {"x": int, "y": int} if goal_known is true; null otherwise.
- At the start of an episode, goal_known may be false. Once any agent sees the goal, goal_known becomes true and goal_pos is set for everyone for this map-sharing baseline.

MAP SHARING:

- "map_sharing": "none" | "radio_sync" | "global"

  - "none": your grid updates only from your own observations.
  - "radio_sync": when you are within radio range of another agent, your base tile knowledge (WALL/FREE/GOAL/UNKNOWN) is merged with theirs. Overlays that depend on you (@, your '~', your '*', and '!') remain personal.
  - "global": all agents share and update the same base tile grid (WALL/FREE/GOAL/UNKNOWN). Overlays (@, ~, \*, !) are still per-agent.

- IMPORTANT: Whenever a cell is not 'X', treat its classification as reliable (even if you did not personally see it).

COLLISION FEEDBACK:

- "last_result": summarizes the outcome of your previous action:

  {
    "kind": "OK" | "FINISHED" | "BLOCK_WALL" | "BLOCK_AGENT" | "SWAP_CONFLICT",
    "cell": {"x": int, "y": int} | null,
    "opponents": [string]
  }

  - If kind is OK:
    - cell is null,
    - opponents is [].
  - If kind is FINISHED:
    - cell is the GOAL position (goal_pos),
    - opponents is [].
  - If kind is a block (BLOCK_WALL, BLOCK_AGENT, SWAP_CONFLICT):
    - cell is the target you tried to move into,
    - opponents lists the agents you collided with (may be empty for walls).
  - The last collision cell is also marked as '!' in grid.rows for exactly one turn.

TURN AND BUDGET:

- "turn_index": current turn (0-based).
- "max_turns": maximum number of turns allowed in this episode.

COMMUNICATION:

- There is NO radio communication in this baseline (comm_strategy = "none").
- You will receive no messages and cannot send messages. All coordination must come from your movement decisions.

DECISION HIERARCHY (APPLY IN ORDER):

1. DO NOT HIT WALLS:
   - Never choose MOVE into a cell that is a wall ('#') or OUT_OF_BOUNDS in "adjacent".

2. RESPECT YOUR LAST COLLISION:
   - If last_result.kind is a block and last_result.cell is not null (and this cell shows as '!'), avoid choosing a MOVE that targets that same cell again, unless there is absolutely no safe alternative.

3. BREAK IMMEDIATE BACKTRACKS:
   - Use your current position and nearby '*' cells to detect very recent backtracking (e.g., going back and forth between two cells).
   - If your last few moves bounced between a small set of cells, prefer a different direction or STAY instead of repeating the same pattern.

4. EXPLORE UNKNOWN FIRST:
   - Prefer MOVE actions that step into positions listed in adjacent_frontiers when they are safe (not obviously repeating a collision, not stepping into an AGENT).
   - This expands your known map (converting 'X' to '.', '~', or '*') and helps find the goal.

5. USE GOAL LOCATION WHEN KNOWN:
   - While goal_known is false, prioritize exploration and collision avoidance.
   - Once goal_known is true, plan routes that move you closer to goal_pos using FREE cells ('.', '~', '*') while avoiding:
     - walls '#',
     - the last collision cell '!',
     - and dense clusters of '*' in narrow corridors that indicate recent thrashing.
   - If an adjacent cell is GOAL ('G'), MOVE into it unless you have clear evidence it will cause an immediate collision.

6. WHEN ALL OPTIONS LOOK BAD:
   - If every FREE neighbor either:
     - is the last collision target, or
     - is part of your recent trail ('*') forming a tight loop,
     then choose STAY and explain why, or choose the direction that leads into the least recently visited '~' region to unwind the loop.

REASONING NOTE (comment field, ≤25 words):

- Briefly explain why you chose this action.
- Refer to coordinates and grid features when helpful, e.g. “exploring X at (3,1)” or “avoiding last collision at (5,2)”.
- If you move into a '*' or '~' cell, justify why revisiting is necessary (e.g. “must backtrack to reach new frontier”).

OUTPUT CONTRACT:

Return ONE JSON object with this exact shape (no code fences, no extra text):

{"action":{"kind":"MOVE"|"STAY","direction":"N"|"E"|"S"|"W" (only present for MOVE)},"comment":"OK; <=25 words"}

Examples:
{"action":{"kind":"MOVE","direction":"N"},"comment":"Exploring frontier at (3,1); no wall or collision risk; goal still known at (6,1)."}
{"action":{"kind":"STAY"},"comment":"All free moves repeat last collision or recent trail; waiting to avoid blocking teammate at (5,2)."}

Do NOT emit explanations outside that JSON blob.

=== INPUTS ===
<OBSERVATION_JSON>
... (JSON follows)
</OBSERVATION_JSON>
```

---

## Observation JSON Contract (with Example)

### Required fields

- `protocol_version: string`
- `turn_index: int`
- `max_turns: int`
- `grid: { width: int, height: int, rows: List[List[str]] }`
- `legend: { symbol: description, ... }`
- `self: { agent_id: str, pos: { x: int, y: int } }`
- `neighbors_in_view: List[{ agent_id: str, pos: { x: int, y: int } }]`
- `adjacent: List[{ dir: "N"|"E"|"S"|"W", state: "FREE"|"WALL"|"GOAL"|"AGENT"|"OUT_OF_BOUNDS" }]`
- `adjacent_frontiers: List[{ x: int, y: int }]`
- `goal_known: bool`
- `goal_pos: { x: int, y: int } | null`
- `last_result: { kind: "OK"|"FINISHED"|"BLOCK_WALL"|"BLOCK_AGENT"|"SWAP_CONFLICT", cell: { x: int, y: int } | null, opponents: List[str] }`
- `map_sharing: "none"|"radio_sync"|"global"`

### Sample observation JSON

```json
{
  "protocol_version": "3.0.0",
  "turn_index": 5,
  "max_turns": 100,

  "grid": {
    "width": 8,
    "height": 5,
    "rows": [
      ["#","#","#","#","#","#","#","#"],
      ["#",".",".","X","X","2","G","#"],
      ["#",".","@","~","~","!",".","#"],
      ["#","*",".",".",".",".",".","#"],
      ["#","#","#","#","#","#","#","#"]
    ]
  },

  "legend": {
    "#": "WALL (impassable)",
    "G": "GOAL (reach to finish)",
    "X": "UNKNOWN (unseen)",
    ".": "FREE (seen, not visited)",
    "~": "FREE (visited 2+ turns ago)",
    "*": "FREE (in your last 3 positions)",
    "!": "FREE (your last collision target)",
    "@": "SELF (you are here)",
    "1,2,3...": "OTHER AGENTS (visible in neighbors_in_view)"
  },

  "self": {
    "agent_id": "a1",
    "pos": {"x": 2, "y": 2}
  },

  "neighbors_in_view": [
    {"agent_id": "a2", "pos": {"x": 5, "y": 1}}
  ],

  "adjacent": [
    {"dir": "N", "state": "FREE"},
    {"dir": "E", "state": "FREE"},
    {"dir": "S", "state": "FREE"},
    {"dir": "W", "state": "FREE"}
  ],

  "adjacent_frontiers": [
    {"x": 3, "y": 1},
    {"x": 4, "y": 1}
  ],

  "goal_known": true,
  "goal_pos": {"x": 6, "y": 1},

  "last_result": {
    "kind": "BLOCK_AGENT",
    "cell": {"x": 5, "y": 2},
    "opponents": ["a2"]
  },

  "map_sharing": "radio_sync"
}
```

This example adheres to the contract above and should be treated as the canonical reference format for the map-sharing observation used in our no-communication baselines.


---

## Environment Rules & Behavior (Exhaustive Contract)

This section describes how the environment behaves behind the scenes: the simulation loop, movement rules, collisions, visibility, map updates, map sharing, and edge cases. The aim is that any future implementation matching these rules will produce observations consistent with the contract above.

### 1. Global Invariants

- The world is a fixed rectangular grid of size `grid.width × grid.height`.
- Base tiles are static:
  - Each cell has a base type in `{WALL, FREE, GOAL}`.
  - The base type never changes during an episode.
  - There is exactly one GOAL tile.
- UNKNOWN cells (`'X'`) in the agent’s grid are “no information yet”. When revealed, they become one of `'#'`, `'.'`, `'~'`, `'*'`, `'G'` according to the overlay rules.
- Time is discrete: `turn_index` starts at 0 and increments by 1 each turn until `max_turns` or all agents have finished.
- A fixed set of agents is created at episode start, each with a unique `agent_id`.
- All active agents act simultaneously each turn (no ordering bias in decision application).

### 2. Turn Loop (Per Episode)

Each turn consists of the following phases:

1) **Observation phase**
   - For each active (not finished) agent:
     - The environment constructs an Observation JSON matching this contract.
     - This includes:
       - Current `grid`, `self`, `neighbors_in_view`, `adjacent`, `adjacent_frontiers`,
       - `goal_known`, `goal_pos`,
       - `last_result` from the previous turn (or initial default for turn 0),
       - `map_sharing` mode.

2) **Decision phase**
   - Each agent (policy) receives its Observation and returns a `Decision`:
     - `"action": {"kind": "MOVE", "direction": "N"|"E"|"S"|"W"}` or
     - `"action": {"kind": "STAY"}`.
   - All decisions are collected before any moves are resolved.

3) **Movement resolution phase**
   - For all agents, the environment interprets `MOVE` as an attempt to step one cell in the given direction; `STAY` attempts no movement.
   - The engine computes proposed target cells for each agent and resolves outcomes:
     - Successful moves.
     - Collisions with walls/out-of-bounds.
     - Agent-agent collisions (including swap conflicts).
   - Results are mapped into `last_result` for each agent.

4) **Map update phase**
   - Each agent’s local map is updated based on what it can currently see (visibility radius).
   - Map sharing logic is applied (depending on `map_sharing`).
   - Overlay symbols (`@`, digits, `!`, `*`, `~`, `.`) are recomputed from:
     - New positions,
     - The agent’s visit history,
     - `last_result` for that agent,
     - `neighbors_in_view`.

5) **Termination check**
   - If `turn_index + 1 == max_turns`, episode ends with timeout.
   - Agents whose move successfully entered the GOAL cell are marked finished:
     - Their `last_result.kind` becomes `"FINISHED"`.
     - They no longer move on subsequent turns.
   - If all agents are finished, the episode ends early.

6) `turn_index` increments by 1, loop continues or stops depending on termination.

### 3. Movement Rules

#### 3.1 Allowed actions

- MOVE_N: attempt to move from (x,y) to (x, y-1).
- MOVE_S: attempt to move from (x,y) to (x, y+1).
- MOVE_E: attempt to move from (x,y) to (x+1, y).
- MOVE_W: attempt to move from (x,y) to (x-1, y).
- STAY: remain in place; no target cell is proposed.

#### 3.2 Off-grid and walls

For a MOVE to target (tx,ty):

- If (tx,ty) is outside `[0,width-1] × [0,height-1]`:
  - The move fails.
  - Externally, `last_result.kind = "BLOCK_WALL"` and `last_result.cell = {"x": tx, "y": ty}`.
  - In `grid.rows`, no `@` moves; collision symbol `'!'` is only drawn if `last_result.cell` lies within the valid grid bounds. Off-grid collisions are signaled via `last_result` only (no '!' glyph).
- If (tx,ty) is a WALL base tile:
  - The move fails.
  - `last_result.kind = "BLOCK_WALL"`, `last_result.cell = {"x": tx, "y": ty}`, `opponents = []`.
  - The cell (tx,ty) is also marked as `'!'` for that agent’s grid on the next turn.

#### 3.3 Agent-agent collisions

Let multiple agents propose targets in the same turn. The environment resolves them in a collision-aware way:

- **Simple blocking (BLOCK_AGENT)**:
  - Two or more agents attempt to move into the same FREE or GOAL cell that is not a pure swap.
  - None of them successfully enter; all stay in their original tile.
  - For each such agent:
    - `last_result.kind = "BLOCK_AGENT"`,
    - `last_result.cell` is the contested target (tx,ty),
    - `opponents` lists the other agent_ids that also attempted that cell.
  - The contested cell is marked as `'!'` for that agent for one turn.

- **Swap conflict (SWAP_CONFLICT)**:
  - Exactly two agents, A and B, propose to move into each other’s current positions (A→B’s cell, B→A’s cell) at the same time.
  - Both moves fail; neither agent enters the other’s cell.
  - For both agents:
    - `last_result.kind = "SWAP_CONFLICT"`,
    - `last_result.cell` is the target cell they attempted,
    - `opponents` = [other_agent_id].
  - The target cells are marked as `'!'` for the respective agents for one turn.

- **Collisions involving finished agents**:
  - Once an agent has `last_result.kind = "FINISHED"`, it is considered finished and does not move.
  - Other agents may move through or onto the GOAL cell as if it were FREE (post-finish).
  - For this contract, finished agents never block and are not considered in collision checks:
    - Finished agents do NOT block future movement.
    - The GOAL tile remains 'G'.
    - Finished agents are not listed in neighbors_in_view and do not appear as digits on the grid.

#### 3.4 Successful moves

- If a MOVE targets a FREE cell and no collision with walls/out-of-bounds/other agents occurs:
  - The agent moves to (tx,ty).
  - `last_result.kind = "OK"`,
  - `last_result.cell = null`,
  - `opponents = []`.

- If a MOVE targets the GOAL cell and there is no blocking conflict:
  - The agent moves to (tx,ty) where base tile is GOAL.
  - `last_result.kind = "FINISHED"`,
  - `last_result.cell = {"x": tx, "y": ty}`,
  - `opponents = []`.
  - On subsequent turns the agent is considered finished and does not move.

- STAY:
  - The agent remains at (x,y).
  - If no collision-like event is associated with staying, then:
    - `last_result.kind = "OK"`,
    - `last_result.cell = null`,
    - `opponents = []`.

### 4. Visibility & neighbors_in_view

- Each agent has an internal visibility radius `R_vis` (for the map-sharing baseline we fix `R_vis = 2` using Manhattan distance).
- Another agent B is included in A’s `neighbors_in_view` if:
  - B is not finished, and
  - Manhattan distance(A.pos, B.pos) ≤ R_vis.
- Only agents listed in `neighbors_in_view` may appear as digits (`'1','2','3',...`) in A’s grid, at the corresponding positions.
- When an agent leaves visibility range, its digit disappears from the grid and it is removed from `neighbors_in_view`.

### 5. Map Update & Symbol Semantics

This describes how `grid.rows` is updated each turn for a given agent.

#### 5.1 Base tile revelation

- For each agent A, there is an internal base map (WALL/FREE/GOAL/UNKNOWN) that drives `#`, `G`, `X`, and FREE overlays.
- On each turn, after movement resolution:
  - The environment reveals base tiles in a visibility neighborhood around A (radius `R_vis`):
    - UNKNOWN (`'X'`) cells in that region become:
      - `'#'` if base tile is WALL,
      - `G` if GOAL,
      - `'.'` (or its overlay variant) if FREE.
  - Once revealed as WALL or GOAL or FREE, a cell never reverts to UNKNOWN.

#### 5.2 Visit history and trail

For each agent independently:

- Maintain an ordered list of its last K positions for K ≥ 3 (e.g. K = 3):
  - On each turn (whether the agent MOVEs or STAYs):
    - Prepend the current position to the list.
    - Truncate to the last K entries.

- Maintain a set of visited cells for that agent:
  - On each turn, add the current position to this set.

- Symbol assignment for FREE cells in A’s view:
  - If the free cell is the agent’s current position: `'@'`.
  - Else if a visible other agent is at the cell: `'1','2','3',...`.
  - Else if the cell is `last_result.cell` for a BLOCK_* outcome and that cell is within grid bounds: `'!'`.
  - Else if the cell is in A’s last K positions list: `'*'`.
  - Else if the cell is in A’s visited set: `'~'`.
  - Else if the base tile is FREE but neither visited nor recent: `'.'`.

The precedence ensures only one symbol is shown per cell for A, even if multiple conditions apply.

#### 5.3 Collision marker lifetime

- If `last_result.kind` is `BLOCK_WALL`, `BLOCK_AGENT`, or `SWAP_CONFLICT` with `last_result.cell = c`:
  - Cell `c` is drawn as `'!'` on the next turn only for that agent.
  - If another collision occurs on the following turn, the `'!'` moves to the new collision cell.
  - If the next turn’s action is successful, no `'!'` is shown (unless some other logic decides otherwise).

### 6. Map Sharing Semantics

Map sharing only affects base tiles and known free cells; overlays (`@`, digits, `!`, `*`, `~`) remain per-agent.

Three modes:

1. **map_sharing = "none"**
   - Each agent A maintains its own base map.
   - Revelations from A’s visibility only update A’s map.
   - No base-tile information is shared between agents.
   - Agents can still infer others’ presence via `neighbors_in_view` when close, but their grid’s base tiles stay local.

2. **map_sharing = "radio_sync"**
   - A radios base-tile information to nearby agents when within radio range `R_radio`.
   - For any pair of agents (A,B):
     - If distance(A.pos, B.pos) ≤ R_radio:
       - Their base maps are merged for the cells where at least one knows a non-UNKNOWN base tile.
       - If A knows a cell as FREE and B still has UNKNOWN, B’s cell becomes FREE (with appropriate overlay for B).
       - If A knows a cell as WALL or GOAL, that classification overwrites UNKNOWN in B’s map.
   - Overlays remain local:
     - A’s `@`, `~`, `*`, `!` do not show up in B’s grid (and vice versa).

3. **map_sharing = "global"**
   - All agents operate on a single shared base map:
     - Base tiles (WALL/FREE/GOAL/UNKNOWN) are updated from *any* agent’s observations.
   - Overlays remain per-agent:
     - Each agent has its own trail and collision markers drawn over the shared base.

**Exploration invariants:**

- Base tiles are never contradictory: a cell won’t be wall for one agent and free for another in the same mode.
- Once a cell is known non-UNKNOWN in any agent’s base map in "radio_sync" or "global", all agents in range (or all agents, respectively) eventually see a consistent base symbol (`#`, `G`, or a free overlay).

### 7. Goal & Termination Behavior

- The GOAL is a static tile with base symbol `'G'`.
- `goal_known` and `goal_pos` semantics:
  - Initially:
    - `goal_known = false`,
    - `goal_pos = null`,
    - grid rows may or may not show `'G'` depending on initial visibility.
  - When any agent’s visibility region reveals the GOAL:
    - `goal_known` becomes true for all agents in that run.
    - `goal_pos` is set to the GOAL’s coordinates and remains constant.
  - After goal_known is true:
    - `'G'` must be visible at `grid.rows[goal_pos.y][goal_pos.x]` for all agents once their base map covers that cell via sharing.

- Finishing:
  - An agent finishes when it successfully moves into the GOAL tile.
  - On the finishing turn:
    - `last_result.kind = "FINISHED"`,
    - `last_result.cell = goal_pos`,
    - `opponents = []`.
  - On subsequent turns:
    - The agent is marked finished and does not move.
    - Its trail overlays stop updating.
    - Other agents may pass through the GOAL tile; from their perspective, it remains `'G'` plus their own overlays.

- Episode termination:
  - When all agents are finished:
    - The episode ends early with success.
  - When `turn_index + 1 == max_turns` and some agents are not finished:
    - The episode ends due to timeout.
    - Interpretation of partial success vs failure is up to the evaluation harness; the environment only enforces the time limit.

### 8. Edge Cases & Guarantees

This subsection enumerates important edge cases and how they are handled.

1. **Off-grid moves**
   - A MOVE that targets off-grid coordinates is treated as `BLOCK_WALL`:
     - `last_result.kind = "BLOCK_WALL"`,
     - `last_result.cell` is set to the off-grid target,
     - `opponents = []`.

2. **Collision into GOAL**
   - If two or more agents simultaneously attempt to move into GOAL from different directions:
     - This is treated as a BLOCK_AGENT conflict:
       - None of the moves succeed.
       - Each such agent receives `last_result.kind = "BLOCK_AGENT"`, `cell = goal_pos`, `opponents` listing the other contenders for that cell.

3. **Swap on GOAL**
   - If one agent is on GOAL and another tries to swap positions with it:
     - This is treated as `SWAP_CONFLICT`:
       - Neither agent moves.
       - Each sees `last_result.kind = "SWAP_CONFLICT"` with the relevant target cell.

4. **Simultaneous finish and collision elsewhere**
   - It is possible for some agents to finish while others collide elsewhere on the same turn.
   - Each agent’s `last_result` reflects its own outcome independently.

5. **Multiple frontiers adjacent**
   - `adjacent_frontiers` may contain multiple entries (e.g. two or more adjacent 'X' cells).
   - The agent is free to choose any of them, guided by the decision hierarchy (e.g. preferring ones that align with goal_pos once known).

6. **No frontiers available**
   - If `adjacent_frontiers` is empty and goal_known is false:
     - The agent must use the full grid and trail structure to choose a direction to push deeper into known free space in search of new UNKNOWN zones.
     - The environment does not guarantee access to a frontier every turn.

7. **Finished agents and neighbors_in_view**
   - Finished agents are not included in `neighbors_in_view`.
   - Their icons (digits) do not appear in any active agent’s grid; only the GOAL or base tiles remain.

8. **Map sharing and UNKNOWN overrides**
   - If in "radio_sync" or "global" modes, one agent knows a cell as FREE and another still has it as UNKNOWN:
     - After sharing, both agents should see a FREE-compatible symbol (`.`, `~`, `*` or overlays), not 'X'.
   - No agent ever reverts a cell from known (non-'X') back to UNKNOWN.

9. **Self-consistency of `adjacent`**
   - For each entry in "adjacent", the state must be consistent with "grid" and "neighbors_in_view":
     - If state == FREE, then the cell is not '#', not 'X', and not out-of-bounds.
     - If state == GOAL, the cell must be 'G'.
     - If state == AGENT, there must be a matching entry in neighbors_in_view at that coordinate (unless it is self, in which case '@' and state==AGENT are never combined; self is not reported as AGENT).
     - If state == OUT_OF_BOUNDS, the target coordinate lies outside [0,width-1] × [0,height-1].

10. **Self never appears as AGENT in `adjacent`**
    - The cell where self.pos is located is represented as '@' in the grid and is not considered a neighbor.
    - Adjacent entries with state == AGENT always refer to other agents, not self.

11. **Consistency between last_result and '!'**
    - If last_result.kind is a blocking kind and last_result.cell is within bounds:
      - Exactly one cell in grid.rows must show '!' for that agent and it must be at last_result.cell.
    - If last_result.kind is OK or FINISHED:
      - No '!' should be shown in the grid for that agent.

These rules, together with the earlier Prompt and Observation JSON sections, constitute the full contract for the map-sharing environment behavior in the no-communication baseline. Any implementation used for experiments should adhere to this contract or clearly document deviations.
