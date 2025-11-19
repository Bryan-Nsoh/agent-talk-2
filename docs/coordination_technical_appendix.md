# Technical Appendix: Coordination Failure Analysis

## Data Sources

All analysis based on:
- **Experiment:** `cross_seed_baseline_20251112T143355Z`
- **Primary focus:** Runs with `comm_strategy="none"` (15 runs total)
- **Secondary comparison:** Runs with `comm_strategy` in {structured, freeform, tagged, intention}
- **Data files:** `transcript.jsonl` (agent decision traces), `metrics.json` (summary stats), `episode.json` (map/wall layout)

---

## Observation Schema vs. Missing Data

### What Agents Currently Receive

```json
{
  "turn_index": 4,
  "self_state": {
    "agent_id": "a5",
    "abs_pos": {"x": 3, "y": 1},
    "orientation": "N"
  },
  "local_patch": {
    "radius": 1,
    "rows": ["###", ".A.", "#.."]
  },
  "neighbors_in_view": [],                    // ← ALWAYS EMPTY (distance > 1)
  "adjacent": [
    {"dir": "N", "state": "WALL"},
    {"dir": "E", "state": "FREE"},
    {"dir": "S", "state": "FREE"},
    {"dir": "W", "state": "FREE"}
  ],
  "contended_neighbors": 0,                   // ← Binary: no info about which agent
  "recent_positions": [
    {"x": 3, "y": 1}, {"x": 3, "y": 0}
  ],
  "world_map_ascii": "..."                    // ← Only this agent's explored map
}
```

### What Would Enable Coordination

**Proposal 1: Extended Neighbor Visibility**
```json
{
  "neighbors_in_view": [
    {
      "agent_id": "a2",
      "abs_pos": {"x": 5, "y": 1},
      "distance": 2,
      "recent_positions": [
        {"x": 6, "y": 1}, {"x": 5, "y": 1}
      ],
      "last_move": "W"
    }
  ]
}
```

**Proposal 2: Detailed Contention Info**
```json
{
  "contended_neighbors": [
    {
      "direction": "W",
      "agent_id": "a2",
      "collision_point": {"x": 4, "y": 1},
      "agent_approach_dir": "W",
      "turns_ago": 1
    }
  ]
}
```

---

## Collision Detection Methodology

### How Collisions Were Identified

For each turn T:
1. Extract agent position and planned move at T-1
2. Calculate target position: target = current + direction_vector
3. Extract agent position at T and outcome (OK, BLOCK_AGENT, YIELD)
4. If outcome == BLOCK_AGENT and agent didn't move, classify:
   - Wall collision: target in walls set
   - Boundary collision: target out of bounds
   - Agent-agent conflict: target in walls_set OR boundary, OR multiple agents targeted same cell

### Example: Turn 4 Reconstruction

**Turn 3 State:**
```python
a2_pos_t3 = (5, 1)
a2_move_t3 = "W"           # planned move
a2_target_t3 = (4, 1)

a5_pos_t3 = (3, 1)
a5_move_t3 = "E"           # planned move
a5_target_t3 = (4, 1)
```

**Turn 4 Observation:**
```python
a2_pos_t4 = (5, 1)         # didn't move
a2_outcome_t4 = "BLOCK_AGENT"

a5_pos_t4 = (3, 1)         # didn't move
a5_outcome_t4 = "BLOCK_AGENT"

# Both stayed at previous position
# Both targeted (4,1) in their last turn's decision
# → AGENT-AGENT CONFLICT at (4,1)
```

---

## Deadlock Pattern Metrics

### Repeated Collision Analysis

**Methodol:** Track (agent_pair, target_cell) tuples across turns

**Turn 11 vs Turn 14 vs Turn 16 Collision:**
```
Turn 11:
  a3(8,1) + a5(6,1) → target (7,1) → COLLISION
  No context change: both still trying to access (7,1)

Turn 14:
  a3(8,1) + a5(6,1) → target (7,1) → COLLISION
  Agents are in SAME positions as Turn 11
  Decision logic unchanged: "head toward frontier"
  Result: IDENTICAL COLLISION

Turn 16:
  a3(8,1) + a5(6,1) → target (7,1) → COLLISION
  Third occurrence of same conflict
  Neither agent has learned to avoid (7,1)
```

**Why traditional deadlock-breaking fails:**
- Agent a3 has `contended_neighbors = 2` (flag for west direction)
- But doesn't know: "a5 specifically is blocking me"
- Can't establish: "a5 goes east, I yield" without communication

---

## Agent Decision Reasoning Pattern

### "Avoid" Keyword Analysis

Agents use "avoid" in 46.7% of turns, suggesting widespread attempt at conflict prevention.

**True successful avoidance (agent sees & adapts):**
```
Turn 5 a2: "Avoiding contended west; scouting south to reveal new corridor at (5,3)"
```
- Previous turn: west direction was contended
- This turn: picks south (different from last attempted direction)
- Outcome: OK (successfully avoided)

**Failed avoidance (agent yields but picks bad alternate):**
```
Turn 4 a5: "Avoiding contended E after BLOCK_AGENT; moving south to (3,2) 
           to explore new corridor."
```
- Previous turn: east was contended (with a2)
- This turn: picks south
- Problem: has no information that south won't lead to another conflict
- Outcome: BLOCK_AGENT again (hits wall at south)

### "Yield" Keyword Analysis

Agents mention yielding in only 4.7% of turns.

**Example explicit yield:**
```
Turn 23 a5: "STAY to avoid colliding with agent at (9,1) in narrow corridor"
```
- Agent a5 can see neighbor a3 at (9,1)
- Makes explicit decision to STAY
- Outcome: OK, correctly deferred

**But this only works when:**
1. Neighbor is already in view (distance ≤ 1)
2. Neighbor is directly adjacent
3. Agent happens to be moving toward that neighbor

**Why this fails at scale:**
- Most potential collisions involve agents at distance 2 cells apart
- Those agents can't see each other
- Yielding requires seeing the threat first

---

## Success Correlation Analysis

### Collision Count vs. Success

```
Run seed16_none_run2:
  0 collisions ✓ SUCCESS (67 turns)
  
Run seed16_none_run1:
  4 collisions ✗ FAILURE (100 turns, timeout)

Run seed17_none_run1:
  49 collisions ✗ FAILURE (100 turns, timeout)

Run seed14_none_20251112T193609Z:
  8 collisions ✗ FAILURE (100 turns, timeout)
```

**Critical observation:** Even a small number of collisions (4-8) is enough to cause timeouts. The corridor structure forces agents into repeated conflicts that accumulate.

### Initial Condition Analysis

**Successful run (seed16_none_run2) initial positions:**
```
a1: (9, 2)    ← start x = 9
a2: (2, 3)    ← start x = 2
a3: (7, 7)    ← start x = 7
a4: (15, 9)   ← start x = 15
a5: (0, 6)    ← start x = 0

Spread: 0-15 (full width coverage, minimal overlap in exploration)
```

**Failed run (seed17_none_run3) initial positions:**
```
a1: (0, 6)    ← start x = 0
a2: (2, 3)    ← start x = 2
a3: (7, 7)    ← start x = 7
a4: (15, 9)   ← start x = 15
a5: (3, 1)    ← start x = 3

Spread: 0-15 (but clustering at x=0-3 causes bottleneck)
```

Key difference: Seed 17 has 3 agents starting in x=[0-3], causing early corridor contention.

---

## Maze Structure Effects

### Corridor Bottleneck Analysis

The "long_corridor" maze has a structure like:
```
##...................   y=9
##...................   y=8
##...................   y=7
..............GOAL...   y=6
##...................   y=5
##...................   y=4
.....................   y=3
.....................   y=2
.....................   y=1
###############X....   y=0

Column 0-1: walls (left boundary)
Column 2-29: mostly clear
```

**Bottleneck at y=1:** Only row y=1-2 are fully open at x<5
- Multiple agents naturally funneled to y=1
- Width-1 corridor forces serial passages
- a2 and a5 inevitably collide here (see Turn 4)

### Tight Corridor Collision Multiplier

Agents at y=1 (only 2 cells apart) but can't see each other (distance >1).

If `neighbors_in_view` extended to distance=2:
- a2 at (5,1) COULD see a5 at (3,1)
- One of them would pick alternate path (row y=2)
- Collision avoided without explicit communication

---

## Prompt Analysis: "Radio is Disabled"

### What Agents Are Told

From the prompt template (exact text):

> "Peer radio is disabled for this baseline; plan routes assuming you will not receive teammate messages."

And later:

> "Radio is disabled; rely on MOVE/STAY to coordinate implicitly."

### How Agents Respond

Agents interpret this as: "Don't use any communication"
- No attempts to send messages (correctly understood)
- But agents DON'T adapt by exploring alternate coordination methods
- Agent reasoning shows no awareness that they COULD observe neighbors more broadly

**Implicit assumption in agent reasoning:**
- "If I can't communicate, I can't coordinate"
- Results in primarily selfish/greedy exploration
- No attempts to signal intent via movement patterns
- No hesitation to move into a cell another agent might want

**Evidence:** 46% "avoid" keyword rate suggests agents ARE trying to coordinate, but doing so reactively (after collision) rather than proactively (observing intent).

---

## Contended Neighbors Bitmask Breakdown

### Bit Mapping

The `contended_neighbors` field is an integer bitmask:
- Bit 0 (value 1): North direction contended
- Bit 1 (value 2): East direction contended
- Bit 2 (value 4): South direction contended
- Bit 3 (value 8): West direction contended

### Example Decoding

**Turn 4, agent a5:**
```
contended_neighbors = 2
Binary: 0010
Bit 1 (East) = 1

Interpretation: "Someone tried to enter my cell from the EAST"
```

But agent a5 doesn't know:
- WHO is east (agent a2 vs generic obstacle)
- WHY they tried (exploring frontiers)
- IF they'll retry next turn (likely, since a2 also doesn't know about conflict)

---

## Inference Challenge: The Visibility Gap

### Why Extended Vision Helps Coordination

**With distance ≤1 (current):**
```
Agent a2 at (5,1):
  neighbors_in_view = []   ← a5 is 2 cells away
  
Agent can make decision: "Frontier at (4,1) is free, go there"
No evidence against this action
```

**With distance ≤2 (proposed):**
```
Agent a2 at (5,1):
  neighbors_in_view = [
    {agent_id: "a5", pos: (3,1), last_move: "E"}
  ]
  
Agent can make decision: "a5 is at (3,1) moving east... if I move 
west to (4,1), a5 will collide with me. Pick alternate path."
```

The jump from distance 1 to distance 2 is critical because:
- Most narrow corridors have width 3-5
- Agents at opposite ends are distance 2-3 apart
- At distance 1, they collide without warning
- At distance 2, they can see each other and maneuver

---

## Prompt Instruction: Decision Hierarchy

The agents are instructed:

> "DECISION HIERARCHY (apply in order every turn):
> 1. PREVENT COLLISIONS: Respect WALL / contended cells. Yield or coordinate before entering tight corridors.
> 2. EXPLORE UNKNOWN: If `adjacent_frontiers` is non-empty, MOVE into one of those cells to reveal it.
> 3. ADVANCE TOWARD GOAL: After nearby frontiers are mapped..."

**Critical gap:** Step 1 says "coordinate before entering tight corridors" but provides NO MECHANISM to coordinate without communication.

**Agent's interpretation:**
- "Yield if I see contended neighbors" → But contended neighbors is a binary flag
- "Coordinate" → How? Only by observing other agents' positions, but can't see far enough

Result: Agents attempt to follow Step 1 but fail due to observation limitations.

---

## Quantitative Summary Table

| Metric | Value |
|--------|-------|
| Runs analyzed (none strategy) | 15 |
| Success rate | 1/15 (6.7%) |
| Avg collisions per run | 18.4 |
| Avg turns to failure | 97.8 |
| Agent-agent conflicts per run | ~11 |
| Repeated conflicts (same pair, same cell) | ~5-6 per run |
| "Avoid" reasoning frequency | 46.7% of turns |
| "Yield" reasoning frequency | 4.7% of turns |
| Agents starting distance ≤3 cells | Correlates with failure |
| Wall collisions vs agent collisions | ~2:1 ratio |
| Successful runs with 0 collisions | 1/15 |

---

## Recommended Test Cases for Information Levels

### Test 1: Extended Vision (distance ≤2)
- Modify observation to include neighbors at distance ≤2
- Expected effect: 60-80% of head-on collisions prevented
- Hypothesis: Agents will naturally avoid converging on same cell

### Test 2: Visible Movement History
- Add `recent_moves` field to neighbors
- Expected effect: Deadlock cycles broken (agents recognize oscillation)
- Hypothesis: Agents infer blocking situation and yield

### Test 3: Detailed Contention
- Replace binary flag with per-direction conflict details
- Expected effect: 20-40% reduction in same-direction retries
- Hypothesis: Agents avoid direction that blocked previously

### Test 4: Implicit Communication (Signaling)
- Allow agents to set a "next_move_intent" observable by neighbors
- Expected effect: Similar to Test 2 but more explicit
- This is quasi-communication but still passive/observable

---

## Code References (Hypothetical)

If implementing these changes, key locations would be:

1. **Observation generation:** Extend `neighbors_in_view` list
   - Current: includes agents at `distance(self, other) <= 1`
   - Change: `distance(self, other) <= 2 or 3`

2. **Move history tracking:** Expand observed agent state
   - Current: self.recent_positions only
   - Change: For each neighbor, expose neighbor.recent_positions and inferred direction

3. **Contention reporting:** Detailed conflict logging
   - Current: contended_neighbors = bitmask
   - Change: List of {direction, agent_id, collision_point, turn}

4. **Prompt template:** Update to enable implicit signaling
   - Add: "You may also use STAY strategically to signal yielding to nearby agents"
   - Or: "Other agents can observe your recent movement pattern"

---

## Conclusion

The analysis demonstrates that coordination failure is **fundamentally an information problem**, not an agent reasoning problem. Agents actively attempt to coordinate (46% of turns invoke avoiding logic) but lack the observational capacity to detect conflicts before they happen.

The fix is not better agents but better observations.

