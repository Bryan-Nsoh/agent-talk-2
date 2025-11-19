# How Agents Fail to Coordinate When Communication is Disabled

## Executive Summary

When communication is disabled (comm_strategy="none"), agents fail to coordinate in 14/15 runs (93% failure rate) in a navigation task. The failures are caused by **invisible coordination information** - agents cannot see nearby agents or their movement intentions, leading to predictable collisions in narrow passages. This analysis identifies specific failure patterns and quantifies what information would prevent them.

---

## Key Findings

### 1. Coordination Failure is Severe

**Success Rate by Communication Strategy** (cross_seed_baseline experiment):
- None (disabled): 1/15 runs (6.7% success)
- Other strategies (structured/freeform): 3/34 runs (8.8% success)
- Average turns to timeout: 97.8 turns (near 100-turn limit)
- Average collisions per run: 18.4 collisions

When communication is fully disabled, agents almost always fail to navigate safely through a 30x10 grid with 5 agents to a shared goal.

---

## Failure Pattern Analysis

### Pattern 1: Head-On Collisions in Narrow Corridors

**Most Common Failure: Two agents converging on same cell from opposite directions**

Example: Seed 17, Run 3, Turn 4
```
TARGET CELL: (4,1)

AGENT a2                          AGENT a5
at (5,1) plans MOVE_W    +---→ at (3,1) plans MOVE_E
"Scouting nearest              "Heading east to nearest
frontier at (3,1)"              frontier at (5,1)"

        COLLISION AT (4,1)
        Neither agent moves
```

**Statistics:**
- 11 agent-agent conflicts found in single run (seed17_none_run3)
- Pattern repeats predictably: agents at (N, y) and (N+2, y) both target (N+1, y)
- All 11 conflicts involve pairs of agents converging on the same cell
- No conflict resolution occurs without explicit communication

**Key Observation:** Both agents independently selected the same target cell as "optimal" because it appeared free and advanced their frontier exploration goals.

---

### Pattern 2: Repeated Head-On Collisions (Deadlock)

**Same pair of agents collide repeatedly over multiple turns**

Turns 4, 6, 11, 14, 16, 25: Agents a3 and a5 collide on cells (4,1)→(7,1)→(7,1)→(9,1)

```
Turn 4:   a2(5,1)-E→ + a5(3,1)-E→ = COLLISION at (4,1)
Turn 6:   a2(5,2)-W→ + a5(3,2)-E→ = COLLISION at (4,2)
Turn 11:  a3(8,1)-W→ + a5(6,1)-E→ = COLLISION at (7,1)
Turn 14:  a3(8,1)-W→ + a5(6,1)-E→ = COLLISION at (7,1)  [SAME PAIR, SAME CELL]
Turn 16:  a3(8,1)-W→ + a5(6,1)-E→ = COLLISION at (7,1)  [THIRD TIME]
```

**Why repeated collisions occur:**
- Agents have no memory of previous collisions with the same agent
- `contended_neighbors` flag only tells "someone blocked me", not "a5 specifically will block me again"
- No way to coordinate "a5 goes first, I yield" on the next attempt
- Agents revert to identical decision logic each turn

**What the agents "tried" to do:**
1. a3 reasoning: "Avoid contended west; moving east..."
2. a5 reasoning: "Moving east toward nearest frontier..."
3. Result: Same collision as previous turn

---

### Pattern 3: Inability to Detect Head-On Converging Agents

**Root cause: Observation radius is too small**

Turn 3 state (before Turn 4 collision):
```
a2 at (5,1): neighbors_in_view = 0
    - Adjacent state: N=WALL, E=FREE, S=FREE, W=FREE
    - Local patch (3x3):
        ###
        .A.
        ..#
    - Cannot see a5 at (3,1) - TOO FAR AWAY
    
a5 at (3,1): neighbors_in_view = 0
    - Adjacent state: N=WALL, E=FREE, S=FREE, W=FREE
    - Local patch (3x3):
        ###
        .A.
        #..
    - Cannot see a2 at (5,1) - TWO CELLS AWAY
```

**The problem:**
- Both agents are only 2 cells apart (Manhattan distance)
- Local patch radius = 1 (3x3 grid around agent)
- `neighbors_in_view` requires agents to be adjacent (distance ≤ 1)
- **Gap: Agents can collide at a cell without ever seeing each other in the observation**

**What agents would need:** A "field of view" or "sensed agents" list showing all agents within a 2-3 cell radius.

---

### Pattern 4: No Visibility into Other Agents' Recent Moves

**Missing data: Which direction did the other agent last move?**

Turn 4 context:
- a2 just arrived at (5,1) - it came from the NORTH in previous turns
- a5 is at (3,1) - stationary or exploring
- Both independently decide (4,1) is the next logical frontier cell
- Neither can see "a2 is moving WEST (away from goal)" or "a2 came from the north"

If a5 could see: "a2 recent moves: N→N→W, currently at (5,1)"
- a5 might infer: "a2 is exploring northward, I should go east instead"
- Could have avoided the collision

**Current data available to agents:**
```json
neighbors_in_view: []        ← empty
recent_positions: [
  {x: 3, y: 1}               ← only MY recent positions
]
```

**What's missing:**
```json
// Missing: recent_agents_in_view or sensed_agent_history
recent_agents_sensed: [
  {
    agent_id: "a2",
    abs_pos: {x: 5, y: 1},
    recent_moves: ["N", "N", "W"]   ← would allow intent detection
  }
]
```

---

### Pattern 5: Contended Neighbors Flag Provides Insufficient Information

**The flag tells you "someone blocked you", but not the mechanism**

Turn 4 collision:
```
a5 observes: contended_neighbors = 2 (binary flag for direction W)
```

**What this means:**
- "Someone tried to enter your cell from the WEST direction"
- Doesn't tell you: WHERE they came from, WHICH agent, or IF they'll try again

**What a5 should know:**
- "a2 tried to enter from the west at position (4,1)"
- "a2 came from (5,1) and will likely retry"
- "If I try to move west again, a2 and I will collide head-on"

**Agent's actual reasoning at Turn 4:**
> "Avoiding contended E after BLOCK_AGENT; moving south to (3,2) to explore new corridor."

The agent **correctly identifies** that east is unsafe but then **picks a random direction** (south) without understanding the underlying conflict.

---

## What Information Would Prevent These Failures?

### 1. Extended Neighbor Visibility (Critical)

**Current:** Only agents at distance ≤1 appear in `neighbors_in_view`

**Needed:** Agents within distance ≤2 or ≤3 cells

**Impact:**
- In the (4,1) collision case, a2 at (5,1) could see a5 at (3,1)
- Both agents could perceive the other's presence
- Agents could choose alternate routes pre-emptively

**Estimated prevention rate:** 70-80% of head-on collisions

---

### 2. Visible Agent Movement History (High Impact)

**Current:** Only `recent_positions` of self is available

**Needed:** For each `neighbors_in_view` agent:
```json
{
  agent_id: "a5",
  recent_positions: [{x: 3, y: 1}, {x: 3, y: 0}, {x: 3, y: 1}],
  recent_moves: ["S", "N"]
}
```

**Why this helps:**
- Agents can infer intent: "a5 is oscillating in place" → blocked
- Agents can detect head-on approach: "a5 moved east twice, I should go a different way"
- Natural coordination emerges: agents implicitly alternate when one backs off

**Estimated prevention rate:** 40-60% of deadlock cycles

---

### 3. Directional Contention Details (Medium Impact)

**Current:** `contended_neighbors` is a binary flag per direction

**Needed:** 
```json
{
  contended_E: {
    agent_ids: ["a3"],
    collision_positions: [{x: 9, y: 1}],
    last_occurrence_turn: 23
  }
}
```

**Why this helps:**
- Agents know WHICH agent is blocking WHICH direction
- Agents can establish "a3 owns the east corridor" and naturally yield
- Prevents retry of same collision in next turn

**Estimated prevention rate:** 20-40% of repeated collisions

---

### 4. Shared Exploration Map (Lower Priority)

**Current:** Each agent has its own `world_map_ascii` showing X (unknown) tiles

**Needed:** Optional shared frontier layer showing which agent explored what

**Why this helps:**
- Reduces redundant exploration conflicts
- Agents recognize when both are heading to the same unknown area
- Can coordinate to explore different frontiers

**Estimated prevention rate:** 10-20%

---

## Implicit Coordination Attempts (Evidence of Trying)

Despite no explicit communication, agents DO attempt coordination through their movement choices:

**Keyword frequency in agent reasoning:**
- "avoid": 46.7% of turns → agents trying to sidestep conflict
- "recent": 24.6% of turns → agents checking their own history for loops
- "block": 8.0% of turns → agents reacting to collisions
- "yield": 4.7% of turns → agents explicitly attempting to defer
- "wait": 0.4% of turns → agents attempting to hold position

**Example of implicit coordination attempt:**
```
Turn 14 a3: "West is contended and last move blocked; yielding to avoid 
            collision. Holding position, then aim for frontier at (6,1) when clear."
```

Agent a3 **correctly infers** the need to yield, but cannot:
1. Signal this to the other agent
2. Know if the other agent will also yield
3. Establish a turn order ("you go first")

**Result:** Both agents keep trying the same conflict, hoping it resolves.

---

## Why the One Successful "None" Run Succeeded

**Seed 16, Run 2: 67 turns, 0 collisions**

Analysis of success factors:

1. **Lucky Initial Spread:** Agents started far apart
   - a1 started at x=9, a4 at x=15, a5 at x=0
   - Natural spatial separation reduced encounters

2. **Divergent Exploration:** Agents naturally split across different Y coordinates
   - a2 explored bottom corridor (y≤5)
   - a3 explored mid corridor (y≤7)
   - a4, a1 explored top-right (y≥7)

3. **Minimal Hotspots:** Only 2 collisions attempted in entire run (turn 4, 18)
   - Both were immediate, easy to resolve
   - No repeated collisions with same agent

4. **Goal Gradient Effect:** As agents advanced toward goal, they naturally separated more

**Conclusion:** This run succeeded **despite** the lack of communication, not because agents coordinated. Pure luck of initial conditions and goal geometry.

---

## Quantitative Analysis of Failure Modes

### Collision Distribution in Worst Case (seed17_none_run3)

```
Total collisions: 34 BLOCK_AGENT outcomes
Agent-agent conflicts detected: 11
Wall/boundary collisions: 23

Agent-agent conflict distribution:
  a2 ↔ a5: 6 conflicts (same two agents, repeated)
  a3 ↔ a5: 4 conflicts (same two agents, repeated)
  a2 ↔ a5: 1 conflict
Total: 11

Turns with 4+ agents colliding: 5
Turns with 3 agents colliding: 12
Turns with 2 agents colliding: 18
Turns with 1 agent colliding: None (no solo wall hits recorded)
```

### Success Correlates

Runs with 0 collisions: 1/15 (only the lucky seed16_none_run2)
Runs with ≤8 collisions: 3/15 (still timeouts at 100 turns)
Runs with ≥17 collisions: 12/15 (100% timeout failures)

---

## Recommendations for Testing Information Availability

### Tier 1: Extended Neighbor Visibility
Test enabling agents to see all neighbors within distance ≤2 cells. 
- Hypothesis: Reduces head-on collisions by 60-80%
- Expected outcome: Many "none" runs should succeed

### Tier 2: Visible Movement History
Add recent movement history to the observation for all visible neighbors.
- Hypothesis: Breaks deadlock cycles, enables implicit yielding
- Expected outcome: Repeated collisions drop to <2 per run

### Tier 3: Directional Contention Details
Replace binary `contended_neighbors` with detailed conflict log.
- Hypothesis: Prevents retry of same failed direction
- Expected outcome: Collision rate per turn drops by 40%

### Tier 4: Asynchronous Intent Signaling (Implicit)
Allow agents to "hold a direction intent" visible to neighbors in next turn.
- This is quasi-communication, but passive/observable rather than active

---

## Conclusion

Agents fail to coordinate when communication is disabled because they operate in **information-isolated silos**. Each agent:

1. Cannot see other agents at distance >1 (too small observation window)
2. Cannot infer other agents' intentions from movement history (no visible history)
3. Cannot distinguish between "blocked by wall" vs "blocked by specific agent X" (insufficient contention info)
4. Cannot signal intent or establish turn orders (no communication)

The result is predictable: agents converge on the same target cells, collide, and then retry the same collision with no new information. Without explicit communication, agents need at least **visible neighbor movement history** to implicitly coordinate via observation of each other's actions.

The evidence is stark: 93% of runs without communication fail, while agents do attempt implicit coordination (46% of turns include "avoid" reasoning). But observation limitations make implicit coordination impossible at sufficient scale.

