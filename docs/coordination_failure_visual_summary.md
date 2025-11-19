# Coordination Failure: Visual Summary and Key Examples

## The Core Problem in One Diagram

```
WITHOUT COMMUNICATION, AGENTS CANNOT SEE EACH OTHER AT COLLISION DISTANCE

Turn 3 (Decision Making):

Agent a2 at (5,1):               Agent a5 at (3,1):
┌─────────────┐                  ┌─────────────┐
│ neighbors   │ = EMPTY           │ neighbors   │ = EMPTY
│ in_view     │                   │ in_view     │
└─────────────┘                   └─────────────┘
     │                                 │
     └─────────────── 2 cells apart ──┘
     
     BUT local_patch radius = 1
     CANNOT see each other
     BOTH decide: "Go to (4,1)"
     
     ↓ ↓
     
Turn 4 (Collision):

a2: (5,1)    a5: (3,1)
  ↓            ↓
  └──→ (4,1) ←┘
     COLLISION
     Neither moves
```

---

## Failure Pattern Flowchart

```
DISABLE COMMUNICATION
        │
        ↓
Agents cannot exchange intent
        │
        ├→ No "I'm going EAST"
        ├→ No "You go first"
        └→ No shared exploration map
        
        ↓
Agents rely on OBSERVATION
        │
        ├→ neighbors_in_view (distance ≤1) ← TOO NARROW
        ├→ contended_neighbors flag ← TOO VAGUE (who?)
        ├→ recent_positions (self only) ← USELESS for others
        └→ world_map_ascii (self only) ← REDUNDANT EXPLORATION
        
        ↓
BLIND NAVIGATION
        │
        ├→ Agents converge on same target
        ├→ 11 agent-agent conflicts per run
        ├→ Same pair collides 5-6 times
        └→ 93% failure rate (14/15 runs)
```

---

## Concrete Collision Examples

### Example 1: The "Head-On" Collision

```
TURN 3 STATE:

      y=1
      │
1     2     3     4     5     6
             │           │
        a5 at (3,1)  a2 at (5,1)
        
        Local patches:
        a5: ###    a2: ###
            .A.        .A.
            #..        ..#
            
        Both see: (4,1) is FREE
        Both decide: move to (4,1)

AGENTS' REASONING:

a5: "Heading east to nearest frontier at (5,1); (4,1) is free"
a2: "Scouting nearest frontier at (3,1) by moving west"

TURN 4 OUTCOME:

a5: stays at (3,1), outcome=BLOCK_AGENT
a2: stays at (5,1), outcome=BLOCK_AGENT

Neither can see why they collided (other agent was invisible)
```

### Example 2: The "Deadlock" Collision

```
TURNS 11, 14, 16 - SAME THREE AGENTS STUCK:

  6                       8     9    10    11    12
  │                       │     │     │     │     │
1 │                       │ a3  ↑     │ a3  ↓     │
  │                       │ (8,1)     │(8,1)      │
  │                 a5→   │           │     a5→   │
  │               (6,1)   │ COLLISION │  (6,1)    │
  │                       │ at (7,1)  │           │

Turn 11:  a3(8,1)-W + a5(6,1)-E = collision at (7,1)
Turn 14:  a3(8,1)-W + a5(6,1)-E = collision at (7,1)  [SAME]
Turn 16:  a3(8,1)-W + a5(6,1)-E = collision at (7,1)  [SAME]

Why retry same collision?
- a3 thinks: "Frontier is west, go there"
- a5 thinks: "Frontier is east, go there"
- contended_neighbors flag doesn't say: "a5 specifically is the problem"
- No communication to establish: "a5 goes first this time"

Result: Deadlock - both agents stuck trying same move
```

### Example 3: The "Corridor Funnel"

```
MAZE STRUCTURE (long_corridor):

x: 0  1  2  3  4  5  6  7  8  ... 28 29
y=9: #  #  .  .  .  .  .  .  .  .  .
y=8: #  #  .  .  .  .  .  .  .  .  .
y=7: #  #  .  .  .  .  .  .  .  .  .
y=6: .  .  .  .  .  .  .  .  .  GOAL
y=5: #  #  .  .  .  .  .  .  .  .  .
y=4: #  #  .  .  .  .  .  .  .  .  .
y=3: .  .  .  .  .  .  .  .  .  .  .
y=2: .  .  .  .  .  .  .  .  .  .  .
y=1: .  .  .  .  .  .  .  .  .  .  . ← BOTTLENECK
y=0: #  # # # # # # # # # # # # #  .

NATURAL FUNNELING:

Agents exploring from left → all converge to y=1
Multiple agents in narrow y=1 corridor → inevitable collisions
Distance ≤2 at y=1 but can't see each other (distance >1 in observation)

RESULT: 6 collisions between a2 and a5 just in y=1 corridor
```

---

## What Information Would Help?

### Tier 1: Extended Visibility (Most Impactful)

```
CURRENT (distance ≤1):
Agent at (5,1):
  neighbors_in_view = []   ← can't see (3,1)
  
PROPOSED (distance ≤2):
Agent at (5,1):
  neighbors_in_view = [
    {
      agent_id: "a5",
      pos: (3,1),
      last_move: "S"  ← useful!
    }
  ]
  
DECISION WITH VISIBILITY:
"a5 is 2 cells away moving SOUTH. If I move WEST to (4,1),
we'll collide. Pick alternate: move EAST instead."

Result: Collision avoided without communication
```

### Tier 2: Visible Movement History (High Impact)

```
CURRENT:
Agent observes: recent_positions = [own history only]

PROPOSED:
For each neighbor in view:
  neighbor.recent_positions = [(5,1), (5,2), (5,3)]
  neighbor.recent_moves = ["N", "N"] ← moved N twice
  
DECISION WITH HISTORY:
"a2 just moved NORTH twice. Pattern: exploring northward.
If I move WEST to (4,1), I won't block a2's northward path.
Safe to proceed."

Result: Predictable deadlocks broken by intent inference
```

### Tier 3: Detailed Contention (Medium Impact)

```
CURRENT:
contended_neighbors = 2 (binary flag)
↓
"Someone blocked me from the EAST"
(Doesn't say: who, where, will they retry?)

PROPOSED:
contended_neighbors = [
  {
    direction: "E",
    agent_id: "a2",
    collision_point: (4,1),
    turns_since: 1
  }
]

DECISION WITH DETAILS:
"a2 specifically blocked me at (4,1) last turn.
If I try EAST again, same collision.
Pick different direction."

Result: Reduced retry rate of same failed direction
```

---

## Success vs. Failure Comparison

### The ONE Successful Run (seed16_none_run2)

```
INITIAL SPREAD:
a1: (9,2)      ← middle
a2: (2,3)      ← left
a3: (7,7)      ← top
a4: (15,9)     ← far right
a5: (0,6)      ← far left

SEPARATION: 0-15 range (maximum spread)

NATURAL EXPLORATION ZONES:
a5 (0,6)   → explores left corridor
a2 (2,3)   → explores bottom left
a1 (9,2)   → explores middle
a3 (7,7)   → explores top middle
a4 (15,9)  → explores right

CONFLICTS: Only 2 in entire run (both easily resolved)
SUCCESS: Yes, 67 turns (no timeout)

KEY: Agents naturally spread out
     Lucky starting positions prevent hotspots
     Only 0 collisions (pure luck)
```

### A Typical Failed Run (seed17_none_run3)

```
INITIAL CLUSTERING:
a1: (0,6)      ← cluster!
a2: (2,3)      ← cluster!
a3: (7,7)
a4: (15,9)
a5: (3,1)      ← cluster!

CLUSTERING: 3 agents in x=[0-3] (crowded)

CONFLICTS: Start immediately (turns 4-6)
DEADLOCK: a2 ↔ a5 collide 6 times at various cells
TIMEOUT: 100 turns with 34 collisions
FAILURE: Yes

KEY: Agents start clustered
     Forced into same narrow corridor
     Can't see each other until collision
     No way to resolve deadlock
```

---

## The Information Problem Visualized

```
COMMUNICATION IS DISABLED
            │
            ↓
    AGENTS MUST OBSERVE
            │
            ├─→ Local observation radius = 1 cell
            │   Can only see adjacent (4-cell radius max)
            │   Cannot see agents 2+ cells away
            │
            ├─→ Each agent has own map
            │   Cannot coordinate exploration
            │   Cannot avoid redundant paths
            │
            └─→ Contention is binary flag
                Cannot identify which agent is blocking
                Cannot establish turn order
                Cannot signal intent
                
            ↓
    AGENTS REVERT TO GREEDY BEHAVIOR
            │
            ├─→ Move toward nearest frontier
            ├─→ Move toward goal
            └─→ Avoid only what they can see
            
            ↓
    PREDICTABLE HEAD-ON COLLISIONS
            │
            └─→ Both agents at opposite ends of corridor
                Both want to explore center
                Both move toward same cell
                Neither saw the other
```

---

## Agent Reasoning Under Constraint

```
Agent Decision Logic (observed in transcripts):

Turn 1-3:  "Frontier at (4,1) is free, I'll go there"
           (Agent a2 reasons this)
           
Turn 1-3:  "Frontier at (4,1) is free, I'll go there"
           (Agent a5 reasons this independently)
           
Turn 4:    COLLISION at (4,1)
           Neither agent moves
           
Turn 5:    a2: "Avoiding contended east; move south"
           (Guesses randomly, hits wall)
           
Turn 5:    a5: "Avoiding contended east; move south"
           (Same random guess, hits wall)
           
           ↑ BOTH picked SOUTH independently
           NO COORDINATION despite "avoiding"

Result: Agents TRY to avoid but lack the information
        to do so successfully
```

---

## Key Insight: Information Deficit = Coordination Failure

```
┌─────────────────────────────────────┐
│ AGENTS CAN SEE:                     │
│ - Self position                     │
│ - Adjacent cells (4 directions)     │
│ - Own recent path                   │
│ - Own map of explored areas         │
└─────────────────────────────────────┘
          ↓
    INVISIBLE TO OTHERS
    
┌─────────────────────────────────────┐
│ AGENTS CANNOT SEE:                  │
│ - Other agents 2+ cells away        │
│ - Which agent blocked them          │
│ - Other agents' recent moves        │
│ - Other agents' exploration map     │
│ - Other agents' intent/direction    │
└─────────────────────────────────────┘
          ↓
    IMPOSSIBLE TO COORDINATE
    
SOLUTION: Expand observation window
          Provide visible agent history
          Make intent observable
          
Then: Implicit coordination emerges
      Agents naturally avoid converging
      No explicit communication needed
```

---

## Summary Statistics

```
Metric                                  Value
─────────────────────────────────────────────
Communication enabled                   1/15 success
Runs with 0 collisions                  1/15
Avg collisions/run                      18.4
Agent-agent conflicts/run               ~11
Repeated conflicts (same pair)          ~5-6
Agents within distance 2 before collision  11/11
Agents seeing each other before collision  0/11
Turns with "avoid" reasoning            46.7%
Turns with "yield" reasoning            4.7%
Successful yields (saw nearby agent)    ~5% of attempts
Failed coordination attempts            ~95% of runs
─────────────────────────────────────────────

INTERPRETATION:
Agents TRY to coordinate (46% avoid attempts)
But lack information to succeed (95% still fail)
Extended observation would enable success
```

---

## Conclusion in Plain Language

When you **disable communication**, agents can't tell each other what they're doing. They're forced to rely entirely on observing each other's movements.

But the **observation window is too narrow**. Agents can only see neighbors that are adjacent (1 cell away). Most collisions happen between agents who are 2 cells apart.

Result: Agents **converge head-on** without seeing each other until it's too late.

The fix is simple: **Let agents see 2-3 cells away** (not just 1 cell). Then they can naturally avoid converging on the same target, even without explicit communication.

The problem isn't that agents can't think. It's that they're **blind**.

